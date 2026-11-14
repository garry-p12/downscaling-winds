"""Residual diffusion on top of a deterministic wind model: realistic fine-scale texture plus a calibrated ensemble.

Path B of WIND_WORK_LOG 7c/7d. The deterministic model (by default the station-fusion runs, path A) gives the best
point estimate; what it cannot place it shrinks (sd ratio ~0.5 below 100 km). This learns the distribution of the
RESIDUAL (AORC anomaly - deterministic anomaly), conditioned on the base run's own input channels plus the
deterministic field, with the EDM denoiser of models/diffusion.py (the temperature program's, unchanged). A sample is
deterministic + residual draw; the ensemble mean tends back to the deterministic field.

Trained on the store of the base runs (fold 0) with the holdout masked out of the loss, sampled on the 2023 test days
on BOTH fold stores (so stations are scored only where they were not inputs), then scored:

  grid (holdout, vs AORC)   member and ensemble-mean RMSE; per band (<100, 100-300, >300 km) sd ratio and r2;
                            CRPS (ensemble, fair estimator) vs the deterministic MAE; spread / skill
  stations (withheld ISD, Mesonet)  CRPS vs deterministic MAE, coverage of the central 80% interval

Usage:
    DOWNSCALE_DEVICE=cuda python scripts/wind_diffusion.py --epochs 60          # train + sample + score
    python scripts/wind_diffusion.py --sample-only                               # resume from checkpoint
    python scripts/wind_diffusion.py --score-only [--center]                      # -> <tag>[_centered]_score.json
    DOWNSCALE_DEVICE=cuda python scripts/wind_diffusion.py --train-split val --epochs 400 --tag obsval
    python scripts/wind_diffusion.py --base w10_L_s1337 w10_L_s7 w10_L_s42 --tag nostation   # on the POWER-only model
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")
from data.build_dataset import NO_NORMALIZE                                      # noqa: E402
from models.diffusion import EDM, ResidualDenoiser                               # noqa: E402
import wind_inference_maps as wim                                                 # noqa: E402
import wind_obs_eval as woe                                                       # noqa: E402
import wind_plan                                                                  # noqa: E402

OUT = REPO / "outputs" / "wind" / "diffusion"


def log(*a):
    print(*a, flush=True)


def fold_tags(base_store):
    """Stores to sample on: both fold stores for the fusion runs, the store itself otherwise."""
    return ["w10o0", "w10o1"] if base_store.startswith("w10o") else [base_store]


def build(tag_store, base, channels, days=None):
    """(cond, det, truth anomaly) on one store; cond = normalised base channels + deterministic field."""
    st = wim.store_for(tag_store)
    n = st.ds.sizes["time"]
    days = np.arange(n) if days is None else days
    det = np.mean([wim.predict(st, b, days) for b in base], axis=0).astype("float32")
    z = st._zarr
    idx = [st.stored.index(c) for c in channels]
    cond = np.empty((len(days), len(channels) + 1, *st.shape), "float32")
    for b0 in range(0, len(days), 128):
        sl = days[b0:b0 + 128]
        blk = np.asarray(z["input"][sl[0]:sl[-1] + 1], "float32")[sl - sl[0]][:, idx]
        for c, nm in enumerate(channels):
            if nm not in NO_NORMALIZE:
                blk[:, c] = st.nz.transform(f"in::{nm}", blk[:, c])
        cond[b0:b0 + len(sl), :len(channels)] = blk
    cond[:, -1] = det
    np.nan_to_num(cond, copy=False)
    T = np.asarray(z["target"][days[0]:days[-1] + 1, 0], "float32")[days - days[0]]
    return st, cond, det, T


def crps_ens(X, y):
    """Fair ensemble CRPS, X (m, N), y (N). Mean over N."""
    m = X.shape[0]
    t1 = np.mean(np.abs(X - y[None]), axis=0)
    Xs = np.sort(X, axis=0)
    i = np.arange(1, m + 1)[:, None]
    t2 = np.sum((2 * i - m - 1) * Xs, axis=0) / (m * (m - 1))           # = sum_{i,j}|Xi-Xj| / (2 m (m-1))
    return float(np.mean(t1 - t2))


def train(a, base, channels, tag_store):
    st, cond, det, T = build(tag_store, base, channels)
    resid = np.nan_to_num(T - det).astype("float32")
    # which days' residuals to learn: "train" (in-sample for the base model, residuals too small) or "val" (2022,
    # out-of-sample for the base model, so the residual spread and drift are realistic)
    tr = np.where(st.split == a.train_split)[0]
    lmask = (st.m_train & np.isfinite(T).all(0)).astype("float32")
    device = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    sigma_data = float(resid[tr][:, lmask > 0].std())
    log(f"store {tag_store}, base {base}, {cond.shape[1]} conditioning channels, residual sd {sigma_data:.4f} m/s")
    net = EDM(ResidualDenoiser(cond_channels=cond.shape[1], width=a.width), sigma_data=sigma_data).to(device)
    ck = REPO / "checkpoints" / f"wind_diffusion_{a.tag}" / "last.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if ck.exists():
        sd = torch.load(ck, map_location=device)
        net.load_state_dict(sd["model"]); start = sd["epoch"] + 1
        log(f"resumed from {ck} at epoch {start}")
    if a.sample_only:
        return net, device
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    P, rng = a.patch, np.random.default_rng(a.seed + start)
    H, W = st.shape
    steps = max(1, len(tr) * a.per_day // a.batch)
    for ep in range(start, a.epochs):
        net.train(); tot = 0.0; t0 = time.time()
        for _ in range(steps):
            d = rng.choice(tr, a.batch)
            i = rng.integers(0, H - P + 1, a.batch); j = rng.integers(0, W - P + 1, a.batch)
            c = np.stack([cond[d[k], :, i[k]:i[k] + P, j[k]:j[k] + P] for k in range(a.batch)])
            y = np.stack([resid[d[k], i[k]:i[k] + P, j[k]:j[k] + P] for k in range(a.batch)])[:, None]
            mk = np.stack([lmask[i[k]:i[k] + P, j[k]:j[k] + P] for k in range(a.batch)])[:, None]
            loss = net.loss(torch.from_numpy(y).to(device), torch.from_numpy(c).to(device),
                            torch.from_numpy(mk).to(device))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); tot += float(loss.detach())
        log(f"epoch {ep + 1:>3}/{a.epochs}  loss {tot / steps:.4f}  {time.time() - t0:.1f}s")
        torch.save({"model": net.state_dict(), "sigma_data": sigma_data, "epoch": ep, "base": base,
                    "channels": channels}, ck)
    return net, device


def sample(a, net, device, base, channels, store_tag):
    st = wim.store_for(store_tag)
    te = st.i_test
    _, cond, det, _ = build(store_tag, base, channels, te)
    land = torch.from_numpy(st.land.astype("float32")).to(device)
    net.eval()
    g = torch.Generator(device=device).manual_seed(a.seed)
    mem = np.empty((a.members, len(te), *st.shape), "float32")
    ct = torch.from_numpy(cond).to(device)
    with torch.no_grad():
        for m in range(a.members):
            outs = []
            for k in range(0, len(te), a.chunk):
                s = net.sample(ct[k:k + a.chunk], steps=a.steps, generator=g)
                outs.append((s[:, 0] * land).cpu().numpy())
            mem[m] = np.concatenate(outs) + det
            if (m + 1) % 5 == 0:
                log(f"  {store_tag}: member {m + 1}/{a.members}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{a.tag}_{store_tag}_test.npy", mem)
    np.save(OUT / f"{a.tag}_{store_tag}_det.npy", det)
    log(f"wrote {OUT / f'{a.tag}_{store_tag}_test.npy'} {mem.shape}")


def score(a, base_store):
    stations = json.loads((REPO / "outputs" / "wind" / "obs_stations.json").read_text())["stations"] \
        if base_store.startswith("w10o") else []
    import pandas as pd
    isd = {}
    if stations:
        d = pd.read_csv(REPO / "outputs" / "wind" / "isd_domain_daily.csv", parse_dates=["date"])
        isd = {k: g.set_index("date")["ws"] for k, g in d.groupby("key")}
    res = {}
    for store_tag in fold_tags(base_store):
        E = np.load(OUT / f"{a.tag}_{store_tag}_test.npy", mmap_mode="r")
        det = np.load(OUT / f"{a.tag}_{store_tag}_det.npy")
        if a.center:                       # members re-centred on the deterministic field: ensemble mean == det
            E = np.asarray(E) - np.asarray(E).mean(0, keepdims=True) + det[None]
        if not store_tag.startswith("w10o"):
            raise SystemExit("scoring is implemented for the fold stores; pass a w10o* base")
        F = woe.Fold(int(store_tag[-1]), stations, isd)
        st, m = F.st, F.mask
        Ta = F.Ta
        r = {}
        r["det"] = {"grid_rmse": float(np.sqrt(np.mean((det - Ta)[:, m] ** 2))),
                    "bands": woe.bands(det, Ta, F.land, m, st)}
        mem0 = np.asarray(E[0])
        r["member"] = {"grid_rmse": float(np.sqrt(np.mean((mem0 - Ta)[:, m] ** 2))),
                       "bands": woe.bands(mem0, Ta, F.land, m, st)}
        mean = np.asarray(E.mean(0))
        r["ens_mean"] = {"grid_rmse": float(np.sqrt(np.mean((mean - Ta)[:, m] ** 2))),
                         "bands": woe.bands(mean, Ta, F.land, m, st)}
        # grid CRPS on a subsample of holdout cells (all days)
        ii, jj = np.where(m)
        sel = np.random.default_rng(0).choice(len(ii), min(400, len(ii)), replace=False)
        Xg = np.asarray(E[:, :, ii[sel], jj[sel]]).reshape(E.shape[0], -1)
        yg = Ta[:, ii[sel], jj[sel]].ravel()
        spread = float(np.sqrt(np.mean(np.var(Xg, axis=0, ddof=1))))
        skill = float(np.sqrt(np.mean((Xg.mean(0) - yg) ** 2)))
        r["grid_crps"] = crps_ens(Xg, yg)
        r["grid_det_mae"] = float(np.mean(np.abs(det[:, ii[sel], jj[sel]].ravel() - yg)))
        r["grid_spread_skill"] = spread / skill
        clim = F.clim
        for net, rows in (("isd", F.eval_isd), ("meso", F.eval_meso)):
            cr, mae, cov = [], [], []
            for _, i, j, o in rows:
                ok = np.isfinite(o)
                if ok.sum() < 100:
                    continue
                X = np.asarray(E[:, ok, i, j]) + clim[ok, i, j][None]
                y = o[ok]
                cr.append(crps_ens(X, y)); mae.append(float(np.mean(np.abs(det[ok, i, j] + clim[ok, i, j] - y))))
                lo, hi = np.percentile(X, [10, 90], axis=0)
                cov.append(float(np.mean((y >= lo) & (y <= hi))))
            if cr:
                r[net] = dict(crps=float(np.mean(cr)), det_mae=float(np.mean(mae)), cover80=float(np.mean(cov)),
                              n=len(cr))
        res[store_tag] = r
        log(f"{store_tag}: {json.dumps(r, default=float)[:600]}")
    (OUT / f"{a.tag}{'_centered' if a.center else ''}_score.json").write_text(json.dumps(res, indent=1, default=float))


def main(a):
    plan = {r["name"]: r for r in wind_plan.plan()}
    base = [b for b in a.base if (REPO / "checkpoints" / b / "last.pt").exists()]
    if not base:
        raise SystemExit(f"no base checkpoints among {a.base}")
    channels = list(plan[base[0]]["channels"])
    base_store = plan[base[0]]["store"]
    if not a.score_only:
        net, device = train(a, base, channels, base_store)
        for s in fold_tags(base_store):
            sample(a, net, device, base, channels, s)
    score(a, base_store)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", nargs="+", default=["w10o0_LO_s1337", "w10o0_LO_s7", "w10o0_LO_s42"])
    p.add_argument("--tag", default="obs")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patch", type=int, default=48)
    p.add_argument("--per-day", type=int, default=4)
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--members", type=int, default=25)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--device", default=os.environ.get("DOWNSCALE_DEVICE", "cuda"))
    p.add_argument("--sample-only", action="store_true")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--train-split", default="train", choices=["train", "val"],
                   help="days whose residuals the diffusion learns (val = 2022, out-of-sample for the base model)")
    p.add_argument("--center", action="store_true", help="score members re-centred on the deterministic field")
    main(p.parse_args())
