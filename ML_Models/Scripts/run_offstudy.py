"""Train the runs of the target-transform / split study (see offstudy.py). Resumable: finished runs are skipped.

    python3 ML_Models/Scripts/run_offstudy.py --stage 1 --jobs 2

Stages (all MLP 128-64-32-16, the settings of the training notebooks, seed 12175305):
  1  NMOS_OFF, random split: log target vs signed-log target (I0 = q-quantile of |I|, q searched), +/- Kirchhoff loss
  2  NMOS_OFF, structured splits: log target vs the signed-log target chosen in stage 1 (reads the stage-1 results)
  3  the other three cases, structured splits, log target only (is NMOS_OFF special?)
  4  NMOS_OFF, random split, other training seeds (run-to-run variation; the models also form an ensemble)
  5  NMOS_OFF, random split, larger networks (log target)
  6  NMOS_OFF without early stopping: more seeds on the random split + every structured split, log vs signed-log
  7  PMOS_OFF structured splits without early stopping (log target)
  8  NMOS_ON and PMOS_ON structured splits, early stopping as in the notebooks (log target)
  9  NMOS_OFF, T_interp and W_interp with two more training seeds (log target, no early stopping)
  10 NMOS_OFF, random split: robust (Huber) loss, median-based epoch selection, cosine schedule, learning curve
  11 Huber loss follow-up: transition width, the other three cases, + Kirchhoff term, + cosine decay
  12 smaller Huber transition widths (0.005, 0.002)
  13 final Huber models of the four cases (2 seeds each; candidates in Saved_Models/huber/, then
     export_huber_models.py installs the better one per case as the final Saved_Models/<case>.pt)
Results go to ML_Models/Results/offstudy/{runs,preds}/.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offstudy as o                                   # noqa: E402

Q_GRID = [0.001, 0.01, 0.05, 0.2]


def _work(job):
    import torch
    torch.set_num_threads(3)
    res = o.run_one(**job)
    a = res["val"]["average"]
    print(f"{res['tag']:60s} epochs {res['epochs']:4d}  {res['seconds']:5.0f}s  "
          f"val mean/median/95th {a['mean err %']:8.2f} {a['median err %']:5.2f} {a['95th pct err %']:6.2f}  "
          f"test mean/median/95th {res['test']['average']['mean err %']:8.2f} "
          f"{res['test']['average']['median err %']:5.2f} {res['test']['average']['95th pct err %']:6.2f}", flush=True)
    return res["tag"]


def best_q(case="NMOS_OFF", lam=0.0):
    """q with the lowest validation mean error among the finished stage-1 signed-log runs (same rule as the notebooks)."""
    scores = {}
    for q in Q_GRID:
        p = o.ROOT / "ML_Models/Results/offstudy/runs" / f"{o.run_tag(case, 'random', 'signedlog', q, lam, o.SEED)}.json"
        if p.exists():
            scores[q] = json.loads(p.read_text())["val"]["average"]["mean err %"]
    return min(scores, key=scores.get)


def best_huber_beta(case="NMOS_OFF", grid=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002), seeds=(0, 1)):
    """Huber transition width with the lowest validation median error (mean over the seeds that finished all runs)."""
    scores = {}
    for b in grid:
        vals = []
        for k in seeds:
            tag = o.run_tag(case, "random", "log", 0.01, 0.0, o.SEED + k, patience=3000, loss_kind="huber", huber_beta=b)
            p = o.ROOT / "ML_Models/Results/offstudy/runs" / f"{tag}.json"
            if p.exists():
                vals.append(json.loads(p.read_text())["val"]["average"]["median err %"])
        if len(vals) == len(seeds):
            scores[b] = sum(vals) / len(vals)
    return min(scores, key=scores.get)


def jobs_for(stage):
    if stage == 1:
        jobs = [dict(case="NMOS_OFF", scheme="random", target="log", lam=0.0),
                dict(case="NMOS_OFF", scheme="random", target="log", lam=10.0)]
        jobs += [dict(case="NMOS_OFF", scheme="random", target="signedlog", q=q, lam=0.0) for q in Q_GRID]
        return jobs
    if stage == 2:
        q = best_q()
        print(f"stage 2 uses signed-log with q = {q} (best validation mean error in stage 1)", flush=True)
        jobs = []
        for s in o.SCHEMES[1:]:
            jobs.append(dict(case="NMOS_OFF", scheme=s, target="log", lam=0.0))
            jobs.append(dict(case="NMOS_OFF", scheme=s, target="signedlog", q=q, lam=0.0))
        return jobs
    if stage == 3:                                      # (patience 300; superseded by stages 7 and 8)
        return [dict(case=c, scheme=s, target="log", lam=0.0)
                for c in ("PMOS_OFF", "NMOS_ON", "PMOS_ON") for s in o.SCHEMES[1:]]
    if stage == 4:                                      # run-to-run variation (training seed only; the split is fixed)
        q = best_q()
        jobs = [dict(case="NMOS_OFF", scheme="random", target="log", lam=0.0, seed=o.SEED + k) for k in range(1, 5)]
        jobs += [dict(case="NMOS_OFF", scheme="random", target="signedlog", q=q, lam=0.0, seed=o.SEED + k)
                 for k in range(1, 3)]
        return jobs
    if stage == 5:                                      # does a bigger network (more base accuracy) shrink the tail?
        return [dict(case="NMOS_OFF", scheme="random", target="log", lam=0.0, hidden=h)
                for h in ((256, 128, 64, 32), (256, 256, 128, 64))]
    # stages 6 - 8: no early stopping (patience = 3000 = max_epochs: every run trains 3000 epochs, best validation epoch kept),
    # because under patience 300 many runs stall at 400 - 900 epochs and the stalled runs, not the method or the split, decide the result
    full = dict(lam=0.0, patience=3000)
    if stage == 6:
        q = best_q()
        jobs = [dict(case="NMOS_OFF", scheme="random", target="log", seed=o.SEED + k, **full) for k in (1, 2, 3)]
        jobs += [dict(case="NMOS_OFF", scheme="random", target="signedlog", q=q, seed=o.SEED + k, **full) for k in (0, 1)]
        for s in o.SCHEMES[1:]:
            jobs.append(dict(case="NMOS_OFF", scheme=s, target="log", **full))
            jobs.append(dict(case="NMOS_OFF", scheme=s, target="signedlog", q=q, **full))
        return jobs
    if stage == 7:
        return [dict(case="PMOS_OFF", scheme=s, target="log", **full) for s in o.SCHEMES[1:]]
    if stage == 8:       # the ON cases do not stall (their notebook models stopped early at good solutions): patience 300 as in the notebooks
        return [dict(case=c, scheme=s, target="log", lam=0.0) for c in ("NMOS_ON", "PMOS_ON") for s in o.SCHEMES[1:]]
    if stage == 9:       # is "temperature gaps cost more than width gaps" a seed effect? two more seeds of both interpolation splits
        return [dict(case="NMOS_OFF", scheme=s, target="log", seed=o.SEED + k, **full)
                for s in ("T_interp", "W_interp") for k in (1, 2)]
    if stage == 10:      # learning curve of the Huber model: how much does more training data help? (seed 12175305, 3000 epochs)
        return [dict(case="NMOS_OFF", scheme="random", target="log", lam=0.0, patience=3000, loss_kind="huber",
                     huber_beta=best_huber_beta(), seed=o.SEED, n_train=n) for n in (4000, 8000, 12000)]
    if stage == 11:      # follow-up of the Huber loss: transition width, other cases, with cosine decay, with the Kirchhoff term
        hub = dict(target="log", patience=3000, loss_kind="huber")
        s5, s6 = o.SEED, o.SEED + 1
        jobs = [dict(hub, case="NMOS_OFF", scheme="random", lam=0.0, seed=sd, huber_beta=b) for b in (0.02, 0.2, 0.01, 0.1) for sd in (s5, s6)]
        jobs += [dict(hub, case=c, scheme="random", lam=0.0, seed=s5) for c in ("PMOS_OFF", "NMOS_ON", "PMOS_ON")]
        jobs += [dict(hub, case="NMOS_OFF", scheme="random", lam=10.0, seed=s5)]
        jobs += [dict(hub, case="NMOS_OFF", scheme="random", lam=0.0, seed=sd, sched_kind="cosine") for sd in (s5, s6)]
        return jobs
    if stage == 12:      # smaller Huber transition widths (the trend of stage 11 had not turned)
        hub = dict(case="NMOS_OFF", scheme="random", target="log", patience=3000, loss_kind="huber", lam=0.0)
        return [dict(hub, seed=sd, huber_beta=b) for b in (0.005, 0.002) for sd in (o.SEED, o.SEED + 1)]
    if stage == 13:      # final models: all four cases, Huber loss with the width chosen on validation, weights saved
        b = best_huber_beta()
        out = o.ROOT / "ML_Models" / "Results" / "offstudy" / "final"
        (o.ROOT / "ML_Models" / "Saved_Models" / "huber").mkdir(parents=True, exist_ok=True)
        print(f"stage 13 uses huber_beta = {b}", flush=True)
        return [dict(case=c, scheme="random", target="log", lam=0.0, patience=3000, loss_kind="huber", huber_beta=b, seed=o.SEED + k,
                     out_dir=out, save_model=o.ROOT / "ML_Models" / "Saved_Models" / "huber" / f"{c}__seed{o.SEED + k}.pt")
                for c in ("NMOS_OFF", "PMOS_OFF", "NMOS_ON", "PMOS_ON") for k in (0, 1)]
    raise ValueError(stage)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    jobs = jobs_for(args.stage)
    print(f"stage {args.stage}: {len(jobs)} runs, {args.jobs} at a time", flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(_work, jobs))
    print("done", flush=True)
