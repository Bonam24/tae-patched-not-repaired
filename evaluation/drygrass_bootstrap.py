# =========================================================
# Bootstrap CIs for the dry grass comparison (Table 6 caption).
# Recomputes, from traces/drygrass_seeded_per_image.csv, the 95% interval
# on (treatment - control) mean max probability for each of the four runs:
# 10,000 resamples of the 52 images, paired across arms (the same image
# indices drawn for both arms of a pair).
#
# Published intervals (paper, Table 6 caption):
#   reported run [-0.670, -0.406]   seed 1 [-0.429, -0.173]
#   seed 2       [-0.695, -0.429]   seed 3 [-0.586, -0.328]
# Endpoints are Monte Carlo quantities; a different RNG seed moves them by
# a few thousandths. The assertion below allows 0.01.
# =========================================================
import sys
import numpy as np, pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "traces/drygrass_seeded_per_image.csv"
N_BOOT, SEED = 10_000, 0
PUBLISHED = {
    ("reported_fur", "reported_control"): (-0.670, -0.406),
    ("seed1_fur",    "seed1_control"):    (-0.429, -0.173),
    ("seed2_fur",    "seed2_control"):    (-0.695, -0.429),
    ("seed3_fur",    "seed3_control"):    (-0.586, -0.328),
}

df = pd.read_csv(CSV)
arms = {a: g.sort_values("file").max_prob.to_numpy() for a, g in df.groupby("arm")}
rng = np.random.default_rng(SEED)

print(f"{CSV}: {len(df)} rows, arms {sorted(arms)}")
for (t, c), (plo, phi) in PUBLISHED.items():
    x, y = arms[t], arms[c]
    assert len(x) == len(y) == 52, (t, len(x), len(y))
    point = x.mean() - y.mean()
    idx = rng.integers(0, 52, size=(N_BOOT, 52))
    deltas = x[idx].mean(axis=1) - y[idx].mean(axis=1)   # paired: same idx both arms
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    ok = abs(lo - plo) <= 0.01 and abs(hi - phi) <= 0.01
    print(f"{t:<13} delta {point:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"published [{plo:+.3f}, {phi:+.3f}]   {'OK' if ok else 'MISMATCH'}")
    assert ok, f"{t}: CI does not reproduce the published interval"
print("all four intervals reproduce; none crosses zero")
