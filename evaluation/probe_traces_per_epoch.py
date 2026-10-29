# =========================================================
# EXPANDED FUR HOLDOUT + PROBE TRACES
# The 30 training images are frozen inside the checkpoints, so the
# evaluation set = the original 10 holdout + all new t-prefixed images.
# Asserts no overlap with anything the models were trained on.
# =========================================================
import os, re, glob, random, math
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation

FUR_DIR = "fur"
NOISE   = "noise_probe"
START   = "checkpoints_smashed_scratchfix_aug/smashed_scratchfix_ep8.pth"
ARMS    = {"treatment": "checkpoints_smashed_furfix/smashed_furfix_ep{}.pth",
           "control":   "checkpoints_smashed_control/smashed_control_ep{}.pth"}
EPOCHS  = range(1, 7)
HW, THR, FIRE_AREA = (512, 512), 0.5, 0.001
EXTS = ('.jpg','.jpeg','.jfif','.png','.bmp','.webp')
NEW  = re.compile(r'^t\d')                      # t1, t2, t3, ...

# ---------- rebuild the exact training split, then the expanded holdout ----------
allf = sorted(f for f in os.listdir(FUR_DIR) if f.lower().endswith(EXTS))
new_files = [f for f in allf if NEW.match(f)]
old_files = [f for f in allf if not NEW.match(f)]
print(f"fur/ total {len(allf)} | original {len(old_files)} | new (t-prefixed) {len(new_files)}")
assert len(old_files) == 40, f"expected the original 40, found {len(old_files)} — check the prefix rule"

random.seed(0); np.random.seed(0); torch.manual_seed(0)
paths = [os.path.join(FUR_DIR, f) for f in old_files]
random.shuffle(paths)                            # identical to the training script
k = max(1, int(len(paths)*0.25))
fur_val_orig, fur_train = paths[:k], paths[k:]

FUR_EVAL = fur_val_orig + [os.path.join(FUR_DIR, f) for f in new_files]
trained = {os.path.basename(p) for p in fur_train}
leak = [p for p in FUR_EVAL if os.path.basename(p) in trained]
assert not leak, f"CONTAMINATION: {leak[:5]}"
print(f"trained on {len(fur_train)} | EVALUATION SET n = {len(FUR_EVAL)} "
      f"({len(fur_val_orig)} original holdout + {len(new_files)} new), verified no overlap")

noise = sorted(glob.glob(f"{NOISE}/*.png"))
print(f"noise probe n = {len(noise)}")

# ---------- model + probe ----------
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3", num_labels=2, ignore_mismatched_sizes=True)
    sd = torch.load(path, map_location=DEV, weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    r  = m.load_state_dict(sd, strict=False)
    cls = [x for x in r.missing_keys if "classifier" in x or "decode_head" in x]
    bb  = [x for x in r.missing_keys if x not in cls]
    assert not cls, f"HEAD DID NOT LOAD: {path}"
    assert len(bb) <= 5, f"{len(bb)} backbone keys missing: {path}"
    return m.to(DEV).eval()

@torch.no_grad()
def probe(model, paths):
    rows = []
    for p in paths:
        im = cv2.imread(p)
        if im is None: print("unreadable:", p); continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, HW[::-1]).astype(np.float32)/255.0
        x = torch.from_numpy(((x-MEAN)/STD).transpose(2,0,1)).unsqueeze(0).float().to(DEV)
        lo = F.interpolate(model(x).logits, size=HW, mode="bilinear", align_corners=False)
        pm = torch.softmax(lo,1)[0,1].cpu().numpy()
        area = float((pm >= THR).mean())
        rows.append(dict(file=os.path.basename(p),
                         is_new=bool(NEW.match(os.path.basename(p))),
                         max_prob=float(pm.max()), mean_prob=float(pm.mean()),
                         area=area, fired=int(area > FIRE_AREA)))
    return pd.DataFrame(rows)

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),)*2
    p = k/n; d = 1+z*z/n; c = (p+z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0, c-h), min(1, c+h)

# ---------- sweep every checkpoint ----------
records, per_image = [], []
jobs = [("start", 0, START)] + [(a, e, t.format(e)) for a, t in ARMS.items() for e in EPOCHS]
for arm, ep, path in jobs:
    if not os.path.exists(path): print("MISSING", path); continue
    m = load_model(path)
    for name, ps in [("noise", noise), ("fur_heldout", FUR_EVAL)]:
        df = probe(m, ps); df["arm"], df["epoch"], df["probe"] = arm, ep, name
        per_image.append(df)
        lo, hi = wilson(int(df.fired.sum()), len(df))
        rec = dict(arm=arm, epoch=ep, probe=name, n=len(df),
                   fired=int(df.fired.sum()), fire_rate=df.fired.mean(),
                   ci_lo=lo, ci_hi=hi,
                   max_prob_mean=df.max_prob.mean(), max_prob_min=df.max_prob.min(),
                   max_prob_std=df.max_prob.std(), mean_prob_mean=df.mean_prob.mean(),
                   area_mean=df.area.mean())
        if name == "fur_heldout":                 # split old holdout vs newly collected
            for tag, sub in [("orig", df[~df.is_new]), ("new", df[df.is_new])]:
                if len(sub): rec[f"fire_{tag}"] = f"{int(sub.fired.sum())}/{len(sub)}"
        records.append(rec)
    del m; torch.cuda.empty_cache()
    print(f"done {arm} epoch {ep}")

res = pd.DataFrame(records)
pd.concat(per_image).to_csv("probe_traces_per_image_expanded.csv", index=False)
res.to_csv("probe_traces_summary_expanded.csv", index=False)

for pr in ["noise", "fur_heldout"]:
    print(f"\n===== {pr} =====")
    cols = ["arm","epoch","n","fired","fire_rate","ci_lo","ci_hi","max_prob_mean",
            "max_prob_min","max_prob_std","mean_prob_mean","area_mean"]
    cols += [c for c in ("fire_orig","fire_new") if c in res.columns and pr == "fur_heldout"]
    print(res[res.probe == pr][cols].round(4).to_string(index=False))

# ---------- figure ----------
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
for i, pr in enumerate(["fur_heldout", "noise"]):
    g = res[res.probe == pr]; s0 = g[g.arm == "start"]
    for arm, st, col in [("treatment","s-","#1f4e9c"), ("control","D--","#555")]:
        h = g[g.arm == arm].sort_values("epoch")
        xs = [0]+list(h.epoch)
        ax[i].plot(xs, list(s0.fire_rate)+list(h.fire_rate), st, color=col, lw=2, ms=5,
                   label=f"{arm} fire rate")
        ax[i].plot(xs, list(s0.max_prob_mean)+list(h.max_prob_mean), st[0]+":", color=col,
                   lw=1.2, ms=3, alpha=.8, label=f"{arm} mean max prob")
    ax[i].set_title(f"{pr} (n={int(g.n.iloc[0])})", fontsize=10)
    ax[i].set_xlabel("epoch"); ax[i].set_ylabel("rate / probability")
    ax[i].set_ylim(-.05, 1.08); ax[i].grid(alpha=.25, lw=.6); ax[i].legend(fontsize=7)
plt.tight_layout(); plt.savefig("probe_traces_expanded.png", dpi=200); plt.show()

print("\nWatch fire_orig vs fire_new: if the newly collected fur behaves like the original")
print("holdout, the correction generalises within the family. If the new images fire more,")
print("the fix is narrower than 'fur' and that must be reported.")