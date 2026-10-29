# The full evaluation harness behind Table 3: six-family 280-image frozen
# benchmark + the 49 later fur images + the 52 dry grass images, four
# models, both firing criteria.

# =========================================================
# SIX FAMILY BENCHMARK + POST TRAINING FUR + DRYGRASS, four models
# Frozen benchmark untouched; furextra and drygrass evaluated alongside it.
# =========================================================
!pip install -q transformers==5.13.0

import os, re, glob, math, random, time
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from tqdm.auto import tqdm

SETC = glob.glob("/kaggle/input/**/contextShortcutData", recursive=True)[0]
_fx  = [d for d in glob.glob("/kaggle/input/**/furextra", recursive=True) if os.path.isdir(d)]
FUREXTRA = sorted(_fx, key=len)[-1] if _fx else None      # deepest match: .../furextra/furextra
_dg  = [d for d in glob.glob("/kaggle/input/**/drygrass", recursive=True) if os.path.isdir(d)]
DRYGRASS = sorted(_dg, key=len)[-1] if _dg else None      # deepest match: .../drygrass/drygrass
MINE = "/kaggle/input/datasets/bonamosen"
EXTS = ('.jpg','.jpeg','.jfif','.png','.bmp','.webp')
TAUS = [0.50, 0.85]
WANT = [("baseline", "vanilla"), ("corrected ep8", "scratchfix_ep8"),
        ("fur corrected ep4", "furfix_ep4"), ("control ep4", "control_ep4")]

print("SETC     =", SETC)
print("FUREXTRA =", FUREXTRA)
print("DRYGRASS =", DRYGRASS)
assert DRYGRASS, "drygrass not mounted — add it under Data > Add Input"

found = {os.path.basename(p).replace(".pth", ""): p
         for p in sorted(glob.glob(f"{MINE}/**/*.pth", recursive=True))}
CKPTS, LABEL = {}, {}
for lab, key in WANT:
    hit = [k for k in found if key in k]
    if not hit: print(f"  NOT FOUND: {key}"); continue
    CKPTS[hit[0]] = found[hit[0]]; LABEL[hit[0]] = lab
print("models   :", [LABEL[k] for k in CKPTS])
assert len(CKPTS) == 4, "expected four checkpoints"

BACKBONE, HW = "nvidia/mit-b3", (512, 512)
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device   :", DEV)

def load(n):
    m = SegformerForSemanticSegmentation.from_pretrained(
        BACKBONE, num_labels=2, ignore_mismatched_sizes=True)
    sd = torch.load(CKPTS[n], map_location=DEV, weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    r  = m.load_state_dict(sd, strict=False)
    cls = [x for x in r.missing_keys if "classifier" in x or "decode_head" in x]
    bb  = [x for x in r.missing_keys if x not in cls]
    assert not cls, f"HEAD DID NOT LOAD: {n}: {cls[:4]}"
    assert len(bb) <= 5, f"{len(bb)} backbone keys missing: {n}: {bb[:4]}"
    return m.to(DEV).eval()

@torch.no_grad()
def prob_map(model, rgb):
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, HW[::-1]).astype(np.float32)/255.0
    x = torch.from_numpy(((x-MEAN)/STD).transpose(2,0,1)).unsqueeze(0).float().to(DEV)
    lo = F.interpolate(model(x).logits, size=HW, mode="bilinear", align_corners=False)
    return cv2.resize(torch.softmax(lo,1)[0,1].float().cpu().numpy(), (w,h))

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)

# ---------- reconstruct which fur images were SUPPLIED as negatives ----------
FURDIR = os.path.join(SETC, "fur")
random.seed(0); np.random.seed(0); torch.manual_seed(0)
orig = sorted(f for f in os.listdir(FURDIR)
              if f.lower().endswith(EXTS) and not re.match(r'^t\d', f))
assert len(orig) == 40, f"expected 40 original fur images, found {len(orig)}"
_p = [os.path.join(FURDIR, f) for f in orig]; random.shuffle(_p)
_k = max(1, int(len(_p)*0.25))
SUPPLIED = {os.path.basename(p) for p in _p[_k:]}
print(f"fur      : 40 original, {len(SUPPLIED)} supplied as negatives, {40-len(SUPPLIED)} held out")

# ---------- build the work list once ----------
fams = sorted(d for d in os.listdir(SETC) if os.path.isdir(os.path.join(SETC, d)))
work = []
for fam in fams:
    for p in sorted(glob.glob(f"{SETC}/{fam}/**/*", recursive=True)):
        if p.lower().endswith(EXTS): work.append((fam, p))
if FUREXTRA:
    for p in sorted(glob.glob(f"{FUREXTRA}/**/*", recursive=True)):
        if p.lower().endswith(EXTS): work.append(("fur_new", p))
for p in sorted(glob.glob(f"{DRYGRASS}/**/*", recursive=True)):
    if p.lower().endswith(EXTS): work.append(("drygrass", p))

EXTRA = (["fur_new"] if FUREXTRA else []) + ["drygrass"]
print(f"images   : {len(work)} per model, {len(work)*len(CKPTS)} forward passes total")
for fam in fams + EXTRA:
    print(f"   {fam:<18} {sum(1 for f,_ in work if f == fam):>4}")
n_frozen = sum(1 for f,_ in work if f in fams)
assert n_frozen == 280, f"frozen benchmark is {n_frozen}, expected 280 (t-prefixed fur in the folder?)"

# ---------- evaluate ----------
rows, t0 = [], time.time()
for mi, name in enumerate(CKPTS, 1):
    tm = time.time()
    print(f"\n[{mi}/{len(CKPTS)}] loading {LABEL[name]} ...")
    m = load(name)
    for fam, p in tqdm(work, desc=f"{LABEL[name]:<18}", ncols=90, leave=True):
        im = cv2.imread(p)
        if im is None: continue
        pm = prob_map(m, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))   # native aspect, as Table 3
        f  = os.path.basename(p)
        r  = dict(model=LABEL[name], family=fam, file=f,
                  supplied=(fam == "fur" and f in SUPPLIED), max_prob=float(pm.max()))
        for t in TAUS:
            r[f"fire{t}"] = int(pm.max() >= t)
            r[f"area{t}"] = float((pm >= t).mean())
        rows.append(r); 
    del m; torch.cuda.empty_cache()
    print(f"   {LABEL[name]} done in {time.time()-tm:.0f}s")
print(f"\nall models done in {time.time()-t0:.0f}s")

df = pd.DataFrame(rows)
df.to_csv("/kaggle/working/setC_four_models_drygrass.csv", index=False)

# ---------- ANCHOR: nothing new is comparable to the paper until this passes ----------
a = df[(df.model == "corrected ep8") & (df.family == "gravel")]
fire, area = a["fire0.85"].mean(), a["area0.85"].mean()
print(f"\nANCHOR gravel/ep8  n={len(a)}  fire {fire:.3f} vs 0.480  |  mean area {area:.3f} vs 0.256")
assert len(a) == 50 and abs(fire - 0.480) <= 0.02, \
    "benchmark does not reproduce — drygrass numbers are NOT comparable to Table 2"
print("ANCHOR OK")

# fur split so the fur corrected column is honest
df["group"] = np.where(df.family == "fur_new", "fur (new, post training)",
               np.where(df.family != "fur", df.family,
                np.where(df.supplied, "fur (SUPPLIED as negatives)", "fur (held out)")))
ORDER = ["smash.original", "wall", "crackedmud", "raindrop", "gravel",
         "fur (SUPPLIED as negatives)", "fur (held out)", "fur (new, post training)",
         "drygrass"]
ORDER = [g for g in ORDER if g in set(df.group)] + \
        [g for g in sorted(set(df.group)) if g not in ORDER]
COLS  = [LABEL[k] for k in CKPTS]

for t in TAUS:
    print(f"\n{'='*76}\n  tau = {t}   FIRING RATE\n{'='*76}")
    piv = df.pivot_table(index="group", columns="model", values=f"fire{t}", aggfunc="mean")
    n   = (df.groupby("group").file.size() / df.model.nunique()).astype(int)
    print(pd.concat([n.rename("n"), piv], axis=1).reindex(ORDER)[["n"]+COLS].round(3).to_string())
    print(f"\n  tau = {t}   MEAN PREDICTED AREA")
    piv2 = df.pivot_table(index="group", columns="model", values=f"area{t}", aggfunc="mean")
    print(piv2.reindex(ORDER)[COLS].round(3).to_string())

print(f"\n{'='*76}\n  Wilson 95% intervals at tau 0.85, fur corrected vs control\n{'='*76}")
for g in ORDER:
    line = f"  {g:<30}"
    for col in ["fur corrected ep4", "control ep4"]:
        sub = df[(df.model == col) & (df.group == g)]
        if not len(sub): continue
        k_, n_ = int(sub["fire0.85"].sum()), len(sub)
        lo, hi = wilson(k_, n_)
        line += f"{col.split()[0]:>4} {k_:>3}/{n_:<3} [{lo:.2f},{hi:.2f}]   "
    print(line)

# ---------- drygrass detail, all four models ----------
print(f"\n{'='*76}\n  DRYGRASS, an unreached family: full profile\n{'='*76}")
dg = df[df.family == "drygrass"]
for col in COLS:
    s = dg[dg.model == col]
    if not len(s): continue
    k_, n_ = int(s["fire0.85"].sum()), len(s)
    lo, hi = wilson(k_, n_)
    print(f"  {col:<20} fire@0.85 {k_:>3}/{n_:<3} = {k_/n_:.3f} [{lo:.3f},{hi:.3f}]   "
          f"mean max {s.max_prob.mean():.4f}   median {s.max_prob.median():.4f}   "
          f"area {s['area0.85'].mean():.4f}   silent<0.01 {(s.max_prob<0.01).mean():.3f}")

b = dg[dg.model == "baseline"].set_index("file").max_prob
e = dg[dg.model == "corrected ep8"].set_index("file").max_prob
if len(b) and len(e):
    common = b.index.intersection(e.index)
    print(f"\n  baseline fires on {int((b[common]>=0.85).sum())}/{len(common)}; "
          f"of those, corrected ep8 still fires on "
          f"{int(((b[common]>=0.85)&(e[common]>=0.85)).sum())}")
    print("  top 5 by corrected ep8 max prob:")
    for f_, v in e[common].sort_values(ascending=False).head(5).items():
        print(f"    {f_:<44} ep8 {v:.4f}   baseline {b[f_]:.4f}")

print("\nHOW TO READ THIS")
print("  Compare 'fur corrected ep4' with 'control ep4' row by row.")
print("  Both improve  -> continued training, not the fur negatives.")
print("  Only fur arm  -> attributable to supplying fur.")
print("  'fur (SUPPLIED as negatives)' is contaminated for the fur corrected column by")
print("  construction; report 'fur (held out)' and 'fur (new, post training)' instead.")
print("  DRYGRASS received no negatives at any stage: the baseline and corrected ep8")
print("  columns are the informative ones. A low fur corrected value there is global")
print("  silencing, not learning about grass — the control column separates the two.")