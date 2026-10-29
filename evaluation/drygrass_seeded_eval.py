# Dry grass (52 images) over the eight epoch-4 checkpoints: reported fur +
# control and seeds 1-3 (SS7.1, Table 3 and Table 6 dry grass rows). Ran on
# Azure ML; produced traces/drygrass_seeded_per_image.csv and the summary JSON.

# =========================================================
# Dry grass x six seeded epoch-4 arms + the reported pair.
# Evaluation only, nothing retrained. Same harness that reproduced Table 2.
# =========================================================
# !pip install -q transformers==5.13.0 opencv-python-headless
#   (if you install, RESTART THE KERNEL, then run this cell)

CKPTS = {
  "reported_fur":     "checkpoints_smashed_furfix/smashed_furfix_ep4.pth",
  "reported_control": "checkpoints_smashed_control/smashed_control_ep4.pth",
  "seed1_fur":        "checkpoints_seeds/seed1_fur/ep4.pth",
  "seed1_control":    "checkpoints_seeds/seed1_control/ep4.pth",
  "seed2_fur":        "checkpoints_seeds/seed2_fur/ep4.pth",
  "seed2_control":    "checkpoints_seeds/seed2_control/ep4.pth",
  "seed3_fur":        "checkpoints_seeds/seed3_fur/ep4.pth",
  "seed3_control":    "checkpoints_seeds/seed3_control/ep4.pth",
}
IMG_DIR = "drygrass"
OUT_DIR = "drygrass_out"

import os, glob, math, json
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
import transformers
from transformers import SegformerForSemanticSegmentation

print("transformers", transformers.__version__, "| cwd", os.getcwd())
assert transformers.__version__.startswith("5.13"), "pin transformers==5.13.0 and restart the kernel"
os.makedirs(OUT_DIR, exist_ok=True)

def resolve(rel, is_dir=False):
    """rel as given, else search a few levels around cwd and home for the same tail."""
    if (os.path.isdir(rel) if is_dir else os.path.isfile(rel)):
        return os.path.abspath(rel)
    head = rel.split("/")[0]
    for root in [".", "..", "../..", os.path.expanduser("~")]:
        for hit in glob.glob(f"{root}/**/{head}", recursive=True):
            cand = os.path.join(os.path.dirname(hit), rel)
            if (os.path.isdir(cand) if is_dir else os.path.isfile(cand)):
                return os.path.abspath(cand)
    raise FileNotFoundError(rel)

CK = {}
print("\nRESOLVED CHECKPOINTS:")
for lab, rel in CKPTS.items():
    CK[lab] = resolve(rel)
    print(f"  {lab:<18} {CK[lab]}")
assert len(CK) == 8

# ---- PRE-REGISTERED REPORTING RULE, fixed before any result is seen ----
RULE = """Per seed, compare the treatment arm against its own matched control at epoch 4 in
mean max probability over the 52 dry grass images (the Table 6 quantity).
  4 of 4 negative -> "the direction reproduces in four of four runs"
  3 of 4 negative -> "in three of four, and at seed n it does not" (as 7.1 handles mud and rain)
  fewer           -> report the count plainly; no claim of broad reach
The row goes into Table 6 either way. Fire rates at tau 0.85 carry Wilson intervals and are
secondary; mean max probability is the pre-registered quantity."""
print("\n=== PRE-REGISTERED RULE ===\n" + RULE)

ANCHOR = {"reported_fur": (4, 52), "reported_control": (31, 52)}   # fire@0.85, from the Kaggle run
BACKBONE, HW = "nvidia/mit-b3", (512, 512)
EXTS = ('.jpg','.jpeg','.jfif','.png','.bmp','.webp')
TAUS = [0.50, 0.85]
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
DEV  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\ndevice:", DEV)

def load(path):
    m = SegformerForSemanticSegmentation.from_pretrained(
        BACKBONE, num_labels=2, ignore_mismatched_sizes=True)
    sd = torch.load(path, map_location=DEV, weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    r  = m.load_state_dict(sd, strict=False)
    cls = [x for x in r.missing_keys if "classifier" in x or "decode_head" in x]
    bb  = [x for x in r.missing_keys if x not in cls]
    assert not cls, f"HEAD DID NOT LOAD: {path}: {cls[:4]}"
    assert len(bb) <= 5, f"{len(bb)} backbone keys missing: {path}: {bb[:4]}"
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

IMG = resolve(IMG_DIR, is_dir=True)
files = sorted(p for p in glob.glob(f"{IMG}/**/*", recursive=True)
               if p.lower().endswith(EXTS) and os.path.isfile(p))
files = [p for p in files if not os.path.basename(p).startswith("t")]
print(f"\nimages: {len(files)} from {IMG} (expect 52)")
assert len(files) == 52, "image count differs from the set the Kaggle numbers were measured on"

rows = []
for lab, path in CK.items():
    m = load(path)
    for p in files:
        im = cv2.imread(p)
        if im is None: print("  unreadable:", p); continue
        pm = prob_map(m, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        r = dict(arm=lab, family="drygrass", file=os.path.basename(p), max_prob=float(pm.max()))
        for t in TAUS:
            r[f"fire{t}"] = int(pm.max() >= t); r[f"area{t}"] = float((pm >= t).mean())
        rows.append(r)
    del m; torch.cuda.empty_cache(); print(f"  {lab:<18} done")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT_DIR}/drygrass_seeded_per_image.csv", index=False)

print("\n=== ANCHOR ===")
ok = True
for lab, (k_exp, n_exp) in ANCHOR.items():
    s = df[df.arm == lab]; k_, n_ = int(s["fire0.85"].sum()), len(s)
    good = (k_ == k_exp and n_ == n_exp); ok &= good
    print(f"  {lab:<18} fire@0.85 {k_}/{n_}  expected {k_exp}/{n_exp}  {'OK' if good else 'FAIL'}")
assert ok, "harness does not reproduce the reported pair — seeded numbers are NOT comparable"
print("ANCHOR OK")

print("\n" + "="*96)
print(f"  DRYGRASS, n={len(files)}, all arms at epoch 4")
print("="*96)
prof = {}
for lab in CK:
    s = df[df.arm == lab]; k_, n_ = int(s["fire0.85"].sum()), len(s); lo, hi = wilson(k_, n_)
    prof[lab] = dict(n=n_, k=k_, fire=k_/n_, lo=lo, hi=hi,
                     mean_max=float(s.max_prob.mean()), median_max=float(s.max_prob.median()),
                     area=float(s["area0.85"].mean()), silent=float((s.max_prob < 0.01).mean()))
    p = prof[lab]
    print(f"  {lab:<18} fire@0.85 {k_:>3}/{n_:<3} = {p['fire']:.3f} [{lo:.3f},{hi:.3f}]"
          f"   mean max {p['mean_max']:.4f}   median {p['median_max']:.4f}"
          f"   area {p['area']:.4f}   silent {p['silent']:.3f}")

print("\n" + "="*96)
print("  TREATMENT MINUS MATCHED CONTROL, mean max probability (the Table 6 row)")
print("="*96)
pairs = [("reported run","reported_fur","reported_control"),
         ("seed 1","seed1_fur","seed1_control"),
         ("seed 2","seed2_fur","seed2_control"),
         ("seed 3","seed3_fur","seed3_control")]
deltas, neg = {}, 0
for name, t, c in pairs:
    d_ = prof[t]["mean_max"] - prof[c]["mean_max"]; deltas[name] = d_; neg += (d_ < 0)
    print(f"  {name:<14} {prof[t]['mean_max']:.4f} - {prof[c]['mean_max']:.4f} = {d_:+.4f}"
          f"   (fire {prof[t]['fire']:.3f} vs {prof[c]['fire']:.3f})")
print(f"\n  quieter than control in {neg} of 4 runs")
if neg == 4:
    verdict = "the direction reproduces in four of four runs"
elif neg == 3:
    bad = [n for n, d_ in deltas.items() if d_ >= 0][0]
    verdict = f"in three of four, and at {bad} it does not"
else:
    verdict = f"quieter than control in only {neg} of 4 runs; no broad reach claim"
print(f"  PRE-REGISTERED VERDICT: {verdict}")
print("\n  Table 6 row:  Dry grass   " + "   ".join(f"{deltas[n]:+.3f}" for n, _, _ in pairs))

json.dump(dict(profile=prof, deltas=deltas, n_negative=neg, verdict=verdict, rule=RULE,
               checkpoints=CK, images=IMG, transformers=transformers.__version__),
          open(f"{OUT_DIR}/drygrass_seeded_summary.json","w"), indent=2)
print(f"\nwrote {OUT_DIR}/drygrass_seeded_per_image.csv and _summary.json")