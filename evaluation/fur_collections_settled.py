# =========================================================
# SETTLE THE FUR COLLECTIONS
# Frozen 40 (contextShortcutData/fur, non t-prefixed) vs the 49 later
# images (furextra), under BOTH firing criteria, on all four models.
# =========================================================
# pip install transformers==5.13.0  

import os, re, glob, math, random
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
import transformers
from transformers import SegformerForSemanticSegmentation

assert transformers.__version__.startswith("5.13"), "restart the kernel after the pip install"
MINE = "/kaggle/input/datasets/bonamosen"
EXTS = ('.jpg','.jpeg','.jfif','.png','.bmp','.webp')
TNEW = re.compile(r'^t\d')
TAU, AREA_THR, AREA_MIN = 0.85, 0.5, 0.001
HW   = (512, 512)
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
DEV  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WANT = [("baseline","vanilla"), ("corrected ep8","scratchfix_ep8"),
        ("fur corrected ep4","furfix_ep4"), ("control ep4","control_ep4")]
found = {os.path.basename(p): p for p in sorted(glob.glob(f"{MINE}/**/*.pth", recursive=True))}
CK = {}
for lab, key in WANT:
    hit = [f for f in found if key in f]
    assert len(hit) == 1, f"{lab} ({key}) -> {hit}"
    CK[lab] = found[hit[0]]
    print(f"{lab:<20} {CK[lab]}")

SETC = glob.glob("/kaggle/input/**/contextShortcutData", recursive=True)[0]
_fx  = [d for d in glob.glob("/kaggle/input/**/furextra", recursive=True) if os.path.isdir(d)]
FUREXTRA = sorted(_fx, key=len)[-1]
print("\nSETC     =", SETC, "\nFUREXTRA =", FUREXTRA)

# ---------- the three fur sets, with provenance asserted ----------
furdir = os.path.join(SETC, "fur")
allf   = sorted(f for f in os.listdir(furdir) if f.lower().endswith(EXTS))
frozen = [f for f in allf if not TNEW.match(f)]
tpref  = [f for f in allf if TNEW.match(f)]
extra  = sorted(f for f in os.listdir(FUREXTRA) if f.lower().endswith(EXTS))
print(f"\nfur/ holds {len(allf)}: {len(frozen)} frozen + {len(tpref)} t-prefixed")
print(f"furextra/ holds {len(extra)}")
assert len(frozen) == 40, f"frozen family is {len(frozen)}, expected 40"
assert len(extra)  == 49, f"furextra is {len(extra)}, expected 49"
assert not (set(frozen) & set(extra)), "frozen 40 and furextra overlap by filename"

# reconstruct the supplied / held-out split exactly as training did
random.seed(0); np.random.seed(0); torch.manual_seed(0)
_p = [os.path.join(furdir, f) for f in frozen]; random.shuffle(_p)
k = max(1, int(len(_p)*0.25))
HELD, SUPPLIED = _p[:k], _p[k:]
print(f"split: {len(SUPPLIED)} supplied as negatives, {len(HELD)} held out")

SETS = {
    "fur supplied (30)":   SUPPLIED,
    "fur held out (10)":   HELD,
    "fur frozen 40":       [os.path.join(furdir, f) for f in frozen],
    "fur later 49":        [os.path.join(FUREXTRA, f) for f in extra],
    "gravel 50 (ANCHOR)":  sorted(p for p in glob.glob(f"{SETC}/gravel/*") if p.lower().endswith(EXTS)),
}
for k_, v in SETS.items(): print(f"  {k_:<22} n={len(v)}")

# ---------- model + scoring ----------
def load(path):
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
def score(model, path):
    """Returns max prob (frame invariant) and area at 0.5 in BOTH frames."""
    im = cv2.imread(path)
    if im is None: return None
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
    x = cv2.resize(rgb, HW[::-1]).astype(np.float32)/255.0
    x = torch.from_numpy(((x-MEAN)/STD).transpose(2,0,1)).unsqueeze(0).float().to(DEV)
    lo = F.interpolate(model(x).logits, size=HW, mode="bilinear", align_corners=False)
    pm512 = torch.softmax(lo,1)[0,1].float().cpu().numpy()      # probe harness frame
    pmnat = cv2.resize(pm512, (w,h))                            # benchmark harness frame
    return dict(max_prob=float(pm512.max()),
                area_native=float((pmnat >= AREA_THR).mean()),
                area_512=float((pm512 >= AREA_THR).mean()),
                area_tau_native=float((pmnat >= TAU).mean()))

def wilson(k, n, z=1.96):
    if n == 0: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)

rows = []
for lab, path in CK.items():
    m = load(path)
    for sname, paths in SETS.items():
        recs = [score(m, p) for p in paths]
        recs = [r for r in recs if r]
        mx  = np.array([r["max_prob"] for r in recs])
        an  = np.array([r["area_native"] for r in recs])
        a5  = np.array([r["area_512"] for r in recs])
        atn = np.array([r["area_tau_native"] for r in recs])
        n = len(recs)
        rows.append(dict(model=lab, set=sname, n=n,
                         tau_fired=int((mx>=TAU).sum()),
                         train_fired_native=int((an>AREA_MIN).sum()),
                         train_fired_512=int((a5>AREA_MIN).sum()),
                         mean_max=mx.mean(), mean_area_tau=atn.mean()))
    del m; torch.cuda.empty_cache(); print(f"  scored {lab}")

df = pd.DataFrame(rows)
df["tau_rate"]   = df.tau_fired/df.n
df["train_nat"]  = df.train_fired_native/df.n
df["train_512"]  = df.train_fired_512/df.n
df.to_csv("/kaggle/working/fur_collections_settled.csv", index=False)

# ---------- ANCHOR ----------
a = df[(df.model=="corrected ep8") & (df.set=="gravel 50 (ANCHOR)")].iloc[0]
print(f"\nANCHOR gravel/ep8: tau rate {a.tau_rate:.3f} vs 0.480 | mean area@tau {a.mean_area_tau:.3f} vs 0.256")
assert abs(a.tau_rate-0.480) <= 0.02 and abs(a.mean_area_tau-0.256) <= 0.02, \
    "harness does not reproduce Table 3 — the fur numbers below are not comparable"
print("ANCHOR OK\n")

print("="*104)
print("  FIRING BY SET AND CRITERION   (tau = max prob >= 0.85; train = area > 0.001 at 0.5)")
print("="*104)
for sname in SETS:
    print(f"\n{sname}")
    for lab in CK:
        r = df[(df.model==lab)&(df.set==sname)].iloc[0]
        lo,hi = wilson(int(r.tau_fired), int(r.n))
        print(f"   {lab:<20} tau {int(r.tau_fired):>3}/{int(r.n):<3} = {r.tau_rate:.3f} [{lo:.3f},{hi:.3f}]"
              f"   train(native) {int(r.train_fired_native):>3}/{int(r.n):<3} = {r.train_nat:.3f}"
              f"   train(512) {int(r.train_fired_512):>3}/{int(r.n):<3} = {r.train_512:.3f}"
              f"   mean max {r.mean_max:.4f}")

print("\n" + "="*104)
print("  THE QUESTION: does the later collection fire at half the rate, and on which model?")
print("="*104)
for lab in CK:
    f40 = df[(df.model==lab)&(df.set=="fur frozen 40")].iloc[0]
    f49 = df[(df.model==lab)&(df.set=="fur later 49")].iloc[0]
    ratio = f49.tau_rate/f40.tau_rate if f40.tau_rate else float('nan')
    print(f"   {lab:<20} frozen40 {f40.tau_rate:.3f}   later49 {f49.tau_rate:.3f}   ratio {ratio:.2f}")
print("\n   A ratio near 1.00 means the collections agree on that model.")
print("   A ratio near 0.50 is the halving the paper's qualification describes.")
print("\nwrote /kaggle/working/fur_collections_settled.csv")
