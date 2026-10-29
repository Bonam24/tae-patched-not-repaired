!pip install -q transformers==5.13.0
# RESTART THE KERNEL after this install, then run the rest.

import os, glob
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
import transformers
from transformers import SegformerForSemanticSegmentation
print("transformers", transformers.__version__)
assert transformers.__version__.startswith("5.13"), "restart the kernel after the pip install"

MINE  = "/kaggle/input/datasets/bonamosen"
WANT  = [("baseline","vanilla"), ("corrected ep8","scratchfix_ep8"),
         ("fur corrected ep4","furfix_ep4"), ("control ep4","control_ep4")]
HW, MODES = (512,512), ["blur","mean","shuffle"]
MEAN = np.array([0.485,0.456,0.406],np.float32); STD = np.array([0.229,0.224,0.225],np.float32)
DEV  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- checkpoints ----------
found = {os.path.basename(p).replace(".pth",""): p
         for p in sorted(glob.glob(f"{MINE}/**/*.pth", recursive=True))}
print("checkpoints on disk:", list(found))
CK = {}
for lab,key in WANT:
    hit=[k for k in found if key in k]
    if hit: CK[lab]=found[hit[0]]
    else:   print("  NOT FOUND:", key)
assert CK, "no checkpoints matched WANT"

# ---------- noise stimuli: search, or regenerate ----------
cands = [d for d in glob.glob("/kaggle/input/**/noise_probe", recursive=True) if os.path.isdir(d)]
cands += [d for d in ["noise_probe", "/kaggle/working/noise_probe"] if os.path.isdir(d)]
paths = []
for d in cands:
    paths = sorted(p for p in glob.glob(f"{d}/**/*", recursive=True)
                   if p.lower().endswith(('.png','.jpg','.jpeg')))
    if paths: print(f"noise stimuli: {len(paths)} from {d}"); break

if not paths:
    print("no noise_probe folder found, regenerating the stimuli (same recipe as training)")
    out = "/kaggle/working/noise_probe"; os.makedirs(out, exist_ok=True)
    for lo, hi in [(0.01,0.05), (0.05,0.12), (0.12,0.22)]:
        for s in range(5):
            rng = np.random.default_rng(1000*int(hi*100)+s); S = 320
            Fr = np.fft.fftshift(np.fft.fft2(rng.standard_normal((S,S))))
            u = np.fft.fftshift(np.fft.fftfreq(S)); r = np.hypot(*np.meshgrid(u,u,indexing='ij'))
            b = np.real(np.fft.ifft2(np.fft.ifftshift(Fr*((r>=lo)&(r<=hi)))))
            b = (b-b.mean())/(b.std()+1e-8)
            patch = (np.clip(0.5+0.35*b,0,1)[...,None].repeat(3,2)*255).astype(np.uint8)
            img = np.full((512,512,3),128,np.uint8); img[96:96+S,96:96+S] = patch
            cv2.imwrite(f"{out}/noise_{lo:.2f}_{s}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    paths = sorted(glob.glob(f"{out}/*.png"))
    print(f"  regenerated {len(paths)}")
assert paths, "still no noise stimuli"

def load(p):
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3", num_labels=2, ignore_mismatched_sizes=True)
    sd = torch.load(p, map_location=DEV, weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    ov = len(set(m.state_dict()) & set(sd))
    assert ov > 600, f"only {ov}/644 keys overlap, wrong transformers version: {p}"
    m.load_state_dict(sd, strict=False)
    return m.to(DEV).eval()

@torch.no_grad()
def pmax(m, rgb):
    x = cv2.resize(rgb, HW[::-1]).astype(np.float32)/255.
    x = torch.from_numpy(((x-MEAN)/STD).transpose(2,0,1)).unsqueeze(0).float().to(DEV)
    lo = F.interpolate(m(x).logits, size=HW, mode="bilinear", align_corners=False)
    return float(torch.softmax(lo,1)[0,1].max())

def erase(img, mode, seed=0):
    if mode == "blur":
        return cv2.GaussianBlur(img, (0,0), max(1.0, 0.02*min(img.shape[:2])*4))
    if mode == "mean":
        return np.full_like(img, img.reshape(-1,3).mean(0).astype(np.uint8))
    if mode == "shuffle":
        flat = img.reshape(-1,3)
        return flat[np.random.default_rng(seed).permutation(len(flat))].reshape(img.shape)

rows = []
for lab in CK:
    m = load(CK[lab]); n = 0
    for p in paths:
        im = cv2.imread(p)
        if im is None: print("  unreadable:", p); continue
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        base = pmax(m, rgb)
        r = dict(model=lab, file=os.path.basename(p), before=base)
        for mode in MODES:
            r[mode] = (base - pmax(m, erase(rgb, mode)))/max(base, 1e-6)
        rows.append(r); n += 1
    del m; torch.cuda.empty_cache(); print(f"  {lab}: scored {n} images")

assert rows, "no rows produced"
d = pd.DataFrame(rows); d.to_csv("/kaggle/working/noise_erasure_signature.csv", index=False)
print("\n=== noise probe, global erasure, mean relative drop ===")
print(d.groupby("model")[["before"]+MODES].mean().round(4).to_string())

print("\n=== the shuffle column against the natural families (baseline) ===")
for fam, v in [("smash.original",-0.069),("wall",0.152),("raindrop",0.205),
               ("gravel",0.259),("fur",0.383),("crackedmud",0.446)]:
    print(f"    {fam:<16} {v:+.3f}")
b = d[d.model=="baseline"]
if len(b):
    print(f"    {'NOISE PROBE':<16} {b['shuffle'].mean():+.3f}   "
          f"(min {b['shuffle'].min():+.3f}, max {b['shuffle'].max():+.3f})")