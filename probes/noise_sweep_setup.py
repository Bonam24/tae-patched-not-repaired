# Frequency sweep on bandpass noise, baseline + corrected (paper SS5).
# Also DEFINES get_model / prob_map / bandpass_patch / CKPTS used by the
# three scripts below — run this first in the same session.

!pip install -q transformers==5.13.0

import os, glob
import numpy as np, pandas as pd, torch, cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation

MINE = "/kaggle/input/datasets/bonamosen"
ALL_MODELS = False        # False = vanilla + scratchfix_ep8 only (the two the paper needs)

# ---------- models ----------
found = {os.path.basename(p).replace(".pth",""): p for p in sorted(glob.glob(f"{MINE}/**/*.pth", recursive=True))}
order = ([k for k in found if "vanilla" in k.lower()] +
         [k for k in found if "vanilla" not in k.lower()])          # baseline first
CKPTS = {k: found[k] for k in order} if ALL_MODELS else \
        {k: found[k] for k in order if "vanilla" in k.lower() or "scratchfix" in k.lower()}
assert any("vanilla" in k.lower() for k in CKPTS), "vanilla checkpoint not found"
print("Running:", list(CKPTS))

BACKBONE, SIZE = "nvidia/mit-b3", (512, 512)
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu"); _M = {}

def get_model(name):
    if name in _M: return _M[name]
    m = SegformerForSemanticSegmentation.from_pretrained(BACKBONE, num_labels=2, ignore_mismatched_sizes=True)
    sd = torch.load(CKPTS[name], map_location=DEV, weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    r  = m.load_state_dict(sd, strict=False)
    cls = [k for k in r.missing_keys if "classifier" in k or "decode_head" in k]
    bb  = [k for k in r.missing_keys if k not in cls]
    assert not cls, f"HEAD DID NOT LOAD for {name}: {cls[:5]}"
    assert len(bb) <= 5, f"{len(bb)} backbone keys missing for {name}: {bb[:5]}"
    print(f"loaded {name} | missing {len(bb)} | unexpected {len(r.unexpected_keys)}")
    m.to(DEV).eval(); _M[name] = m; return m

@torch.no_grad()
def prob_map(model, rgb):
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, SIZE[::-1]).astype(np.float32)/255.0
    x = torch.from_numpy(((x-MEAN)/STD).transpose(2,0,1)).unsqueeze(0).float().to(DEV)
    lo = F.interpolate(model(x).logits, size=SIZE, mode="bilinear", align_corners=False)
    return cv2.resize(torch.softmax(lo,1)[0,1].float().cpu().numpy(), (w,h))

# ---------- stimulus ----------
def bandpass_patch(size, f_lo, f_hi, contrast, seed=0):
    rng = np.random.default_rng(seed)
    Fr = np.fft.fftshift(np.fft.fft2(rng.standard_normal((size, size))))
    u  = np.fft.fftshift(np.fft.fftfreq(size))
    r  = np.hypot(*np.meshgrid(u, u, indexing='ij'))
    b  = np.real(np.fft.ifft2(np.fft.ifftshift(Fr * ((r >= f_lo) & (r <= f_hi)))))
    b  = (b - b.mean()) / (b.std() + 1e-8)
    return (np.clip(0.5 + contrast*b, 0, 1)[..., None].repeat(3, 2) * 255).astype(np.uint8)

BANDS     = [(0.01,0.05), (0.05,0.12), (0.12,0.22), (0.22,0.35), (0.35,0.50)]
CONTRASTS = [0.05, 0.10, 0.20, 0.35, 0.50]     # 0.05 = the "sand" condition: fine but flat
SEEDS, TAU = range(8), 0.50
H = Wd = 512; S = 160; Y = X = (H - S)//2

fig, ax = plt.subplots(len(CONTRASTS), len(BANDS), figsize=(1.5*len(BANDS), 1.5*len(CONTRASTS)))
for i, cst in enumerate(CONTRASTS):
    for j, (lo, hi) in enumerate(BANDS):
        ax[i,j].imshow(bandpass_patch(S, lo, hi, cst, 0)); ax[i,j].axis("off")
        if i == 0: ax[i,j].set_title(f"{lo:.2f}–{hi:.2f}", fontsize=7)
        if j == 0: ax[i,j].text(-0.3, 0.5, f"c={cst}", fontsize=7, transform=ax[i,j].transAxes,
                                rotation=90, va="center")
plt.suptitle("stimuli — frequency (cols) × contrast (rows)", fontsize=10); plt.tight_layout(); plt.show()

# ---------- sweep ----------
rows = []
for name in CKPTS:
    m = get_model(name)
    for lo, hi in BANDS:
        for cst in CONTRASTS:
            for sd in SEEDS:
                img = np.full((H, Wd, 3), 128, np.uint8)
                img[Y:Y+S, X:X+S] = bandpass_patch(S, lo, hi, cst, sd)
                p = prob_map(m, img)[Y:Y+S, X:X+S]
                rows.append(dict(model=name, band=f"{lo:.2f}–{hi:.2f}", f_mid=(lo+hi)/2,
                                 contrast=cst, seed=sd, max_prob=float(p.max()), mean_prob=float(p.mean())))
df = pd.DataFrame(rows); df.to_csv("/kaggle/working/bandpass_grid.csv", index=False)

# ---------- results ----------
for name in CKPTS:
    g = df[df.model == name]
    print(f"\n=== {name} — mean max prob ===")
    print(g.pivot_table(index="band", columns="contrast", values="max_prob").round(3))
    print(f"--- fire rate (tau={TAU}) ---")
    print(g.assign(f=g.max_prob >= TAU).pivot_table(index="band", columns="contrast", values="f").round(2))

van = [k for k in CKPTS if "vanilla" in k.lower()][0]
for name in [k for k in CKPTS if k != van]:
    a = df[df.model==van].groupby(["band","contrast"]).max_prob.mean()
    b = df[df.model==name].groupby(["band","contrast"]).max_prob.mean()
    print(f"\n=== {van} minus {name} (positive = correction reduced firing) ===")
    print((a-b).unstack().round(3))

fig, ax = plt.subplots(1, len(CKPTS), figsize=(4.4*len(CKPTS), 3.4), squeeze=False)
for k, name in enumerate(CKPTS):
    g = df[df.model == name]
    for cst in CONTRASTS:
        s = g[g.contrast == cst].groupby("f_mid").max_prob.mean()
        ax[0,k].plot(s.index, s.values, marker="o", label=f"c={cst}")
    ax[0,k].set_title(name, fontsize=9); ax[0,k].set_xlabel("centre frequency (cyc/px)")
    ax[0,k].set_ylabel("mean max prob"); ax[0,k].set_ylim(-0.02, 1.02)
    ax[0,k].axhline(TAU, ls="--", lw=0.8, c="k"); ax[0,k].legend(fontsize=7)
plt.tight_layout(); plt.savefig("/kaggle/working/bandpass_curves.png", dpi=150); plt.show()