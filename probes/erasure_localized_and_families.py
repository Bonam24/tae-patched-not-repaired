# Localized erasure (SS4, Table 4): blur / mean / inpaint / permutation on
# the firing region vs a displaced control, plus the cross-family erasure
# signature of Appendix C. Run after noise_sweep_setup.py (same session).

# =========================================================
# ERASURE SIGNATURE ACROSS FAMILIES — the shared-mechanism test
# assumes get_model, prob_map, CKPTS, VAN, COR from the earlier cells
# =========================================================
import os, glob, random
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt

SETC   = glob.glob("/kaggle/input/**/contextShortcutData", recursive=True)[0]
EXTS   = ('.jpg','.jpeg','.jfif','.png','.bmp','.webp')
TAU    = 0.50
REGION = 0.25          # region side, fraction of min(H,W)
SIGMA  = 0.05          # blur strength, fraction of region side
N_PER  = 8             # firing images sampled per family
MODES  = ["blur", "mean", "inpaint", "shuffle"]

def firing_box(pm, side):
    H, W = pm.shape
    y, x = np.unravel_index(pm.argmax(), pm.shape)
    return (int(np.clip(x-side//2, 0, max(0,W-side))),
            int(np.clip(y-side//2, 0, max(0,H-side))), side, side)

def displaced_box(shape, box, side, rnd):
    H, W = shape[:2]; bx, by, bw, bh = box; best, bd = None, -1
    for _ in range(600):
        x = rnd.randint(0, max(0,W-side)); y = rnd.randint(0, max(0,H-side))
        if not (x < bx+bw and x+side > bx and y < by+bh and y+side > by):
            d = (x-bx)**2 + (y-by)**2
            if d > bd: best, bd = (x,y,side,side), d
    return best

def erase(img, box, mode, seed=0):
    x, y, w, h = box; out = img.copy(); reg = out[y:y+h, x:x+w]
    if mode == "blur":
        out[y:y+h, x:x+w] = cv2.GaussianBlur(reg, (0,0), max(1.0, SIGMA*min(w,h)*4))
    elif mode == "mean":
        out[y:y+h, x:x+w] = reg.reshape(-1,3).mean(0).astype(np.uint8)
    elif mode == "inpaint":
        m = np.zeros(img.shape[:2], np.uint8); m[y:y+h, x:x+w] = 255
        out = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), m, 5,
                                       cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
    elif mode == "shuffle":
        flat = reg.reshape(-1,3)
        out[y:y+h, x:x+w] = flat[np.random.default_rng(seed).permutation(len(flat))].reshape(reg.shape)
    return out

fams = sorted(d for d in os.listdir(SETC) if os.path.isdir(os.path.join(SETC, d)))
rows = []

for mname in CKPTS:
    m = get_model(mname)
    for fam in fams:
        paths = sorted(p for p in glob.glob(f"{SETC}/{fam}/**/*", recursive=True) if p.lower().endswith(EXTS))
        picked = 0
        for path in paths:
            if picked >= N_PER: break
            im = cv2.imread(path)
            if im is None: continue
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB); H, Wd = rgb.shape[:2]
            pm = prob_map(m, rgb)
            if pm.max() < TAU: continue                     # nothing to erase
            side = int(max(16, REGION*min(H, Wd)))
            box  = firing_box(pm, side)
            rnd  = random.Random(abs(hash(os.path.basename(path))) % (2**31))
            ctrl = displaced_box(rgb.shape, box, side, rnd)
            bx, by, bw, bh = box
            base = float(pm[by:by+bh, bx:bx+bw].max())
            for mode in MODES:
                for site, bb in (("target", box), ("displaced", ctrl)):
                    if bb is None: continue
                    after = float(prob_map(m, erase(rgb, bb, mode))[by:by+bh, bx:bx+bw].max())
                    rows.append(dict(model=mname, family=fam, file=os.path.basename(path),
                                     mode=mode, site=site, before=base, after=after,
                                     rel_drop=(base-after)/max(base,1e-6)))
            picked += 1
        print(f"{mname:<26} {fam:<18} {picked} firing images probed")

df = pd.DataFrame(rows); df.to_csv("/kaggle/working/erasure_signature.csv", index=False)

# ---------- the signature, per family ----------
for mname in CKPTS:
    g = df[df.model == mname]
    if g.empty: continue
    print(f"\n===== {mname} — mean relative drop =====")
    piv = g.pivot_table(index="family", columns=["site","mode"], values="rel_drop").round(3)
    print(piv.to_string())

# ---------- distance to the foliage/vehicle reference ----------
REF = "smash.original"     # the original foliage-in-vehicle-scene failure
for mname in CKPTS:
    g = df[(df.model == mname) & (df.site == "target")]
    sig = g.pivot_table(index="family", columns="mode", values="rel_drop").reindex(columns=MODES)
    if REF not in sig.index or sig.loc[REF].isna().any():
        print(f"\n{mname}: no usable {REF} reference"); continue
    ref = sig.loc[REF].values
    print(f"\n=== {mname}: signature distance from '{REF}' (target site) ===")
    for fam, r in sig.iterrows():
        if r.isna().any(): continue
        print(f"  {fam:<18} L1 {np.abs(r.values-ref).mean():.3f}   "
              f"[{', '.join(f'{m}:{v:.2f}' for m, v in zip(MODES, r.values))}]")
    print("  Small L1 = same fingerprint = one mechanism. Large L1 = distinct failure modes.")

# ---------- plot ----------
for mname in CKPTS:
    g = df[(df.model == mname) & (df.site == "target")]
    if g.empty: continue
    sig = g.pivot_table(index="family", columns="mode", values="rel_drop").reindex(columns=MODES)
    ax = sig.plot(kind="bar", figsize=(8,3.6), ylim=(-0.05,1.05), rot=20)
    ax.set_ylabel("relative drop"); ax.set_title(f"{mname} — erasure signature by family", fontsize=10)
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"/kaggle/working/signature_{mname}.png", dpi=150); plt.show()