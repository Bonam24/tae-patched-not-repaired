# The patch-size control that refutes the spectral account (Appendix E):
# same band, patch side 80/160/320. Run after noise_sweep_setup.py.

# assumes get_model, prob_map, bandpass_patch, CKPTS from the previous cell
import os, glob
import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt

assert 'prob_map' in dir(), "run the bandpass cell first"
SETC = "/kaggle/input/datasets/bonamosen/contextshortcur/contextShortcutData"
EXTS = ('.jpg','.jpeg','.png','.bmp','.webp'); TAU = 0.50
VAN  = [k for k in CKPTS if "vanilla" in k.lower()][0]
COR  = [k for k in CKPTS if k != VAN][0]

# ================= A. patch-size control: cycles/pixel or cycles/patch? =================
BANDS = [(0.05,0.12), (0.12,0.22), (0.22,0.35), (0.35,0.50)]
rows = []
for S in [80, 160, 320]:
    Y = X = (512 - S)//2
    for lo, hi in BANDS:
        for sd in range(5):
            img = np.full((512,512,3), 128, np.uint8)
            img[Y:Y+S, X:X+S] = bandpass_patch(S, lo, hi, 0.35, sd)
            for name in CKPTS:
                p = prob_map(get_model(name), img)[Y:Y+S, X:X+S]
                rows.append(dict(S=S, band=f"{lo:.2f}–{hi:.2f}", model=name, max_prob=float(p.max())))
ps = pd.DataFrame(rows)
for name in CKPTS:
    print(f"\n=== {name} — patch-size control (c=0.35), mean max prob ===")
    print(ps[ps.model==name].pivot_table(index="band", columns="S", values="max_prob").round(3))
print("\nIf a row is CONSTANT across S -> cutoff is in cycles/pixel (a real filter property).")
print("If it SHIFTS with S -> you are measuring patch extent, and the frequency axis is mislabelled.\n")

# ================= B. set C: does spectral centroid predict firing? =================
def spec_stats(rgb512):
    """Computed on the 512x512 the MODEL sees, so frequencies are in the same units as the grid."""
    g = cv2.cvtColor(rgb512, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = (g - g.mean()) * np.outer(np.hanning(512), np.hanning(512))
    P = np.abs(np.fft.fftshift(np.fft.fft2(g)))**2
    u = np.fft.fftshift(np.fft.fftfreq(512))
    r = np.hypot(*np.meshgrid(u, u, indexing='ij'))
    m = r > 0
    centroid = float((P[m]*r[m]).sum() / (P[m].sum() + 1e-12))
    # fraction of power above the ep8 cutoff, and RMS contrast
    hi_frac  = float(P[m & (r > 0.22)].sum() / (P[m].sum() + 1e-12))
    return centroid, hi_frac, float(cv2.cvtColor(rgb512, cv2.COLOR_RGB2GRAY).std())

fams = sorted(d for d in os.listdir(SETC) if os.path.isdir(os.path.join(SETC, d)))
print("families:", fams)

recs = []
for fam in fams:
    fs = [p for p in sorted(glob.glob(f"{SETC}/{fam}/**/*", recursive=True)) if p.lower().endswith(EXTS)]
    for p in fs:
        im = cv2.imread(p)
        if im is None: continue
        rgb = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (512,512))
        cen, hif, rms = spec_stats(rgb)
        rec = dict(family=fam, file=os.path.basename(p), centroid=cen, hi_frac=hif, rms=rms)
        for name in CKPTS:
            rec[name] = float(prob_map(get_model(name), rgb).max())
        recs.append(rec)

sc = pd.DataFrame(recs); sc.to_csv("/kaggle/working/setC_spectral.csv", index=False)
print(f"\n{len(sc)} images\n")
print(sc.groupby("family").agg(n=("file","size"),
        centroid=("centroid","mean"), rms=("rms","mean"),
        **{f"{VAN}_fire": (VAN, lambda s: (s>=TAU).mean()),
           f"{COR}_fire": (COR, lambda s: (s>=TAU).mean())}).round(3))

# ================= C. threshold analysis =================
for name in CKPTS:
    fire = sc[name] >= TAU
    if fire.nunique() < 2:
        print(f"\n{name}: all images on the same side of tau — no threshold to find"); continue
    best = max(((t, ((sc.centroid < t) == fire).mean()) for t in np.arange(0.02, 0.45, 0.005)),
               key=lambda z: z[1])
    print(f"\n{name}: best centroid threshold {best[0]:.3f} cyc/px "
          f"-> {best[1]:.1%} of images correctly predicted (fires below, cold above)")
    print(f"  centroid | fired: {sc.loc[fire,'centroid'].mean():.3f}   cold: {sc.loc[~fire,'centroid'].mean():.3f}")
    print(f"  rms      | fired: {sc.loc[fire,'rms'].mean():.1f}     cold: {sc.loc[~fire,'rms'].mean():.1f}")

fig, ax = plt.subplots(1, len(CKPTS), figsize=(5.4*len(CKPTS), 4.2), squeeze=False)
for k, name in enumerate(CKPTS):
    for fam in fams:
        s = sc[sc.family == fam]
        ax[0,k].scatter(s.centroid, s[name], s=26, alpha=.75, label=fam)
    ax[0,k].axhline(TAU, ls="--", lw=.8, c="k"); ax[0,k].axvline(0.22, ls=":", lw=.8, c="r")
    ax[0,k].set_xlabel("spectral centroid (cyc/px)"); ax[0,k].set_ylabel("max prob")
    ax[0,k].set_title(name, fontsize=9); ax[0,k].legend(fontsize=7)
plt.tight_layout(); plt.savefig("/kaggle/working/setC_centroid.png", dpi=150); plt.show()