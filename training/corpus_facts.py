# Audits the training corpus and splits and prints every Appendix F figure
# (counts, resolutions, damage area, class-weight derivation). Requires the
# unreleased corpus.

# =========================================================
# SETUP FACTS — everything Section 9 of the paper needs
# run from the same working directory as the training scripts
# =========================================================
import os, glob, random, platform, subprocess, csv
from datetime import datetime
import numpy as np, cv2, torch

POS_IMG = ['extracted_smashed_data/smashed_batch1/images',
           'extracted_smashed_data/smashed_batch2/images']
POS_MSK = ['smashed_masks_batch1', 'smashed_masks_batch2']
NEG_DIRS = ['background_smashed0804', 'background', 'falsesmashed', 'scratchedforsmashed']
EXTRA_DIRS = ['fur', 'noise_probe']
CKPT_DIRS = ['checkpoints_smashed_vanilla', 'checkpoints_for_smashed_corrected_aug',
             'checkpoints_smashed_scratchfix_aug', 'checkpoints_smashed_furfix']
VAL_FRAC, EXTS = 0.12, ('.jpg','.jpeg','.jfif','.png','.bmp','.webp')
L = lambda t: print("\n" + "="*68 + f"\n{t}\n" + "="*68)

# ---------- 1. positives ----------
L("1. POSITIVES (images with masks)")
pairs = []
for idir, mdir in zip(POS_IMG, POS_MSK):
    if not (os.path.isdir(idir) and os.path.isdir(mdir)):
        print(f"  MISSING {idir} | {mdir}"); continue
    mm = {os.path.splitext(f)[0] for f in os.listdir(mdir) if f.lower().endswith('.png')}
    got = [(os.path.join(idir,f), os.path.join(mdir, os.path.splitext(f)[0]+'.png'))
           for f in sorted(os.listdir(idir)) if os.path.splitext(f)[0] in mm]
    n_img = len([f for f in os.listdir(idir) if f.lower().endswith(EXTS)])
    print(f"  {idir:<48} {n_img} images, {len(mm)} masks, {len(got)} paired")
    pairs += got
print(f"  TOTAL PAIRED POSITIVES: {len(pairs)}")

sizes, cover = [], []
for ip, mp in pairs[:400]:
    im = cv2.imread(ip)
    if im is None: continue
    sizes.append(im.shape[:2])
    m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if m is not None: cover.append((m > 0).mean())
if sizes:
    h = sorted(s[0] for s in sizes); w = sorted(s[1] for s in sizes)
    print(f"  resolution: {w[0]}x{h[0]} to {w[-1]}x{h[-1]}, median {w[len(w)//2]}x{h[len(h)//2]}")
    print(f"  damage pixels per positive image: mean {np.mean(cover):.4f}, median {np.median(cover):.4f}")
    print(f"  (implied positive-class frequency {np.mean(cover)*100:.2f}% "
          f"— compare the class weights [0.1281, 1.8719])")

# ---------- 2. negatives ----------
L("2. NEGATIVES (all-background)")
negs, per_dir = [], {}
for d in NEG_DIRS + EXTRA_DIRS:
    got = sorted(os.path.join(d,f) for f in os.listdir(d)
                 if f.lower().endswith(EXTS)) if os.path.isdir(d) else []
    per_dir[d] = len(got)
    print(f"  {d:<32} {len(got)}")
    if d in NEG_DIRS: negs += got
print(f"  TOTAL TRAINING NEGATIVES (4 folders): {len(negs)}")

# ---------- 3. the actual split used ----------
L("3. SPLIT AS TRAINED (seed 0, val_frac 0.12)")
random.seed(0); np.random.seed(0)
p, n = list(pairs), list(negs)
random.shuffle(p); random.shuffle(n)
sp = lambda l, f: (l[max(1,int(len(l)*f)):], l[:max(1,int(len(l)*f))])
pos_tr, pos_val = sp(p, VAL_FRAC); neg_tr, neg_val = sp(n, VAL_FRAC)
fur = sorted(glob.glob("fur/*")); fur = [x for x in fur if x.lower().endswith(EXTS)]
random.shuffle(fur); k = max(1, int(len(fur)*0.25)); fur_val, fur_tr = fur[:k], fur[k:]
print(f"  positives   train {len(pos_tr):>5}  val {len(pos_val):>4}")
print(f"  negatives   train {len(neg_tr):>5}  val {len(neg_val):>4}")
print(f"  fur         train {len(fur_tr):>5}  held out {len(fur_val):>4}  (of {len(fur)})")
print(f"  TOTAL train {len(pos_tr)+len(neg_tr):>5}  val {len(pos_val)+len(neg_val):>4}")
print(f"  NOTE: there is no separate test set — validation is the only in-house holdout.")

# ---------- 4. checkpoints ----------
L("4. CHECKPOINTS")
for d in CKPT_DIRS:
    for f in sorted(glob.glob(f"{d}/*.pth")):
        st = os.stat(f)
        print(f"  {f:<62} {st.st_size/1e6:7.1f} MB  "
              f"{datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}")

# ---------- 5. training logs: epochs, best rows, wall-clock ----------
L("5. TRAINING LOGS")
for lf in sorted(glob.glob("*log*.csv")):
    rows = list(csv.DictReader(open(lf)))
    if not rows: continue
    print(f"\n  --- {lf}  ({len(rows)} epochs) ---")
    ts = [r.get("datetime","") for r in rows if r.get("datetime")]
    if len(ts) > 1:
        try:
            t0 = datetime.strptime(ts[0], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(ts[-1], "%Y-%m-%d %H:%M:%S")
            print(f"      wall clock {t1-t0} total, ~{(t1-t0)/max(len(ts)-1,1)} per epoch")
        except Exception: pass
    key = "val_dice" if "val_dice" in rows[0] else None
    if key:
        best = max(rows, key=lambda r: float(r[key]))
        print(f"      best {key} {float(best[key]):.4f} at epoch {best['epoch']}"
              + (f", fpr_ab {float(best['fpr_ab']):.4f}" if "fpr_ab" in best else "")
              + (f", fpr_old_negs {float(best['fpr_old_negs']):.4f}" if "fpr_old_negs" in best else ""))
        print(f"      final epoch {rows[-1]['epoch']}: {key} {float(rows[-1][key]):.4f}")

# ---------- 6. environment ----------
L("6. ENVIRONMENT / HARDWARE")
print(f"  python  {platform.python_version()}   platform {platform.platform()}")
print(f"  torch   {torch.__version__}   CUDA {torch.version.cuda}   available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        pr = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {pr.name}, {pr.total_memory/1e9:.1f} GB, {pr.multi_processor_count} SMs")
for m in ["transformers", "albumentations", "opencv-python", "numpy"]:
    try:
        import importlib.metadata as md; print(f"  {m:<18} {md.version(m)}")
    except Exception: print(f"  {m:<18} ?")
try:
    print("  " + subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,driver_version",
                                          "--format=csv,noheader"]).decode().strip())
except Exception: pass

# ---------- 7. model size ----------
L("7. MODEL")
try:
    from transformers import SegformerForSemanticSegmentation
    m = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b3", num_labels=2,
                                                         ignore_mismatched_sizes=True)
    tot = sum(p.numel() for p in m.parameters())
    print(f"  SegFormer MiT-B3, 2 classes: {tot/1e6:.1f}M parameters, input 512x512")
except Exception as e:
    print("  (skipped:", e, ")")

L("STILL NEEDED FROM YOU (not derivable)")
print("""  a) Where the in-house images came from and roughly when collected
     (e.g. "vehicle inspection photographs from a Kenyan motor-insurance
      claims pipeline, collected 2024-2026")
  b) Whether the images were annotated by you, a team, or a vendor
  c) How many extra fur images you tested yesterday, and how they differ
     from the training fur (colour, pile length, crop distance)
  d) Your email address for the paper""")