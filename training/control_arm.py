# The matched control (SS7): identical to fur_finetune.py except the 30 fur
# negatives are never added; sampler draw count pinned so both arms consume
# RNG identically. Requires the unreleased corpus; included for transparency.

# =========================================================
# CONTROL ARM for the fur intervention.
# Identical to the fur fine tune in every respect EXCEPT that the 30 fur
# training images are never added to the negative set. Same warm start,
# same seed, same schedule, same learning rate, same 30/10 fur split, so
# the held out ten are the same ten images.
# Purpose: separate "supplying fur removes fur" from "6 more epochs of
# training at LR 2e-5". If held out fur stays near 1.00 here while
# validation Dice still rises, the fur drop in the main run is the fur
# negatives and the Dice gain is continued training.
# =========================================================
import os, csv, random, glob
from datetime import datetime
import cv2, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

CFG = {
    "resume_ckpt": "checkpoints_smashed_scratchfix_aug/smashed_scratchfix_ep8.pth",
    "total_epochs": 6,
    "image_size": (512, 512),
    "batch_size": 4,
    "lr": 2e-5,
    "class_weights": [0.1281, 1.8719],
    "dice_factor": 2.0,
    "p_copy_paste": 0.25,
    "cp_instances": (1, 2),
    "cp_scale": (0.6, 1.3),
    "seam_alpha": 0.75,
    "seam_beta": 4.0,
    "fpr_lambda": 1.0,
    "fire_area": 0.001,
    "val_frac": 0.12,
    "out_dir": "checkpoints_smashed_control",     # CHANGED
    "log_csv": "smashed_control_log.csv",         # CHANGED
    "pos_image_dirs": ['extracted_smashed_data/smashed_batch1/images',
                       'extracted_smashed_data/smashed_batch2/images'],
    "pos_mask_dirs":  ['smashed_masks_batch1', 'smashed_masks_batch2'],
    "neg_image_dirs": ['background_smashed0804', 'background', 'falsesmashed', 'scratchedforsmashed'],
    "fur_dir": "fur",                 # still loaded and split, but NOT trained on
    "fur_val_frac": 0.25,
    "noise_probe_dir": "noise_probe"
}
ADD_FUR = False        # <<<<<< the only substantive difference from the fur run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225)
random.seed(0); np.random.seed(0); torch.manual_seed(0); EPS = 1e-7

# ---------- held out probe: bandpass noise, never seen in training ----------
def make_noise_probe(out_dir, n_per_band=5):
    os.makedirs(out_dir, exist_ok=True)
    if len(glob.glob(f"{out_dir}/*.png")) >= n_per_band*3: return
    for lo, hi in [(0.01,0.05), (0.05,0.12), (0.12,0.22)]:
        for s in range(n_per_band):
            rng = np.random.default_rng(1000*int(hi*100)+s)
            S = 320
            Fr = np.fft.fftshift(np.fft.fft2(rng.standard_normal((S,S))))
            u = np.fft.fftshift(np.fft.fftfreq(S)); r = np.hypot(*np.meshgrid(u,u,indexing='ij'))
            b = np.real(np.fft.ifft2(np.fft.ifftshift(Fr*((r>=lo)&(r<=hi)))))
            b = (b-b.mean())/(b.std()+1e-8)
            patch = (np.clip(0.5+0.35*b,0,1)[...,None].repeat(3,2)*255).astype(np.uint8)
            img = np.full((512,512,3),128,np.uint8); img[96:96+S,96:96+S] = patch
            cv2.imwrite(f"{out_dir}/noise_{lo:.2f}_{s}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
make_noise_probe(CFG["noise_probe_dir"])
noise_probe = sorted(glob.glob(f"{CFG['noise_probe_dir']}/*.png"))
print(f"held out noise probe: {len(noise_probe)} images (NOT in training)")

# ---------- data ----------
def list_pairs(img_dirs, mask_dirs):
    pairs = []
    for img_dir, mask_dir in zip(img_dirs, mask_dirs):
        if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
            print(f"skip: {img_dir}"); continue
        mm = {os.path.splitext(f)[0]: os.path.join(mask_dir,f)
              for f in os.listdir(mask_dir) if f.lower().endswith('.png')}
        for f in os.listdir(img_dir):
            b = os.path.splitext(f)[0]
            if b in mm: pairs.append((os.path.join(img_dir,f), mm[b]))
    return pairs

EXTS = ('.jpg','.jpeg','.jfif','.png','.bmp','.webp')
def list_imgs(d):
    return [(os.path.join(d,f), None) for f in sorted(os.listdir(d))
            if f.lower().endswith(EXTS)] if os.path.isdir(d) else []

positives = list_pairs(CFG["pos_image_dirs"], CFG["pos_mask_dirs"])
negatives = []
for d in CFG["neg_image_dirs"]:
    got = list_imgs(d); print(f"  neg '{d}': {len(got)}"); negatives += got

# fur is still loaded and split with the same seed, so fur_val is the SAME ten images
fur = list_imgs(CFG["fur_dir"]); random.shuffle(fur)
k = max(1, int(len(fur)*CFG["fur_val_frac"]))
fur_val, fur_tr = fur[:k], fur[k:]
print(f"  fur: {len(fur_tr)} available / {len(fur_val)} HELD OUT")

random.shuffle(positives); random.shuffle(negatives)
def split(l, f): k = max(1,int(len(l)*f)); return l[k:], l[:k]
pos_tr, pos_val = split(positives, CFG["val_frac"])
neg_tr, neg_val = split(negatives, CFG["val_frac"])

if ADD_FUR:
    neg_tr += fur_tr
    print("  >> fur negatives ADDED to training")
else:
    print(f"  >> CONTROL ARM: the {len(fur_tr)} fur training images are NOT added to training")
print(f"train pos {len(pos_tr)} | train neg {len(neg_tr)}")

def load_rgb(p):
    im = cv2.imread(p)
    if im is None: raise ValueError(p)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
def load_mask(p, hw):
    if p is None: return np.zeros(hw, np.uint8)
    return (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

class DS(Dataset):
    def __init__(self, pos, neg, cfg, is_train):
        self.samples = pos+neg; self.pos = pos; self.neg = neg
        self.is_train = is_train; self.cfg = cfg
        H, W = cfg["image_size"]
        self.tf = A.Compose([A.RandomResizedCrop(size=(H,W), scale=(0.8,1.0), p=1.0),
                             A.HorizontalFlip(p=0.5),
                             A.ColorJitter(brightness=0.2, contrast=0.2, p=0.4),
                             A.Normalize(mean=MEAN, std=STD)]) if is_train else \
                  A.Compose([A.Resize(H,W), A.Normalize(mean=MEAN, std=STD)])
    def __len__(self): return len(self.samples)
    def _copy_paste(self):
        base_p,_ = random.choice(self.neg) if self.neg else random.choice(self.pos)
        base = load_rgb(base_p); Hb,Wb = base.shape[:2]; mask = np.zeros((Hb,Wb), np.uint8)
        for _ in range(random.randint(*self.cfg["cp_instances"])):
            sp_img, sp_mask = random.choice(self.pos)
            src = load_rgb(sp_img); m = load_mask(sp_mask, src.shape[:2])
            ys,xs = np.where(m>0)
            if len(ys)==0: continue
            inst = src[ys.min():ys.max()+1, xs.min():xs.max()+1]
            instm = m[ys.min():ys.max()+1, xs.min():xs.max()+1]
            s = random.uniform(*self.cfg["cp_scale"])
            nh,nw = max(8,int(inst.shape[0]*s)), max(8,int(inst.shape[1]*s))
            if nh>=Hb or nw>=Wb:
                sc = min(Hb/nh, Wb/nw)*0.8; nh,nw = int(nh*sc), int(nw*sc)
            if nh<4 or nw<4: continue
            inst = cv2.resize(inst,(nw,nh)); instm = cv2.resize(instm,(nw,nh), interpolation=cv2.INTER_NEAREST)
            ry,rx = random.randint(0,Hb-nh), random.randint(0,Wb-nw); sel = instm>0
            base[ry:ry+nh, rx:rx+nw][sel] = inst[sel]; mask[ry:ry+nh, rx:rx+nw][sel] = 1
        return base, mask
    def __getitem__(self, i):
        try:
            if self.is_train and self.pos and random.random() < self.cfg["p_copy_paste"]:
                image, mask = self._copy_paste()
            else:
                ip, mp = self.samples[i]; image = load_rgb(ip); mask = load_mask(mp, image.shape[:2])
            a = self.tf(image=image, mask=mask)
            return {"pixel_values": torch.from_numpy(a["image"].transpose(2,0,1)).float(),
                    "labels": torch.from_numpy(a["mask"]).long()}
        except Exception:
            return self.__getitem__((i+1) % len(self.samples))

train_ds = DS(pos_tr, neg_tr, CFG, True)
w = [3.0 if s[1] is not None else 2.0 for s in train_ds.samples]
train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                          sampler=WeightedRandomSampler(w, len(w), replacement=True),
                          num_workers=2, drop_last=True)

# ---------- losses ----------
cw = torch.tensor(CFG["class_weights"], dtype=torch.float32, device=device)
ce = nn.CrossEntropyLoss(weight=cw)
def dice_loss(lo, t):
    p = torch.softmax(lo,1)[:,1].float(); g = (t==1).float()
    return 1 - (2*(p*g).sum()+EPS)/(p.sum()+g.sum()+EPS)
def dice_iou_hard(lo, t):
    p = (torch.argmax(lo,1)==1).float(); g = (t==1).float(); i = (p*g).sum()
    return ((2*i+EPS)/(p.sum()+g.sum()+EPS)).item(), ((i+EPS)/(p.sum()+g.sum()-i+EPS)).item()
_kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).view(1,1,3,3)
_ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).view(1,1,3,3)
def edge_map(x):
    g = x.mean(1, keepdim=True)
    e = torch.sqrt(F.conv2d(g,_kx,padding=1)**2 + F.conv2d(g,_ky,padding=1)**2 + 1e-6)
    return (e/e.amax(dim=(2,3), keepdim=True).clamp_min(1e-6)).squeeze(1)
def seam_fp(lo, m, x, beta):
    prob = torch.softmax(lo,1)[:,1]; benign = (m==0).float()
    return ((1.0+beta*edge_map(x))*prob*benign).sum()/benign.sum().clamp_min(1.0)
def total_loss(lo, m, x):
    return ce(lo,m) + CFG["dice_factor"]*dice_loss(lo,m) + CFG["seam_alpha"]*seam_fp(lo,m,x,CFG["seam_beta"])

# ---------- eval ----------
def preprocess(p, hw):
    im = cv2.resize(load_rgb(p), (hw[1],hw[0])).astype(np.float32)/255.0
    im = (im-np.array(MEAN,np.float32))/np.array(STD,np.float32)
    return torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0).float()

@torch.no_grad()
def fire_rate(model, items, hw, thr=0.5, area=0.001):
    if not items: return float('nan')
    fired = 0
    for p, _ in items:
        lo = F.interpolate(model(preprocess(p,hw).to(device)).logits, size=hw,
                           mode="bilinear", align_corners=False)
        if (torch.softmax(lo,1)[0,1].cpu().numpy() >= thr).mean() > area: fired += 1
    return fired/len(items)

@torch.no_grad()
def validate(model, hw, thr=0.5):
    model.eval(); dices, ious = [], []
    for ip, mp in pos_val:
        gt = cv2.resize(load_mask(mp, load_rgb(ip).shape[:2]), (hw[1],hw[0]), interpolation=cv2.INTER_NEAREST)
        lo = F.interpolate(model(preprocess(ip,hw).to(device)).logits, size=hw,
                           mode="bilinear", align_corners=False)
        pr = (torch.softmax(lo,1)[0,1].cpu().numpy() >= thr).astype(np.uint8)
        tp = (pr&gt).sum(); fp = (pr&(1-gt)).sum(); fn = ((1-pr)&gt).sum()
        dices.append((2*tp+EPS)/(2*tp+fp+fn+EPS)); ious.append((tp+EPS)/(tp+fp+fn+EPS))
    return (float(np.mean(dices)) if dices else 0.0,
            float(np.mean(ious)) if ious else 0.0,
            fire_rate(model, neg_val, hw, thr, CFG["fire_area"]),
            fire_rate(model, fur_val, hw, thr, CFG["fire_area"]),
            fire_rate(model, [(p,None) for p in noise_probe], hw, thr, CFG["fire_area"]))

# ---------- model ----------
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b3", num_labels=2, ignore_mismatched_sizes=True).to(device)
sd = torch.load(CFG["resume_ckpt"], map_location=device, weights_only=True)
sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
res = model.load_state_dict(sd, strict=False)
rm = [k for k in res.missing_keys if not k.startswith("classifier.")]
assert len(rm) <= 5, f"ep8 weights did not load, {len(rm)} missing: {rm[:8]}"
print(f"warm start {CFG['resume_ckpt']} | real missing {len(rm)}")

opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])
os.makedirs(CFG["out_dir"], exist_ok=True)
with open(CFG["log_csv"], "w", newline="") as f:
    csv.writer(f).writerow(["datetime","epoch","train_loss","train_dice",
                            "val_dice","val_iou","fpr_old_negs","fpr_fur_heldout","fpr_noise_heldout","balanced"])

HW = CFG["image_size"]
d0, i0, n0, f0, z0 = validate(model, HW)
print(f"\nEPOCH 0 (ep8, before continuation)  val_dice {d0:.4f} | old neg FPR {n0:.3f} | "
      f"FUR(held out) {f0:.3f} | NOISE(held out) {z0:.3f}")

best = -1e9
for ep in range(CFG["total_epochs"]):
    model.train(); tl, td = 0.0, []
    pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{CFG['total_epochs']} [CONTROL]")
    for b in pbar:
        x = b["pixel_values"].to(device); m = b["labels"].to(device)
        opt.zero_grad()
        lo = F.interpolate(model(x).logits, size=m.shape[1:], mode="bilinear", align_corners=False)
        loss = total_loss(lo, m, x); loss.backward(); opt.step()
        dd, _ = dice_iou_hard(lo, m); tl += loss.item(); td.append(dd)
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "dice": f"{dd:.3f}"})

    vd, vi, fn_, ff, fz = validate(model, HW)
    bal = vd - CFG["fpr_lambda"]*fn_
    with open(CFG["log_csv"], "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ep+1,
                                tl/len(train_loader), float(np.mean(td)), vd, vi, fn_, ff, fz, bal])
    print(f"\n==== EPOCH {ep+1} [CONTROL, no fur negatives] ====")
    print(f"  val Dice {vd:.4f} | IoU {vi:.4f} | old neg FPR {fn_:.3f}")
    print(f"  >> FUR   (held out, NOT supplied here)  {f0:.3f} -> {ff:.3f}")
    print(f"  >> NOISE (held out, never supplied)     {z0:.3f} -> {fz:.3f}")
    torch.save(model.state_dict(), f"{CFG['out_dir']}/smashed_control_ep{ep+1}.pth")
    if bal > best:
        best = bal; torch.save(model.state_dict(), f"{CFG['out_dir']}/smashed_control_BEST.pth")
        print(f"  >> new BEST (balanced {bal:.4f})")

print("\n=== CONTROL ARM RESULT ===")
print(f"val Dice {d0:.4f} -> {vd:.4f}   (if this still rises, the gain in the fur run was continued training)")
print(f"fur      {f0:.3f} -> {ff:.3f}   (should STAY HIGH: fur was never supplied in this arm)")
print(f"noise    {z0:.3f} -> {fz:.3f}   (should stay at 1.00, as in the fur run)")
print("Fur staying high here while it collapsed in the fur run is what attributes the collapse to the negatives.")