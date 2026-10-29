# The seeded replication (SS7, Table 2): seeds 1-3 x {fur, control}, fur
# split frozen at seed 0, with the per-source seam-term logging behind
# Appendix D. Requires the unreleased corpus; included for transparency.

# =========================================================
# SEED REPLICATION: 3 seeds x {fur negatives, control}
# Fur split frozen at seed 0 so the probe is identical across runs.
# Sampler draw count fixed so both arms consume RNG identically.
# NEW: per image seam loss logged by source in epoch 1, to test whether
# fur negatives deliver a much larger gradient than ordinary negatives.
# =========================================================
import os, csv, random, glob, time
from datetime import datetime
import cv2, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

CFG = {
    "resume_ckpt": "checkpoints_smashed_scratchfix_aug/smashed_scratchfix_ep8.pth",
    "total_epochs": 4,                 # 6 if you can spare the hours
    "image_size": (512, 512), "batch_size": 4, "lr": 2e-5,
    "class_weights": [0.1281, 1.8719], "dice_factor": 2.0,
    "p_copy_paste": 0.25, "cp_instances": (1, 2), "cp_scale": (0.6, 1.3),
    "seam_alpha": 0.75, "seam_beta": 4.0, "fire_area": 0.001, "val_frac": 0.12,
    "pos_image_dirs": ['extracted_smashed_data/smashed_batch1/images',
                       'extracted_smashed_data/smashed_batch2/images'],
    "pos_mask_dirs":  ['smashed_masks_batch1', 'smashed_masks_batch2'],
    "neg_image_dirs": ['background_smashed0804', 'background', 'falsesmashed', 'scratchedforsmashed'],
    "fur_dir": "fur", "fur_extra_dir": "fur_extra", "noise_probe_dir": "noise_probe",
    "out_root": "checkpoints_seeds", "log_csv": "seed_replication_log.csv",
    "seam_csv": "seam_by_source.csv",
}
SEEDS = [1, 2, 3]
ARMS  = [("fur", True), ("control", False)]
SRC = {0: "positive", 1: "negative_other", 2: "negative_FUR", 3: "copy_paste"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = (0.485, 0.456, 0.406); STD = (0.229, 0.224, 0.225); EPS = 1e-7
EXTS = ('.jpg','.jpeg','.jfif','.png','.bmp','.webp')

def list_imgs(d):
    return [(os.path.join(d,f), None) for f in sorted(os.listdir(d))
            if f.lower().endswith(EXTS)] if os.path.isdir(d) else []

def list_pairs(img_dirs, mask_dirs):
    pairs = []
    for idir, mdir in zip(img_dirs, mask_dirs):
        if not (os.path.isdir(idir) and os.path.isdir(mdir)): continue
        mm = {os.path.splitext(f)[0]: os.path.join(mdir,f)
              for f in os.listdir(mdir) if f.lower().endswith('.png')}
        for f in os.listdir(idir):
            b = os.path.splitext(f)[0]
            if b in mm: pairs.append((os.path.join(idir,f), mm[b]))
    return pairs

# ---------- FROZEN at seed 0: identical across every run ----------
positives = list_pairs(CFG["pos_image_dirs"], CFG["pos_mask_dirs"])
negatives = []
for d in CFG["neg_image_dirs"]: negatives += list_imgs(d)
fur_all = list_imgs(CFG["fur_dir"])

random.seed(0); np.random.seed(0); torch.manual_seed(0)
random.shuffle(fur_all)
_k = max(1, int(len(fur_all)*0.25))
fur_val_orig, FUR_TRAIN = fur_all[:_k], fur_all[_k:]
random.shuffle(positives); random.shuffle(negatives)
def split(l, f): k = max(1,int(len(l)*f)); return l[k:], l[:k]
POS_TR, POS_VAL = split(positives, CFG["val_frac"])
NEG_TR, NEG_VAL = split(negatives, CFG["val_frac"])
FUR_EVAL = fur_val_orig + list_imgs(CFG["fur_extra_dir"])
NOISE    = [(p, None) for p in sorted(glob.glob(f"{CFG['noise_probe_dir']}/*.png"))]
FUR_PATHS = {p for p,_ in FUR_TRAIN}
trained  = {os.path.basename(p) for p in FUR_PATHS}
assert not [p for p,_ in FUR_EVAL if os.path.basename(p) in trained], "CONTAMINATION"
N_DRAWS = len(POS_TR) + len(NEG_TR)        # fixed for BOTH arms

print(f"pos {len(POS_TR)}/{len(POS_VAL)} | neg {len(NEG_TR)}/{len(NEG_VAL)}")
print(f"fur supplied {len(FUR_TRAIN)} | fur eval {len(FUR_EVAL)} | noise {len(NOISE)}")
print(f"sampler draws per epoch, both arms: {N_DRAWS}")

def load_rgb(p):
    im = cv2.imread(p); assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
def load_mask(p, hw):
    return np.zeros(hw, np.uint8) if p is None else (cv2.imread(p, 0) > 0).astype(np.uint8)

class DS(Dataset):
    def __init__(self, pos, neg, cfg):
        self.samples = pos+neg; self.pos = pos; self.neg = neg; self.cfg = cfg
        H, W = cfg["image_size"]
        self.tf = A.Compose([A.RandomResizedCrop(size=(H,W), scale=(0.8,1.0), p=1.0),
                             A.HorizontalFlip(p=0.5),
                             A.ColorJitter(brightness=0.2, contrast=0.2, p=0.4),
                             A.Normalize(mean=MEAN, std=STD)])
    def __len__(self): return len(self.samples)
    def _cp(self):
        base_p,_ = random.choice(self.neg) if self.neg else random.choice(self.pos)
        base = load_rgb(base_p); Hb,Wb = base.shape[:2]; mask = np.zeros((Hb,Wb), np.uint8)
        for _ in range(random.randint(*self.cfg["cp_instances"])):
            si, sm = random.choice(self.pos)
            src = load_rgb(si); m = load_mask(sm, src.shape[:2])
            ys,xs = np.where(m>0)
            if len(ys)==0: continue
            inst = src[ys.min():ys.max()+1, xs.min():xs.max()+1]
            im_  = m[ys.min():ys.max()+1, xs.min():xs.max()+1]
            s = random.uniform(*self.cfg["cp_scale"])
            nh,nw = max(8,int(inst.shape[0]*s)), max(8,int(inst.shape[1]*s))
            if nh>=Hb or nw>=Wb:
                sc = min(Hb/nh, Wb/nw)*0.8; nh,nw = int(nh*sc), int(nw*sc)
            if nh<4 or nw<4: continue
            inst = cv2.resize(inst,(nw,nh)); im_ = cv2.resize(im_,(nw,nh), interpolation=cv2.INTER_NEAREST)
            ry,rx = random.randint(0,Hb-nh), random.randint(0,Wb-nw); sel = im_>0
            base[ry:ry+nh, rx:rx+nw][sel] = inst[sel]; mask[ry:ry+nh, rx:rx+nw][sel] = 1
        return base, mask
    def __getitem__(self, i):
        try:
            if self.pos and random.random() < self.cfg["p_copy_paste"]:
                img, msk = self._cp(); src = 3
            else:
                ip, mp = self.samples[i]
                img = load_rgb(ip); msk = load_mask(mp, img.shape[:2])
                src = 0 if mp is not None else (2 if ip in FUR_PATHS else 1)
            a = self.tf(image=img, mask=msk)
            return {"pixel_values": torch.from_numpy(a["image"].transpose(2,0,1)).float(),
                    "labels": torch.from_numpy(a["mask"]).long(),
                    "src": torch.tensor(src, dtype=torch.long)}
        except Exception:
            return self.__getitem__((i+1) % len(self.samples))

cw = torch.tensor(CFG["class_weights"], dtype=torch.float32, device=device)
ce = nn.CrossEntropyLoss(weight=cw)
_kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).view(1,1,3,3)
_ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).view(1,1,3,3)
def dice_loss(lo,t):
    p=torch.softmax(lo,1)[:,1].float(); g=(t==1).float()
    return 1-(2*(p*g).sum()+EPS)/(p.sum()+g.sum()+EPS)
def edge_map(x):
    g=x.mean(1,keepdim=True)
    e=torch.sqrt(F.conv2d(g,_kx,padding=1)**2+F.conv2d(g,_ky,padding=1)**2+1e-6)
    return (e/e.amax(dim=(2,3),keepdim=True).clamp_min(1e-6)).squeeze(1)
def seam_per_image(lo,m,x):
    """the seam FP term, kept per image so it can be attributed to a source"""
    prob=torch.softmax(lo,1)[:,1]; ben=(m==0).float()
    return ((1.0+CFG["seam_beta"]*edge_map(x))*prob*ben).sum(dim=(1,2))/ben.sum(dim=(1,2)).clamp_min(1.0)
def total_loss(lo,m,x):
    return ce(lo,m)+CFG["dice_factor"]*dice_loss(lo,m)+CFG["seam_alpha"]*seam_per_image(lo,m,x).mean()

def prep(p, hw):
    im = cv2.resize(load_rgb(p), (hw[1],hw[0])).astype(np.float32)/255.
    im = (im-np.array(MEAN,np.float32))/np.array(STD,np.float32)
    return torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0).float()

@torch.no_grad()
def probe(model, items, hw, thr=0.5):
    if not items: return float('nan'), float('nan'), float('nan')
    fired, mx = 0, []
    for p,_ in items:
        lo = F.interpolate(model(prep(p,hw).to(device)).logits, size=hw, mode="bilinear", align_corners=False)
        v  = torch.softmax(lo,1)[0,1].cpu().numpy()
        mx.append(float(v.max()))
        if (v >= thr).mean() > CFG["fire_area"]: fired += 1
    return fired/len(items), float(np.mean(mx)), float(np.min(mx))

@torch.no_grad()
def validate(model, hw, thr=0.5):
    model.eval(); dices=[]
    for ip, mp in POS_VAL:
        gt = cv2.resize(load_mask(mp, load_rgb(ip).shape[:2]), (hw[1],hw[0]), interpolation=cv2.INTER_NEAREST)
        lo = F.interpolate(model(prep(ip,hw).to(device)).logits, size=hw, mode="bilinear", align_corners=False)
        pr = (torch.softmax(lo,1)[0,1].cpu().numpy() >= thr).astype(np.uint8)
        tp=(pr&gt).sum(); fp=(pr&(1-gt)).sum(); fn=((1-pr)&gt).sum()
        dices.append((2*tp+EPS)/(2*tp+fp+fn+EPS))
    vd = float(np.mean(dices))
    negf,_,_          = probe(model, NEG_VAL, hw)
    furf,furm,furmin  = probe(model, FUR_EVAL, hw)
    noif,noim,noimin  = probe(model, NOISE, hw)
    return vd, negf, furf, furm, furmin, noif, noim, noimin

HW = CFG["image_size"]
os.makedirs(CFG["out_root"], exist_ok=True)
with open(CFG["log_csv"], "w", newline="") as f:
    csv.writer(f).writerow(["datetime","seed","arm","epoch","train_loss","val_dice","fpr_old_negs",
                            "fur_rate","fur_maxprob_mean","fur_maxprob_min",
                            "noise_rate","noise_maxprob_mean","noise_maxprob_min"])
with open(CFG["seam_csv"], "w", newline="") as f:
    csv.writer(f).writerow(["seed","arm","epoch","step","source","seam_loss"])

t_all = time.time()
for seed in SEEDS:
    for arm, add_fur in ARMS:
        tag = f"seed{seed}_{arm}"; out = os.path.join(CFG["out_root"], tag)
        os.makedirs(out, exist_ok=True)
        print(f"\n{'='*70}\n  {tag}   (fur negatives: {add_fur})\n{'='*70}")

        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        neg_tr = list(NEG_TR) + (list(FUR_TRAIN) if add_fur else [])
        ds = DS(POS_TR, neg_tr, CFG)
        w  = [3.0 if s[1] is not None else 2.0 for s in ds.samples]
        g  = torch.Generator(); g.manual_seed(seed)
        loader = DataLoader(ds, batch_size=CFG["batch_size"],
                            sampler=WeightedRandomSampler(w, N_DRAWS, replacement=True, generator=g),
                            num_workers=2, drop_last=True,
                            generator=g, worker_init_fn=lambda i: np.random.seed(seed+i))

        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b3", num_labels=2, ignore_mismatched_sizes=True).to(device)
        sd = torch.load(CFG["resume_ckpt"], map_location=device, weights_only=True)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        r  = model.load_state_dict(sd, strict=False)
        assert len([k for k in r.missing_keys if not k.startswith("classifier.")]) <= 5, "bad warm start"
        opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"])

        if seed == SEEDS[0] and arm == "fur":
            v = validate(model, HW)
            print(f"  epoch 0 (ep8): dice {v[0]:.4f} | fur {v[2]:.3f} (maxp {v[3]:.4f}) "
                  f"| noise {v[5]:.3f} (maxp {v[6]:.4f})")

        for ep in range(CFG["total_epochs"]):
            model.train(); tl = 0.0; seam_rows = []
            for step, b in enumerate(tqdm(loader, desc=f"{tag} e{ep+1}", ncols=88)):
                x = b["pixel_values"].to(device); m = b["labels"].to(device)
                opt.zero_grad()
                lo = F.interpolate(model(x).logits, size=m.shape[1:], mode="bilinear", align_corners=False)
                loss = total_loss(lo, m, x); loss.backward(); opt.step(); tl += loss.item()
                if ep == 0:                       # attribute the seam term in epoch 1 only
                    with torch.no_grad():
                        s_ = seam_per_image(lo, m, x).cpu().numpy()
                    for si, sv in zip(b["src"].numpy(), s_):
                        seam_rows.append([seed, arm, ep+1, step, SRC[int(si)], float(sv)])
            if seam_rows:
                with open(CFG["seam_csv"], "a", newline="") as f:
                    csv.writer(f).writerows(seam_rows)
                sdf = pd.DataFrame(seam_rows, columns=["seed","arm","epoch","step","source","seam_loss"])
                print("  seam loss by source, epoch 1:")
                print(sdf.groupby("source").seam_loss.agg(["count","mean","median","max"]).round(4).to_string())

            vd, negf, furf, furm, furmin, noif, noim, noimin = validate(model, HW)
            with open(CFG["log_csv"], "a", newline="") as f:
                csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), seed, arm, ep+1,
                                        tl/len(loader), vd, negf, furf, furm, furmin, noif, noim, noimin])
            print(f"  e{ep+1}: dice {vd:.4f} | oldneg {negf:.3f} | FUR {furf:.3f} (maxp {furm:.4f}) "
                  f"| NOISE {noif:.3f} (maxp {noim:.4f})")
            torch.save(model.state_dict(), f"{out}/ep{ep+1}.pth")
        del model, loader, ds; torch.cuda.empty_cache()

print(f"\nall runs done in {(time.time()-t_all)/3600:.1f} h")

d = pd.read_csv(CFG["log_csv"])
for col in ["fur_rate","fur_maxprob_mean","noise_rate","noise_maxprob_mean","val_dice","fpr_old_negs"]:
    print(f"\n=== {col} ===")
    print(d.pivot_table(index="epoch", columns=["arm","seed"], values=col).round(4).to_string())
print(f"\n=== final epoch, mean +/- sd across seeds ===")
last = d[d.epoch == CFG["total_epochs"]]
print(last.groupby("arm")[["fur_rate","fur_maxprob_mean","noise_rate","noise_maxprob_mean","val_dice"]]
        .agg(["mean","std"]).round(4).to_string())

s = pd.read_csv(CFG["seam_csv"])
print("\n=== seam loss by source, epoch 1, fur arm only ===")
print(s[s.arm=="fur"].groupby(["seed","source"]).seam_loss
        .agg(["count","mean","median"]).round(4).to_string())
print("\nIf negative_FUR averages several times negative_other, the thirty fur images are")
print("delivering a disproportionate gradient through the seam term, which would explain the")
print("global silencing rather than any family specific learning.")