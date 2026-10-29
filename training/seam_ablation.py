# The corrective-loss ablation (Appendix H): beta=0 and alpha=0 arms from
# the same checkpoint and splits, seed 1. Requires the unreleased corpus;
# included for transparency.

# =========================================================
# SEAM TERM ABLATION
#   beta0  : alpha=0.75, beta=0   FP penalty kept, edge weighting removed
#   alpha0 : alpha=0,    beta=0   no seam term at all
# Both supply the same 30 fur negatives as the reported run (alpha=0.75, beta=4).
# Seed 1; positive and negative splits frozen at seed 0 as in the replication.
# All outputs under ablation_test/.   ~4.5 h per arm on a T4.
#
# Constraints enforced by assertion, each learned the hard way:
#   - transformers 5.13.0, else checkpoints silently half load
#   - state dict overlap > 600/644 on the warm start
#   - fur split reconstructed over the ORIGINAL 40 .jfif only, giving 30/10,
#     because fur/ now also holds the 49 images collected after training
#   - benchmark is exactly the 280 images the paper reports, fur family
#     restricted to the original 40
#   - held out fur is disjoint from the 30 supplied
#   - sampler draw count pinned so both arms consume RNG identically
# =========================================================
import os, csv, random, glob, time, hashlib
from datetime import datetime
import cv2, numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
import transformers
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

print("transformers", transformers.__version__)
assert transformers.__version__.startswith("5.13"), \
    "run  !pip install -q transformers==5.13.0  then RESTART the kernel"

ROOT = "ablation_test"
os.makedirs(f"{ROOT}/checkpoints", exist_ok=True)
os.makedirs(f"{ROOT}/logs", exist_ok=True)

SETC = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/bosene1/code/Users/bosene/contextShortcutData"
EXTS = ('.jpg','.jpeg','.jfif','.png','.bmp','.webp')

CFG = {
    "resume_ckpt": "checkpoints_smashed_scratchfix_aug/smashed_scratchfix_ep8.pth",
    "total_epochs": 4, "image_size": (512,512), "batch_size": 4, "lr": 2e-5,
    "class_weights": [0.1281, 1.8719], "dice_factor": 2.0,
    "p_copy_paste": 0.25, "cp_instances": (1,2), "cp_scale": (0.6,1.3),
    "fire_area": 0.001, "val_frac": 0.12,
    "pos_image_dirs": ['extracted_smashed_data/smashed_batch1/images',
                       'extracted_smashed_data/smashed_batch2/images'],
    "pos_mask_dirs":  ['smashed_masks_batch1','smashed_masks_batch2'],
    "neg_image_dirs": ['background_smashed0804','background','falsesmashed','scratchedforsmashed'],
    "fur_dir": "fur", "noise_probe_dir": "noise_probe",
    "log_csv":  f"{ROOT}/logs/ablation_log.csv",
    "seam_csv": f"{ROOT}/logs/ablation_seam.csv",
    "setc_csv": f"{ROOT}/logs/setC_per_image.csv",
}
SEED = 1
ARMS = [("beta0", 0.75, 0.0), ("alpha0", 0.0, 0.0)]
SRC  = {0:"positive", 1:"negative_other", 2:"negative_FUR", 3:"copy_paste"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225); EPS=1e-7

# ---------- benchmark: exactly the 280 the paper reports ----------
FAMS = ["crackedmud","fur","gravel","raindrop","smash.original","wall"]
SETC_IMGS = []
print("\nbenchmark:")
for fam in FAMS:
    ps = sorted(p for p in glob.glob(f"{SETC}/{fam}/*") if p.lower().endswith(EXTS))
    if fam == "fur":
        ps = [p for p in ps if p.lower().endswith('.jfif')]      # original 40 only
    SETC_IMGS += ps
    print(f"   {fam:<16} {len(ps):>4}")
print(f"   {'TOTAL':<16} {len(SETC_IMGS):>4}")
assert len(SETC_IMGS) == 280, f"expected 280, got {len(SETC_IMGS)}"

def list_imgs(d):
    return [(os.path.join(d,f), None) for f in sorted(os.listdir(d))
            if f.lower().endswith(EXTS)] if os.path.isdir(d) else []
def list_pairs(idirs, mdirs):
    out=[]
    for idir,mdir in zip(idirs,mdirs):
        if not (os.path.isdir(idir) and os.path.isdir(mdir)): continue
        mm={os.path.splitext(f)[0]: os.path.join(mdir,f)
            for f in os.listdir(mdir) if f.lower().endswith('.png')}
        for f in os.listdir(idir):
            b=os.path.splitext(f)[0]
            if b in mm: out.append((os.path.join(idir,f), mm[b]))
    return out

# ---------- fur split: over the ORIGINAL 40 only, giving 30 / 10 ----------
fur_orig  = [(p,None) for p in sorted(glob.glob(f"{CFG['fur_dir']}/*"))
             if p.lower().endswith('.jfif')]
fur_later = [(p,None) for p in sorted(glob.glob(f"{CFG['fur_dir']}/*"))
             if p.lower().endswith(EXTS) and not p.lower().endswith('.jfif')]
assert len(fur_orig) == 40, f"expected 40 original .jfif fur images, found {len(fur_orig)}"
random.seed(0); np.random.seed(0); torch.manual_seed(0)
random.shuffle(fur_orig)
_k = max(1, int(len(fur_orig)*0.25))
fur_val_orig, FUR_TRAIN = fur_orig[:_k], fur_orig[_k:]
assert len(FUR_TRAIN)==30 and len(fur_val_orig)==10, \
    f"split is {len(FUR_TRAIN)}/{len(fur_val_orig)}, expected 30/10"
FUR_EVAL = fur_val_orig + fur_later
print(f"\nfur: {len(FUR_TRAIN)} supplied | {len(FUR_EVAL)} held out "
      f"({len(fur_val_orig)} original + {len(fur_later)} later)")

# ---------- positive and negative splits, frozen at seed 0 ----------
positives=list_pairs(CFG["pos_image_dirs"], CFG["pos_mask_dirs"])
negatives=[]
for d in CFG["neg_image_dirs"]: negatives+=list_imgs(d)
random.seed(0); np.random.seed(0); torch.manual_seed(0)
random.shuffle(positives); random.shuffle(negatives)
def split(l,f): k=max(1,int(len(l)*f)); return l[k:], l[:k]
POS_TR,POS_VAL=split(positives,CFG["val_frac"]); NEG_TR,NEG_VAL=split(negatives,CFG["val_frac"])
NOISE=[(p,None) for p in sorted(glob.glob(f"{CFG['noise_probe_dir']}/*.png"))]
FUR_PATHS={p for p,_ in FUR_TRAIN}
N_DRAWS=len(POS_TR)+len(NEG_TR)

def _md5(p):
    with open(p,'rb') as f: return hashlib.md5(f.read()).hexdigest()
_sup = {_md5(p) for p,_ in FUR_TRAIN}
leak = [p for p,_ in FUR_EVAL if _md5(p) in _sup]
assert not leak, f"CONTAMINATION: {leak[:3]}"
print(f"pos {len(POS_TR)}/{len(POS_VAL)} | neg {len(NEG_TR)}/{len(NEG_VAL)} | "
      f"noise {len(NOISE)} | draws {N_DRAWS} | no fur leakage")

def load_rgb(p):
    im=cv2.imread(p); assert im is not None, p
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
def load_mask(p,hw):
    return np.zeros(hw,np.uint8) if p is None else (cv2.imread(p,0)>0).astype(np.uint8)

class DS(Dataset):
    def __init__(self,pos,neg,cfg):
        self.samples=pos+neg; self.pos=pos; self.neg=neg; self.cfg=cfg
        H,W=cfg["image_size"]
        self.tf=A.Compose([A.RandomResizedCrop(size=(H,W),scale=(0.8,1.0),p=1.0),
                           A.HorizontalFlip(p=0.5),
                           A.ColorJitter(brightness=0.2,contrast=0.2,p=0.4),
                           A.Normalize(mean=MEAN,std=STD)])
    def __len__(self): return len(self.samples)
    def _cp(self):
        bp,_=random.choice(self.neg) if self.neg else random.choice(self.pos)
        base=load_rgb(bp); Hb,Wb=base.shape[:2]; mask=np.zeros((Hb,Wb),np.uint8)
        for _ in range(random.randint(*self.cfg["cp_instances"])):
            si,sm=random.choice(self.pos); src=load_rgb(si); m=load_mask(sm,src.shape[:2])
            ys,xs=np.where(m>0)
            if len(ys)==0: continue
            inst=src[ys.min():ys.max()+1, xs.min():xs.max()+1]
            im_=m[ys.min():ys.max()+1, xs.min():xs.max()+1]
            sc=random.uniform(*self.cfg["cp_scale"])
            nh,nw=max(8,int(inst.shape[0]*sc)), max(8,int(inst.shape[1]*sc))
            if nh>=Hb or nw>=Wb:
                f=min(Hb/nh,Wb/nw)*0.8; nh,nw=int(nh*f),int(nw*f)
            if nh<4 or nw<4: continue
            inst=cv2.resize(inst,(nw,nh)); im_=cv2.resize(im_,(nw,nh),interpolation=cv2.INTER_NEAREST)
            ry,rx=random.randint(0,Hb-nh), random.randint(0,Wb-nw); sel=im_>0
            base[ry:ry+nh, rx:rx+nw][sel]=inst[sel]; mask[ry:ry+nh, rx:rx+nw][sel]=1
        return base,mask
    def __getitem__(self,i):
        try:
            if self.pos and random.random()<self.cfg["p_copy_paste"]:
                img,msk=self._cp(); src=3
            else:
                ip,mp=self.samples[i]; img=load_rgb(ip); msk=load_mask(mp,img.shape[:2])
                src=0 if mp is not None else (2 if ip in FUR_PATHS else 1)
            a=self.tf(image=img,mask=msk)
            return {"pixel_values":torch.from_numpy(a["image"].transpose(2,0,1)).float(),
                    "labels":torch.from_numpy(a["mask"]).long(),
                    "src":torch.tensor(src,dtype=torch.long)}
        except Exception:
            return self.__getitem__((i+1)%len(self.samples))

cw=torch.tensor(CFG["class_weights"],dtype=torch.float32,device=device)
ce=nn.CrossEntropyLoss(weight=cw)
_kx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=torch.float32,device=device).view(1,1,3,3)
_ky=torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=torch.float32,device=device).view(1,1,3,3)
def dice_loss(lo,t):
    p=torch.softmax(lo,1)[:,1].float(); g=(t==1).float()
    return 1-(2*(p*g).sum()+EPS)/(p.sum()+g.sum()+EPS)
def edge_map(x):
    g=x.mean(1,keepdim=True)
    e=torch.sqrt(F.conv2d(g,_kx,padding=1)**2+F.conv2d(g,_ky,padding=1)**2+1e-6)
    return (e/e.amax(dim=(2,3),keepdim=True).clamp_min(1e-6)).squeeze(1)
def seam_per_image(lo,m,x,beta):
    prob=torch.softmax(lo,1)[:,1]; ben=(m==0).float()
    w = 1.0 + beta*edge_map(x) if beta > 0 else torch.ones_like(prob)
    return (w*prob*ben).sum(dim=(1,2))/ben.sum(dim=(1,2)).clamp_min(1.0)

def prep(p,hw):
    im=cv2.resize(load_rgb(p),(hw[1],hw[0])).astype(np.float32)/255.
    im=(im-np.array(MEAN,np.float32))/np.array(STD,np.float32)
    return torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0).float()

@torch.no_grad()
def probe(model,items,hw,thr=0.5):
    if not items: return float('nan'),float('nan')
    fired,mx=0,[]
    for p,_ in items:
        lo=F.interpolate(model(prep(p,hw).to(device)).logits,size=hw,mode="bilinear",align_corners=False)
        v=torch.softmax(lo,1)[0,1].cpu().numpy(); mx.append(float(v.max()))
        if (v>=thr).mean()>CFG["fire_area"]: fired+=1
    return fired/len(items), float(np.mean(mx))

@torch.no_grad()
def score_setc(model,hw,arm,epoch):
    """response over the 280 image benchmark: the number the experiment turns on"""
    rows=[]
    for p in SETC_IMGS:
        lo=F.interpolate(model(prep(p,hw).to(device)).logits,size=hw,mode="bilinear",align_corners=False)
        v=torch.softmax(lo,1)[0,1].cpu().numpy()
        rows.append(dict(arm=arm, epoch=epoch, family=os.path.basename(os.path.dirname(p)),
                         file=os.path.basename(p), max_prob=float(v.max()),
                         mean_prob=float(v.mean()), fire85=int(v.max()>=0.85)))
    df=pd.DataFrame(rows)
    df.to_csv(CFG["setc_csv"], mode="a", header=not os.path.exists(CFG["setc_csv"]), index=False)
    nf = df[df.family!="fur"]          # fur is partly supplied, so report both
    return (float(df.max_prob.median()), float(df.max_prob.mean()),
            float((df.max_prob<0.01).mean()), float(nf.max_prob.median()))

@torch.no_grad()
def validate(model,hw,thr=0.5):
    model.eval(); d=[]
    for ip,mp in POS_VAL:
        gt=cv2.resize(load_mask(mp,load_rgb(ip).shape[:2]),(hw[1],hw[0]),interpolation=cv2.INTER_NEAREST)
        lo=F.interpolate(model(prep(ip,hw).to(device)).logits,size=hw,mode="bilinear",align_corners=False)
        pr=(torch.softmax(lo,1)[0,1].cpu().numpy()>=thr).astype(np.uint8)
        tp=(pr&gt).sum(); fp=(pr&(1-gt)).sum(); fn=((1-pr)&gt).sum()
        d.append((2*tp+EPS)/(2*tp+fp+fn+EPS))
    negf,_=probe(model,NEG_VAL,hw); furf,furm=probe(model,FUR_EVAL,hw); noif,noim=probe(model,NOISE,hw)
    return float(np.mean(d)),negf,furf,furm,noif,noim

HW=CFG["image_size"]
with open(CFG["log_csv"],"w",newline="") as f:
    csv.writer(f).writerow(["datetime","arm","alpha","beta","epoch","train_loss","val_dice",
                            "fpr_old_negs","fur_rate","fur_maxprob","noise_rate","noise_maxprob",
                            "setC_median","setC_mean","setC_frac_silent","setC_median_nofur"])
with open(CFG["seam_csv"],"w",newline="") as f:
    csv.writer(f).writerow(["arm","epoch","source","seam_loss"])
if os.path.exists(CFG["setc_csv"]): os.remove(CFG["setc_csv"])

t0=time.time()
for ai,(arm,alpha,beta) in enumerate(ARMS):
    out=f"{ROOT}/checkpoints/{arm}"; os.makedirs(out,exist_ok=True)
    print(f"\n{'='*70}\n  {arm}:  alpha={alpha}  beta={beta}\n{'='*70}")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

    ds=DS(POS_TR, list(NEG_TR)+list(FUR_TRAIN), CFG)
    w=[3.0 if s[1] is not None else 2.0 for s in ds.samples]
    g=torch.Generator(); g.manual_seed(SEED)
    loader=DataLoader(ds,batch_size=CFG["batch_size"],
                      sampler=WeightedRandomSampler(w,N_DRAWS,replacement=True,generator=g),
                      num_workers=2,drop_last=True,generator=g,
                      worker_init_fn=lambda i: np.random.seed(SEED+i))

    model=SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3",num_labels=2,ignore_mismatched_sizes=True).to(device)
    sd=torch.load(CFG["resume_ckpt"],map_location=device,weights_only=True)
    sd=sd.get("state_dict",sd) if isinstance(sd,dict) else sd
    ov=len(set(model.state_dict())&set(sd))
    assert ov>600, f"only {ov}/644 keys overlap: wrong transformers version"
    model.load_state_dict(sd,strict=False)
    opt=torch.optim.AdamW(model.parameters(),lr=CFG["lr"])

    if ai==0:
        m0,a0,s0,nf0 = score_setc(model,HW,"ep8_start",0)
        v0 = validate(model,HW)
        print(f"  ep8 start: setC median {m0:.4f} (excl fur {nf0:.4f}) | mean {a0:.4f} | "
              f"silent {s0:.1%} | fur {v0[2]:.3f} | noise {v0[4]:.3f}")

    for ep in range(CFG["total_epochs"]):
        model.train(); tl=0.0; rows=[]
        for b in tqdm(loader,desc=f"{arm} e{ep+1}",ncols=88):
            x=b["pixel_values"].to(device); m=b["labels"].to(device)
            opt.zero_grad()
            lo=F.interpolate(model(x).logits,size=m.shape[1:],mode="bilinear",align_corners=False)
            sp=seam_per_image(lo,m,x,beta)
            loss=ce(lo,m)+CFG["dice_factor"]*dice_loss(lo,m)+alpha*sp.mean()
            loss.backward(); opt.step(); tl+=loss.item()
            if ep==0:
                with torch.no_grad(): sv=sp.cpu().numpy()
                for si,v in zip(b["src"].numpy(),sv): rows.append([arm,ep+1,SRC[int(si)],float(v)])
        if rows:
            with open(CFG["seam_csv"],"a",newline="") as f: csv.writer(f).writerows(rows)
            sdf=pd.DataFrame(rows,columns=["arm","epoch","source","seam_loss"])
            agg=sdf.groupby("source").seam_loss.agg(["count","mean","median"])
            print("  seam term by source, epoch 1:"); print(agg.round(6).to_string())
            if {"negative_FUR","negative_other"} <= set(agg.index):
                print(f"    ratio fur / ordinary negative: "
                      f"{agg.loc['negative_FUR','mean']/agg.loc['negative_other','mean']:.1f}x")

        vd,negf,furf,furm,noif,noim=validate(model,HW)
        med,mean_,sil,nofur = score_setc(model,HW,arm,ep+1)
        with open(CFG["log_csv"],"a",newline="") as f:
            csv.writer(f).writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),arm,alpha,beta,ep+1,
                                    tl/len(loader),vd,negf,furf,furm,noif,noim,med,mean_,sil,nofur])
        print(f"  e{ep+1}: dice {vd:.4f} | oldneg {negf:.3f} | FUR {furf:.3f} (maxp {furm:.4f}) "
              f"| NOISE {noif:.3f} (maxp {noim:.4f})")
        print(f"        setC median {med:.4f} (excl fur {nofur:.4f}) | mean {mean_:.4f} | "
              f"silent {sil:.1%}")
        torch.save(model.state_dict(), f"{out}/ep{ep+1}.pth")
    del model,loader,ds; torch.cuda.empty_cache()

print(f"\ndone in {(time.time()-t0)/3600:.1f} h   outputs under {ROOT}/")
d=pd.read_csv(CFG["log_csv"])
print("\n=== setC median max probability, all 280 ===")
print(d.pivot_table(index="epoch",columns="arm",values="setC_median").round(4).to_string())
print("\n=== setC median EXCLUDING fur, 240 images the arms never trained on ===")
print(d.pivot_table(index="epoch",columns="arm",values="setC_median_nofur").round(4).to_string())
print("\n=== held out fur rate, n=59 ===")
print(d.pivot_table(index="epoch",columns="arm",values="fur_rate").round(4).to_string())
print("\n  reported fur arm (alpha 0.75, beta 4): setC median 0.0001 at epoch 4")
print("\nREAD the excluding-fur column: it is the clean measure of global silencing, since")
print("30 of the benchmark's 40 fur images were supplied as negatives. If both arms collapse")
print("it toward zero, the seam term is not the mechanism and Section 7.1 must be withdrawn.")
print("If they hold it high while held out fur still falls, the term is confirmed.")