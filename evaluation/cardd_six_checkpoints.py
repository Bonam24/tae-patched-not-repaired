# CarDD zero-shot re-evaluation, six checkpoints through one code path
# (Appendix B, Table 5): Dice/IoU/precision/recall pooled AND per image,
# fur arm vs matched control at epoch 4.

# =========================================================
# CarDD RE-EVALUATION, SIX CHECKPOINTS, ONE CODE PATH
# Reports Dice/IoU/precision/recall BOTH pooled and per image, so the
# reconciliation shows which definition the earlier runs used, and
# compares the fur arm against its matched control at epoch 4.
# =========================================================
!pip install -q transformers==5.13.0 pycocotools

import os, glob
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
from pycocotools.coco import COCO
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm

CARDD  = glob.glob("/kaggle/input/**/cardd", recursive=True)[0]
MINE   = "/kaggle/input/datasets/bonamosen"
THRS   = [0.50, 0.85]
MAX_IM = None        # 200 for a smoke test first, then None for all 4000

WANT = [("baseline (vanilla)",         "vanilla"),
        ("refined, foliage fix only",  "smashed_ep5"),
        ("+ foliage recipe (ep13)",    "corrected_ep13"),
        ("+ scratch negatives (ep8)",  "scratchfix_ep8"),
        ("+ fur negatives (ep4)",      "furfix_ep4"),
        ("control, no fur negs (ep4)", "control_ep4")]

found = {os.path.basename(p).replace(".pth", ""): p
         for p in sorted(glob.glob(f"{MINE}/**/*.pth", recursive=True))}
CKPTS, LABEL = {}, {}
for label, key in WANT:
    hit = [k for k in found if key in k]
    if not hit:
        print(f"NOT FOUND: {key}"); continue
    CKPTS[hit[0]] = found[hit[0]]; LABEL[hit[0]] = label
print(f"{len(CKPTS)} checkpoints:")
for k in CKPTS: print(f"  {LABEL[k]:<30} {os.path.basename(CKPTS[k])}")

BACKBONE, SIZE = "nvidia/mit-b3", (512, 512)
MEAN = np.array([0.485,0.456,0.406], np.float32); STD = np.array([0.229,0.224,0.225], np.float32)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load(name):
    m = SegformerForSemanticSegmentation.from_pretrained(
        BACKBONE, num_labels=2, ignore_mismatched_sizes=True)
    sd = torch.load(CKPTS[name], map_location=DEV, weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    r  = m.load_state_dict(sd, strict=False)
    cls = [x for x in r.missing_keys if "classifier" in x or "decode_head" in x]
    bb  = [x for x in r.missing_keys if x not in cls]
    assert not cls, f"HEAD DID NOT LOAD for {name}: {cls[:5]}"
    assert len(bb) <= 5, f"{len(bb)} backbone keys missing for {name}: {bb[:5]}"
    return m.to(DEV).eval()

@torch.no_grad()
def prob_map(model, rgb):
    h, w = rgb.shape[:2]
    x = cv2.resize(rgb, SIZE[::-1]).astype(np.float32)/255.0
    x = torch.from_numpy(((x-MEAN)/STD).transpose(2,0,1)).unsqueeze(0).float().to(DEV)
    lo = F.interpolate(model(x).logits, size=SIZE, mode="bilinear", align_corners=False)
    return cv2.resize(torch.softmax(lo,1)[0,1].float().cpu().numpy(), (w,h))

# ---------- image list, built once ----------
records = []
for split in ["train", "val", "test"]:
    js   = glob.glob(f"{CARDD}/**/{split}.json", recursive=True)
    idir = [d for d in glob.glob(f"{CARDD}/**/{split}", recursive=True) if os.path.isdir(d)]
    if not js or not idir:
        print(f"  {split}: not found"); continue
    coco = COCO(js[0])
    cats = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    gids = [i for i, n in cats.items() if "glass" in n.lower()]
    assert gids, cats
    for iid in coco.getImgIds():
        info = coco.loadImgs(iid)[0]
        anns = [a for a in coco.loadAnns(coco.getAnnIds(imgIds=iid)) if a["category_id"] in gids]
        fp = os.path.join(idir[0], info["file_name"])
        if not os.path.exists(fp):
            hit = glob.glob(f"{CARDD}/**/{info['file_name']}", recursive=True)
            if not hit: continue
            fp = hit[0]
        records.append(dict(path=fp, is_glass=len(anns) > 0, anns=anns,
                            h=info["height"], w=info["width"], coco=coco))
if MAX_IM: records = records[:MAX_IM]
print(f"\n{len(records)} images | glass {sum(r['is_glass'] for r in records)} "
      f"| non glass {sum(not r['is_glass'] for r in records)}\n")

def gt_mask(rec):
    m = np.zeros((rec["h"], rec["w"]), np.uint8)
    for a in rec["anns"]: m |= rec["coco"].annToMask(a).astype(np.uint8)
    return m

# ---------- evaluate ----------
out = []
for name in CKPTS:
    model = load(name)
    acc = {t: dict(tp=0, fp=0, fn=0, det=0, ng=0, fired=0, nn=0,
                   pd=[], pi=[], pp=[], pr=[]) for t in THRS}
    for rec in tqdm(records, desc=LABEL[name]):
        im = cv2.imread(rec["path"])
        if im is None: continue
        pm = prob_map(model, cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        gt = gt_mask(rec) if rec["is_glass"] else None
        if gt is not None and gt.shape != pm.shape:
            gt = cv2.resize(gt, (pm.shape[1], pm.shape[0]), interpolation=cv2.INTER_NEAREST)
        for t in THRS:
            pred = (pm >= t).astype(np.uint8); a = acc[t]
            if rec["is_glass"]:
                a["ng"] += 1
                tp = int((pred & gt).sum()); fp = int((pred & (1-gt)).sum()); fn = int(((1-pred) & gt).sum())
                a["tp"] += tp; a["fp"] += fp; a["fn"] += fn
                if tp > 0: a["det"] += 1
                a["pd"].append((2*tp)/max(2*tp+fp+fn, 1))
                a["pi"].append(tp/max(tp+fp+fn, 1))
                a["pp"].append(tp/max(tp+fp, 1))
                a["pr"].append(tp/max(tp+fn, 1))
            else:
                a["nn"] += 1
                if pm.max() >= t: a["fired"] += 1
    for t in THRS:
        a = acc[t]; tp, fp, fn = a["tp"], a["fp"], a["fn"]
        out.append(dict(model=LABEL[name], ckpt=name, thr=t,
            dice_pooled=(2*tp)/max(2*tp+fp+fn,1), iou_pooled=tp/max(tp+fp+fn,1),
            prec_pooled=tp/max(tp+fp,1),          rec_pooled=tp/max(tp+fn,1),
            dice_perimg=float(np.mean(a["pd"])) if a["pd"] else 0,
            iou_perimg=float(np.mean(a["pi"]))  if a["pi"] else 0,
            prec_perimg=float(np.mean(a["pp"])) if a["pp"] else 0,
            rec_perimg=float(np.mean(a["pr"]))  if a["pr"] else 0,
            detected=a["det"]/max(a["ng"],1), n_glass=a["ng"],
            nonglass_fire=a["fired"]/max(a["nn"],1), n_nonglass=a["nn"]))
    del model; torch.cuda.empty_cache()

res = pd.DataFrame(out)
res.to_csv("/kaggle/working/cardd_six_checkpoints.csv", index=False)

for t in THRS:
    print(f"\n===== tau = {t} : POOLED =====")
    print(res[res.thr==t][["model","dice_pooled","iou_pooled","prec_pooled","rec_pooled",
                           "detected","nonglass_fire"]].round(4).to_string(index=False))
    print(f"----- tau = {t} : PER IMAGE -----")
    print(res[res.thr==t][["model","dice_perimg","iou_perimg","prec_perimg","rec_perimg"]]
          .round(4).to_string(index=False))

# ---------- which definition reproduces the paper? ----------
print("\n=== reconciliation against the figures currently in the paper ===")
paper = {("refined, foliage fix only", 0.50): 0.884,
         ("+ foliage recipe (ep13)",   0.85): 0.869,
         ("+ scratch negatives (ep8)", 0.85): 0.842}
for (mdl, t), d in paper.items():
    row = res[(res.model == mdl) & (res.thr == t)]
    if row.empty: continue
    r = row.iloc[0]
    which = "POOLED" if abs(r.dice_pooled-d) < abs(r.dice_perimg-d) else "PER IMAGE"
    print(f"  {mdl} @ tau {t}: paper {d:.3f} | pooled {r.dice_pooled:.3f} "
          f"| per image {r.dice_perimg:.3f}  -> {which}")

# ---------- THE QUESTION: is the fur stage's CarDD regression the fur, or the epochs? ----------
print("\n=== fur stage vs its matched control, both at epoch 4 ===")
for t in THRS:
    g = res[res.thr == t].set_index("model")
    need = ["+ scratch negatives (ep8)", "+ fur negatives (ep4)", "control, no fur negs (ep4)"]
    if not all(m in g.index for m in need):
        print(f"  tau {t}: a required row is missing, cannot compare"); continue
    start, fur, ctl = (g.loc[m] for m in need)
    print(f"\n  tau = {t}")
    print(f"    {'':<26}{'non glass':>11}{'Dice(img)':>12}{'recall':>10}")
    for lab, r in [("ep8 start", start), ("fur arm, epoch 4", fur), ("control arm, epoch 4", ctl)]:
        print(f"    {lab:<26}{r.nonglass_fire:>11.4f}{r.dice_perimg:>12.4f}{r.rec_perimg:>10.4f}")
    d_fur = fur.nonglass_fire - start.nonglass_fire
    d_ctl = ctl.nonglass_fire - start.nonglass_fire
    print(f"    change from ep8:  fur arm {d_fur:+.4f}   control arm {d_ctl:+.4f}")
    if abs(d_fur - d_ctl) < 0.02:
        print("    -> the arms move together: the CarDD regression is CONTINUED TRAINING, not the")
        print("       fur negatives. Report it as such, as with the prior negatives.")
    elif d_fur > d_ctl:
        print("    -> the fur arm regresses MORE than the control: a genuine amplification effect")
        print("       attributable to the fur negatives. Real finding, report it plainly.")
    else:
        print("    -> the fur arm regresses LESS than the control. Unexpected; inspect before claiming.")