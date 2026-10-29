# The seed 3 control at epoch 4, probed after training (the parenthetical
# in SS7): fur probe n=59 and noise probe n=15 under the training criterion.

"""
fill_table1_cell.py — the missing Table 1 cell: control, seed 3, epoch 4
========================================================================
Table 1 currently reads "0.559 / 0.712 / ---" at epoch 4 and the caption
says seed 3 "had completed three of its four epochs when this paper was
submitted". The training log shows that arm finished epoch 4 on 12 August
at 14:43 — only the PROBE EVALUATION was never run. So the caption as
written will be false at submission. This script runs that evaluation.

CRITERION: the training rule, not tau. Table 1 reports fired = predicted
area above 0.001 at threshold 0.5, which is what the training loop
monitored. Do NOT use tau = 0.85 here; the two rules are not comparable
and the paper says so explicitly.

PROBE: the same frozen n = 59 fur set (10 original holdout + 49 collected
after training) and the n = 15 noise probe. Reconstruct the fur split with
random.seed(0) then shuffle over the NON t-prefixed files only, or the
holdout will silently include supplied images.

Runtime: ~30 seconds for one checkpoint over 74 images.

WHAT TO DO WITH THE RESULT
  fires/59 in [0.39, 0.71]  -> in range with the other two control arms;
      fill the cell, and delete the caption's "had completed three of its
      four epochs..." sentence entirely.
  outside that range        -> report it as measured. A control arm that
      moves is still a control arm; do not quietly drop it.
  noise 15/15               -> expected, and it extends "1.000 in every
      epoch of all six runs" to genuinely all six.
"""

import os, glob, random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation

# ----------------------------- CONFIG ---------------------------------
CKPT_S3_CONTROL_EP4 = ""     # <-- the seed 3 control epoch 4 checkpoint
FUR_DIR   = "/kaggle/input/contextshortcutdata/contextShortcutData/fur"
FUREXTRA  = "/kaggle/input/furextra/furextra"     # the 49 collected after training
NOISE_DIR = ""               # 15 stimuli; leave "" to skip the noise probe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
THR, AREA = 0.5, 0.001        # the TRAINING criterion, as in Table 1
# ----------------------------------------------------------------------


def build_model(path):
    cfg = SegformerConfig(num_labels=2, depths=[3, 4, 18, 3],
                          hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768)
    m = SegformerForSemanticSegmentation(cfg)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    miss, unexp = m.load_state_dict(sd, strict=False)
    if len(miss) > 5 or len(unexp) > 5:
        raise RuntimeError(f"{len(miss)} missing / {len(unexp)} unexpected keys — check transformers==5.13.0")
    return m.to(DEVICE).eval()


@torch.no_grad()
def prob(model, img):
    w, h = img.size
    x = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    lg = model(pixel_values=x).logits
    lg = F.interpolate(lg, size=(h, w), mode="bilinear", align_corners=False)
    return torch.softmax(lg, 1)[0, 1].cpu().numpy()


def fur_probe_paths():
    """The frozen n=59: the 10 held out of the original 40, plus all 49 extra."""
    allf = sorted(glob.glob(os.path.join(FUR_DIR, "*")))
    orig = [f for f in allf if not os.path.basename(f).startswith("t")]
    assert len(orig) == 40, f"expected 40 non t-prefixed fur images, found {len(orig)}"
    random.seed(0)
    sh = orig[:]
    random.shuffle(sh)
    supplied, heldout = sh[:30], sh[30:]        # the 30 given as negatives
    extra = sorted(glob.glob(os.path.join(FUREXTRA, "*")))
    probe = heldout + extra
    assert not (set(probe) & set(supplied)), "CONTAMINATION: supplied image in the probe"
    print(f"fur probe: {len(heldout)} held out + {len(extra)} collected after = {len(probe)}")
    assert len(probe) == 59, f"probe is {len(probe)}, expected 59"
    return probe


def main():
    assert CKPT_S3_CONTROL_EP4, "set CKPT_S3_CONTROL_EP4 first"
    model = build_model(CKPT_S3_CONTROL_EP4)

    probe = fur_probe_paths()
    fired, mx = 0, []
    for p in probe:
        pm = prob(model, Image.open(p))
        f = float((pm > THR).mean()) > AREA
        fired += f
        mx.append(float(pm.max()))
    rate = fired / len(probe)
    lo = (rate + 1.96**2/(2*len(probe)) - 1.96*np.sqrt(rate*(1-rate)/len(probe) + 1.96**2/(4*len(probe)**2))) / (1 + 1.96**2/len(probe))
    hi = (rate + 1.96**2/(2*len(probe)) + 1.96*np.sqrt(rate*(1-rate)/len(probe) + 1.96**2/(4*len(probe)**2))) / (1 + 1.96**2/len(probe))
    print(f"\nCONTROL seed 3 epoch 4, held out fur: {fired}/{len(probe)} = {rate:.4f}  "
          f"Wilson 95% [{lo:.3f}, {hi:.3f}]")
    print(f"  mean max probability {np.mean(mx):.4f}  (treatment arms at epoch 4: 0.0001 / 0.0000 / 0.0001;"
          f" controls 0.567 / 0.699)")
    print(f"  other control arms at epoch 4 read 0.559 and 0.712 — "
          f"{'IN RANGE' if 0.39 <= rate <= 0.72 else 'OUTSIDE that range, report as measured'}")

    if NOISE_DIR:
        nps = sorted(glob.glob(os.path.join(NOISE_DIR, "*")))
        nf = sum(float((prob(model, Image.open(p)) > THR).mean()) > AREA for p in nps)
        print(f"\nnoise probe: {nf}/{len(nps)} (expected 15/15)")

    print(f"""
TEX EDIT once you have the number:
  Table 1 epoch 4 control cell:  $0.559 / 0.712 / ---$  ->  $0.559 / 0.712 / {rate:.3f}$
  Caption: delete the sentence beginning "Seed 3 of the control had completed
  three of its four epochs..." and, if the value sits inside [0.39, 0.71],
  leave the existing "No control value falls below 0.390" claim as is after
  checking it still holds.""")


if __name__ == "__main__":
    main()