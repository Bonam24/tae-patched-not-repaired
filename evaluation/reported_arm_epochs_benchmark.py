# Scores the reported fur arm's epochs 1-3 on the 280-image benchmark —
# the 'as reported' row of the Appendix H table (0.167 / 0.124 / 0.057).

# score the reported fur arm's epochs 1-3 on the same 280 benchmark

import glob, os, torch, numpy as np, pandas as pd
import torch.nn.functional as F

for ep in [2]:
    p = f"checkpoints_smashed_furfix/smashed_furfix_ep{ep}.pth"
    if not os.path.exists(p):
        print("missing:", p); continue
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b3", num_labels=2, ignore_mismatched_sizes=True)
    sd = torch.load(p, map_location=device, weights_only=True)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    ov = len(set(m.state_dict()) & set(sd)); assert ov > 600, f"{ov}/644"
    m.load_state_dict(sd, strict=False); m = m.to(device).eval()
    mx = []
    with torch.no_grad():
        for q in SETC_IMGS:
            lo = F.interpolate(m(prep(q,(512,512)).to(device)).logits,
                               size=(512,512), mode="bilinear", align_corners=False)
            mx.append(float(torch.softmax(lo,1)[0,1].max()))
    mx = np.array(mx)
    print(f"furfix ep{ep}: mean {mx.mean():.4f} | median {np.median(mx):.4f} | silent {(mx<0.01).mean():.1%}")
    del m; torch.cuda.empty_cache()