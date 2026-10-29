# =========================================================
# NOISE PROBE GENERATOR — the 15 synthetic bandpass stimuli
# Deterministic: same recipe and seeds as used during training,
# identical to the regeneration block in noise_erasure_signature.py.
# 3 frequency bands x 5 seeds = 15 images, 320 px patch on a
# neutral grey 512x512 field.
# =========================================================
import os
import numpy as np, cv2

OUT = "noise_probe"
os.makedirs(OUT, exist_ok=True)

for lo, hi in [(0.01, 0.05), (0.05, 0.12), (0.12, 0.22)]:
    for s in range(5):
        rng = np.random.default_rng(1000*int(hi*100)+s); S = 320
        Fr = np.fft.fftshift(np.fft.fft2(rng.standard_normal((S, S))))
        u = np.fft.fftshift(np.fft.fftfreq(S)); r = np.hypot(*np.meshgrid(u, u, indexing='ij'))
        b = np.real(np.fft.ifft2(np.fft.ifftshift(Fr*((r >= lo) & (r <= hi)))))
        b = (b - b.mean())/(b.std() + 1e-8)
        patch = (np.clip(0.5 + 0.35*b, 0, 1)[..., None].repeat(3, 2)*255).astype(np.uint8)
        img = np.full((512, 512, 3), 128, np.uint8); img[96:96+S, 96:96+S] = patch
        cv2.imwrite(f"{OUT}/noise_{lo:.2f}_{s}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

n = len(os.listdir(OUT))
assert n == 15, n
print(f"wrote {n} stimuli to {OUT}/")
