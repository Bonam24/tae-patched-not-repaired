# The SS5 controls: blank grey, flat patch, white field, luminance gradient
# (all at most 0.011). Run after noise_sweep_setup.py (same session).

controls = {
    "blank grey":        np.full((512,512,3), 128, np.uint8),
    "flat patch":        None,   # grey field, uniform 128 square — same layout, zero contrast
    "white field":       np.full((512,512,3), 230, np.uint8),
    "vertical gradient": np.tile(np.linspace(60,200,512,dtype=np.uint8)[:,None,None],(1,512,3)),
}
img = np.full((512,512,3), 128, np.uint8); img[176:336,176:336] = 128
controls["flat patch"] = img

for name in CKPTS:
    m = get_model(name)
    for cname, im in controls.items():
        p = prob_map(m, im)
        print(f"{name:<26} {cname:<18} max {p.max():.3f}  mean {p.mean():.3f}")