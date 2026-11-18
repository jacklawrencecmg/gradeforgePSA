import numpy as np, cv2
from .synth import make_card, lit, add_stain, rounded_mask
from gradeforge.cv.lighting import build_stack
from gradeforge.cv.context import make_ctx
from gradeforge.cv import surface

PPM = 1000 / 63.5


def scene(per_light, card=None, noise=1.0, seed=0):
    base, mask = make_card(lr=(80, 80), tb=(80, 80), border=(30, 30, 30))
    base = card if card is not None else base
    rng = np.random.default_rng(seed)
    imgs = []
    for kw in per_light:
        im = lit(base, **kw).astype(np.float32) + rng.normal(0, noise, base.shape)
        imgs.append(np.clip(im, 0, 255).astype(np.uint8))
    lights = [{"ok": True, "role": f"light{i}", "aligned": im, "valid": np.full(mask.shape, 255, np.uint8),
               "residual_shift_px": 0.2, "cc": 0.99} for i, im in enumerate(imgs[1:], 1)]
    stack = build_stack(imgs[0], mask, lights, PPM)
    ctx = make_ctx(imgs[0], mask, (30, 128, 128), PPM, PPM, True)
    return ctx, stack, imgs, mask


def run(ctx, stack, side="front"):
    ring = surface.build_ring(ctx.mask, {"left_px": 80, "right_px": 80, "top_px": 80, "bottom_px": 80}, ctx.ppm)
    resid = []
    for k in range(stack.count):
        lab = cv2.cvtColor(stack_img(stack, k, ctx), cv2.COLOR_BGR2LAB).astype(np.float32)
        resid.append(surface._ring_resid(lab, ring, ctx.ppm))
    f, masks = surface.detect_ring(ctx, ring, stack, side, resid)
    f += surface.detect_differential(ctx, stack, side, masks[0])
    return f


_IMGS = {}
def stack_img(stack, k, ctx):
    return _IMGS[id(stack)][k]


def go(per_light, **kw):
    ctx, stack, imgs, _ = scene(per_light, **kw)
    _IMGS[id(stack)] = imgs
    return run(ctx, stack), stack


def test_clean_two_lights_no_findings():
    f, _ = go([dict(ramp=(0.1, 0)), dict(ramp=(-0.1, 0))])
    assert [x for x in f if x["classification"].startswith("physical")] == []


def test_stain_in_border_is_albedo():
    base, mask = make_card(lr=(80, 80), tb=(80, 80), border=(30, 30, 30))
    base = add_stain(base, 40, 700, 25, 25, color=(60, 120, 170), strength=0.9)
    f, _ = go([dict(), dict(ramp=(0.05, 0))], card=base)
    st = [x for x in f if x["category"] in ("surface_stain", "surface_spot")]
    assert st and st[0]["classification"] == "physical_probable_albedo"


def test_stain_single_image_is_indeterminate():
    base, mask = make_card(lr=(80, 80), tb=(80, 80), border=(30, 30, 30))
    base = add_stain(base, 40, 700, 25, 25, color=(60, 120, 170), strength=0.9)
    f, _ = go([dict()], card=base)
    st = [x for x in f if x["category"] in ("surface_stain", "surface_spot")]
    assert st and all(x["classification"] == "indeterminate_single_image" for x in st)


def test_scratch_visible_only_by_differential():
    sc = ((300, 600), (700, 640), 40, 3)
    f, _ = go([dict(scratch=sc), dict(scratch=((300, 600), (700, 640), -25, 3)), dict()])
    rl = [x for x in f if x["category"] == "surface_relief_line"]
    assert rl and rl[0]["classification"] == "physical_probable_relief"


def test_dent_dipole():
    f, _ = go([dict(dent=(500, 700, 25, 60, 1)), dict(dent=(500, 700, 25, 60, -1))])
    assert any(x["category"] == "surface_relief_dipole" for x in f)


def test_sub_resolution_scratch_not_called_physical():
    ctx, stack, imgs, _ = scene([dict(scratch=((300, 600), (700, 640), 40, 1)), dict()])
    ctx.src_ppm = ctx.ppm * 0.25   # photo resolves only 1/4 of canvas detail
    _IMGS[id(stack)] = imgs
    f = run(ctx, stack)
    assert all(not x["classification"].startswith("physical") for x in f if x["category"] == "surface_relief_line")


def test_glare_findings():
    base, mask = make_card(lr=(80, 80), tb=(80, 80))
    im0 = base.copy(); cv2.circle(im0, (500, 700), 80, (255, 255, 255), -1)
    ctx, stack, imgs, _ = scene([dict(), dict()])
    g = surface.glare_findings(ctx, stack, [im0, base], "front")
    assert g and g[0]["classification"] == "artifact_glare"
