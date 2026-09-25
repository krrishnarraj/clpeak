#!/usr/bin/env python3
"""Frame the raw app captures for the README and the docs site.

Each raw capture (app/integration_test/screenshots_test.dart, 2x) becomes a
window: a slim title bar, rounded corners, a hairline border and a soft
shadow on a transparent ground, scaled to OUT_WIDTH.  The window colours are
the app's own theme tokens (app/lib/src/theme/clpeak_theme.dart), so a frame
never reads as a different product from the pixels inside it.

Written as 8-bit palette PNGs (palettize), which is what keeps them in the
tens of KB: truecolour + alpha is 4-6x the size and looks the same.

Also writes og.png: the hero on an opaque ground at the 1200x630 a link
preview wants, since a transparent image previews as a black or white box.

Usage: frame.py RAW_DIR OUT_DIR   (needs Pillow)
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# The README shows the hero ~850-920 css px wide: 1280 px is ~1.4x on a
# retina screen, which reads the same as 1600 there, while 1000 is visibly
# soft.
OUT_WIDTH = 1280
HERO = "results-npu"      # the README's screenshot, and the link preview

# Palette: the window's colours, then black at SHADOW_LEVELS alpha steps of
# SHADOW_STEP for the shadow (its alpha never reaches 128).  64 window colours
# keep the text anti-aliasing and every theme tone distinct; 4-step alpha
# is fine enough that the shadow does not band.
WINDOW_COLORS = 64
SHADOW_LEVELS = 32
SHADOW_STEP = 4

# Per theme: page ground (title bar), hairline, and the title-bar dots.
THEMES = {
    "dark": {"bar": (11, 12, 14), "line": (48, 52, 58), "dot": (64, 68, 74)},
    "light": {"bar": (246, 246, 247), "line": (214, 217, 222), "dot": (200, 203, 208)},
}

# Geometry at capture scale (2x).
BAR = 56        # title bar height
RADIUS = 22     # window corner radius
PAD = 72        # transparent margin the shadow falls into
SHADOW_BLUR = 28
SHADOW_DY = 18
SHADOW_ALPHA = 70


def rounded_mask(size, radius):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255)
    return mask


def window(shot: Image.Image, theme: str) -> Image.Image:
    """The capture under a title bar, rounded and outlined."""
    c = THEMES[theme]
    w, h = shot.width, shot.height + BAR
    win = Image.new("RGBA", (w, h), c["bar"] + (255,))
    win.paste(shot, (0, BAR))
    d = ImageDraw.Draw(win)
    d.line((0, BAR - 1, w, BAR - 1), fill=c["line"] + (255,), width=2)
    for i in range(3):  # neutral dots: no platform's window controls
        cx, cy, r = 34 + i * 30, BAR // 2, 9
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=c["dot"] + (255,))

    mask = rounded_mask(win.size, RADIUS)
    out = Image.new("RGBA", win.size, (0, 0, 0, 0))
    out.paste(win, (0, 0), mask)
    ImageDraw.Draw(out).rounded_rectangle(
        (0, 0, w - 1, h - 1), radius=RADIUS, outline=c["line"] + (255,), width=2)
    return out


def with_shadow(win: Image.Image) -> Image.Image:
    size = (win.width + 2 * PAD, win.height + 2 * PAD)
    shadow = Image.new("L", size, 0)
    ImageDraw.Draw(shadow).rounded_rectangle(
        (PAD, PAD + SHADOW_DY, PAD + win.width, PAD + SHADOW_DY + win.height),
        radius=RADIUS, fill=SHADOW_ALPHA)
    shadow = shadow.filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    out.putalpha(shadow)
    out.alpha_composite(win, (PAD, PAD))
    return out


def scaled(img: Image.Image, width: int) -> Image.Image:
    if img.width <= width:
        return img
    return img.resize((width, round(img.height * width / img.width)),
                      Image.LANCZOS)


def palettize(img: Image.Image):
    """RGBA -> (8-bit palette image, its tRNS bytes).

    A pixel at least half opaque belongs to the window and gets one of
    WINDOW_COLORS median-cut colours, drawn opaque.  Anything fainter is
    shadow: black, at the nearest of the SHADOW_LEVELS alpha steps.  (PIL's
    own RGBA quantizer, octree, bands the shadow and merges near-white
    tones.)
    """
    alpha = img.getchannel("A")
    window = alpha.point(lambda a: 255 if a >= 128 else 0)
    # Shadow pixels borrow a window colour so they cost no palette entries.
    fill = img.getpixel((img.width // 2, img.height // 2))[:3]
    rgb = Image.new("RGB", img.size, fill)
    rgb.paste(img.convert("RGB"), (0, 0), window)
    out = rgb.quantize(colors=WINDOW_COLORS, method=Image.Quantize.MEDIANCUT,
                       dither=Image.Dither.NONE)
    shadow = alpha.point(
        lambda a: WINDOW_COLORS + min(round(a / SHADOW_STEP), SHADOW_LEVELS - 1))
    out.paste(shadow, (0, 0), window.point(lambda a: 255 - a))
    out.putpalette(out.getpalette()[:WINDOW_COLORS * 3] + [0, 0, 0] * SHADOW_LEVELS)
    trns = bytes([255] * WINDOW_COLORS +
                 [k * SHADOW_STEP for k in range(SHADOW_LEVELS)])
    return out, trns


def social(framed: Image.Image, ground) -> Image.Image:
    """The framed hero, top-anchored and cropped, on an opaque 1200x630."""
    W, H = 1200, 630
    out = Image.new("RGBA", (W, H), ground + (255,))
    win = scaled(framed, W)
    out.alpha_composite(win, (0, 24))
    return out.convert("RGB").quantize(
        colors=WINDOW_COLORS, method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE)


def main():
    raw, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    shots = sorted(raw.glob("*-dark.png")) + sorted(raw.glob("*-light.png"))
    if not shots:
        sys.exit(f"no captures in {raw}")
    for src in shots:
        theme = src.stem.rsplit("-", 1)[1]
        framed = with_shadow(window(Image.open(src).convert("RGBA"), theme))
        img, trns = palettize(scaled(framed, OUT_WIDTH))
        img.save(dst / src.name, optimize=True, transparency=trns)
        print(f"{dst / src.name}")
        if src.stem == f"{HERO}-dark":
            social(framed, (22, 24, 28)).save(dst / "og.png", optimize=True)
            print(f"{dst / 'og.png'}")


if __name__ == "__main__":
    main()
