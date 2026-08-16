#!/usr/bin/env python3
"""Regenerate every platform app icon from the original clpeak wordmark art.

The source (``clpeak_master_1024.png``) is the icon from the pre-Flutter native
app: a white "clpeak" wordmark on a navy rounded rectangle with a drop shadow.
That rounded rectangle is a relic of pre-adaptive Android launchers -- modern
platforms supply the container shape themselves -- so this script recovers just
the wordmark as an alpha mask and re-composes it per platform:

  iOS      full-bleed opaque square (+ dark and tinted iOS 18 variants)
  macOS    Big Sur squircle inset in the canvas, with a soft shadow
  Android  adaptive foreground/monochrome inside the 66/108dp safe circle,
           plus legacy mipmap PNGs and a 512px Play Store icon
  Windows  multi-resolution .ico
  Linux    single 256px PNG, loaded as the GTK window icon at runtime

Usage:  python3 tool/icons/generate_icons.py      (needs Pillow)
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
APP = HERE.parent.parent / "app"
SOURCE = HERE / "clpeak_master_1024.png"

# Brand colours, sampled from the original art.
NAVY = (58, 79, 122)
NAVY_DARK = (32, 43, 67)  # iOS 18 dark-appearance backdrop
WHITE = (255, 255, 255)

# Apple's Big Sur icon grid: an 824x824 squircle with a 185.4pt corner radius,
# centred in a 1024x1024 canvas.
MAC_TILE = 824 / 1024
MAC_RADIUS = 185.4 / 1024

# Wordmark width as a fraction of the tile it sits on. 0.78 keeps the mark clear
# of iOS's superellipse mask and of Android's legacy corner rounding.
TEXT_ON_TILE = 0.78

# Android adaptive icons must keep content inside a 66dp circle on a 108dp
# canvas. The widest 2.98:1 rectangle that fits is 62.6dp; 0.52 backs off from
# that so the mark does not graze the mask on circular launchers.
TEXT_ON_ADAPTIVE = 0.52

SUPERSAMPLE = 4


def wordmark_mask() -> Image.Image:
    """Recover the wordmark as an 'L' alpha mask, tightly cropped.

    The source has white text composited over navy, so the coverage of each
    pixel is how far it travelled from navy towards white.
    """
    src = Image.open(SOURCE).convert("RGB")
    # Comfortably inside the navy rectangle (which spans 42,170 - 981,851).
    text = src.crop((100, 350, 920, 670))
    span = [w - n for w, n in zip((253, 253, 253), NAVY)]

    mask = Image.new("L", text.size)
    src_px, dst_px = text.load(), mask.load()
    for y in range(text.height):
        for x in range(text.width):
            r, g, b = src_px[x, y]
            cover = sum(
                (c - n) / s for c, n, s in zip((r, g, b), NAVY, span)
            ) / 3.0
            dst_px[x, y] = max(0, min(255, round(cover * 255)))

    return mask.crop(mask.getbbox())


WORDMARK = None  # populated in main()


def scaled_wordmark(width: int) -> Image.Image:
    """Wordmark mask resampled to ``width``, preserving aspect ratio."""
    height = max(1, round(WORDMARK.height * width / WORDMARK.width))
    mark = WORDMARK.resize((width, height), Image.LANCZOS)
    if width < 60:
        # Below ~60px the letter stems fall under a pixel and fade to a smear.
        # A gamma boost keeps them present rather than washing out entirely.
        mark = mark.point(lambda a: round(255 * (a / 255) ** 0.7))
    return mark


def tile_mask(size: int, side: float, radius: float) -> Image.Image:
    """Anti-aliased mask for a rounded square of ``side`` px, centred."""
    ss = SUPERSAMPLE
    big = Image.new("L", (size * ss, size * ss), 0)
    draw = ImageDraw.Draw(big)
    inset = (size - side) / 2 * ss
    draw.rounded_rectangle(
        (inset, inset, size * ss - inset - 1, size * ss - inset - 1),
        radius=radius * ss,
        fill=255,
    )
    return big.resize((size, size), Image.LANCZOS)


def render(
    size: int,
    *,
    background=NAVY,
    foreground=WHITE,
    tile: float = 1.0,
    radius: float = 0.0,
    text_frac: float = TEXT_ON_TILE,
    shadow: bool = False,
    opaque: bool = False,
) -> Image.Image:
    """Compose one icon.

    ``tile`` and ``radius`` are fractions of ``size``; ``text_frac`` is a
    fraction of the tile. ``background=None`` leaves the tile transparent,
    which is what the Android foreground and iOS tinted variants want.
    """
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    side = size * tile
    shape = tile_mask(size, side, size * radius) if tile < 1.0 or radius else None

    if shadow and shape is not None:
        blur = max(1.0, size * 0.018)
        drop = shape.filter(ImageFilter.GaussianBlur(blur))
        drop = drop.point(lambda a: round(a * 0.28))
        offset = Image.new("L", (size, size), 0)
        offset.paste(drop, (0, round(size * 0.014)))
        canvas.paste(Image.new("RGBA", (size, size), (0, 0, 0, 255)), (0, 0), offset)

    if background is not None:
        fill = Image.new("RGBA", (size, size), background + (255,))
        canvas.paste(fill, (0, 0), shape) if shape else canvas.alpha_composite(fill)

    mark = scaled_wordmark(max(1, round(side * text_frac)))
    tinted = Image.new("RGBA", mark.size, foreground + (255,))
    canvas.paste(
        tinted,
        ((size - mark.width) // 2, (size - mark.height) // 2),
        mark,
    )

    if opaque:
        flat = Image.new("RGB", (size, size), background or NAVY)
        flat.paste(canvas.convert("RGB"), (0, 0), canvas.split()[3])
        return flat
    return canvas


def write(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    print(f"  {path.relative_to(APP)}  {image.width}x{image.height}")


# --------------------------------------------------------------------------- iOS


def build_ios() -> None:
    """Single 1024 universal icon plus dark and tinted appearances.

    Xcode renders every derived size from these, and the App Store rejects an
    icon with an alpha channel, so all three are flattened.
    """
    out = APP / "ios/Runner/Assets.xcassets/AppIcon.appiconset"
    for stale in out.glob("*.png"):
        stale.unlink()

    write(render(1024, opaque=True), out / "Icon-App-1024x1024@1x.png")
    write(
        render(1024, background=NAVY_DARK, opaque=True),
        out / "Icon-App-1024x1024@1x-dark.png",
    )
    # Tinted: greyscale art on a transparent field; iOS supplies the backdrop
    # and maps luminance onto the user's chosen tint.
    write(render(1024, background=None), out / "Icon-App-1024x1024@1x-tinted.png")

    contents = {
        "images": [
            {
                "filename": "Icon-App-1024x1024@1x.png",
                "idiom": "universal",
                "platform": "ios",
                "size": "1024x1024",
            },
            {
                "appearances": [{"appearance": "luminosity", "value": "dark"}],
                "filename": "Icon-App-1024x1024@1x-dark.png",
                "idiom": "universal",
                "platform": "ios",
                "size": "1024x1024",
            },
            {
                "appearances": [{"appearance": "luminosity", "value": "tinted"}],
                "filename": "Icon-App-1024x1024@1x-tinted.png",
                "idiom": "universal",
                "platform": "ios",
                "size": "1024x1024",
            },
        ],
        "info": {"author": "xcode", "version": 1},
    }
    (out / "Contents.json").write_text(json.dumps(contents, indent=2) + "\n")
    print(f"  {(out / 'Contents.json').relative_to(APP)}")


# ------------------------------------------------------------------------- macOS


def build_macos() -> None:
    out = APP / "macos/Runner/Assets.xcassets/AppIcon.appiconset"
    for size in (16, 32, 64, 128, 256, 512, 1024):
        write(
            render(
                size,
                tile=MAC_TILE,
                radius=MAC_RADIUS,
                shadow=size >= 64,
            ),
            out / f"app_icon_{size}.png",
        )


# ----------------------------------------------------------------------- Android

ANDROID_DENSITIES = {
    "mdpi": 1.0,
    "hdpi": 1.5,
    "xhdpi": 2.0,
    "xxhdpi": 3.0,
    "xxxhdpi": 4.0,
}

ADAPTIVE_XML = """<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@color/ic_launcher_background" />
    <foreground android:drawable="@mipmap/ic_launcher_foreground" />
    <monochrome android:drawable="@mipmap/ic_launcher_monochrome" />
</adaptive-icon>
"""

BACKGROUND_XML = """<?xml version="1.0" encoding="utf-8"?>
<resources>
    <color name="ic_launcher_background">#3A4F7A</color>
</resources>
"""


def build_android() -> None:
    res = APP / "android/app/src/main/res"

    for density, scale in ANDROID_DENSITIES.items():
        mipmap = res / f"mipmap-{density}"

        # Adaptive layers: 108dp canvas, content inside the 66dp safe circle.
        adaptive = round(108 * scale)
        write(
            render(adaptive, background=None, text_frac=TEXT_ON_ADAPTIVE),
            mipmap / "ic_launcher_foreground.png",
        )
        # Themed icons are tinted by the launcher; only the alpha matters.
        write(
            render(adaptive, background=None, text_frac=TEXT_ON_ADAPTIVE),
            mipmap / "ic_launcher_monochrome.png",
        )

        # Legacy launcher icon for API < 26, which gets no mask of its own.
        legacy = round(48 * scale)
        write(
            render(legacy, tile=0.92, radius=0.92 * 0.20),
            mipmap / "ic_launcher.png",
        )

    (res / "mipmap-anydpi-v26").mkdir(parents=True, exist_ok=True)
    (res / "mipmap-anydpi-v26/ic_launcher.xml").write_text(ADAPTIVE_XML)
    (res / "values/ic_launcher_background.xml").write_text(BACKGROUND_XML)
    print("  android/app/src/main/res/mipmap-anydpi-v26/ic_launcher.xml")
    print("  android/app/src/main/res/values/ic_launcher_background.xml")

    # Play Store listing icon: full-bleed 512x512, Play applies its own mask.
    write(render(512), APP / "android/app/src/main/ic_launcher-web.png")


# ----------------------------------------------------------------------- Windows


def build_windows() -> None:
    sizes = (16, 24, 32, 48, 64, 96, 128, 256)
    frames = [render(s, tile=0.94, radius=0.94 * 0.18) for s in sizes]
    out = APP / "windows/runner/resources/app_icon.ico"
    frames[-1].save(out, format="ICO", sizes=[(s, s) for s in sizes],
                    append_images=frames[:-1])
    print(f"  {out.relative_to(APP)}  {', '.join(str(s) for s in sizes)}")


# ------------------------------------------------------------------------- Linux


def build_linux() -> None:
    # GTK has no resource-embedding step like Windows' .rc, so the runner loads
    # this PNG from the bundle at startup (linux/runner/my_application.cc).  One
    # 256px master is enough -- GTK downscales for the titlebar and taskbar.
    out = APP / "linux/runner/resources/clpeak_icon.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    write(render(256, tile=0.94, radius=0.94 * 0.18), out)


def main() -> None:
    global WORDMARK
    print("extracting wordmark from", SOURCE.name)
    WORDMARK = wordmark_mask()
    print(f"  mask {WORDMARK.width}x{WORDMARK.height}")

    for name, build in (
        ("iOS", build_ios),
        ("macOS", build_macos),
        ("Android", build_android),
        ("Windows", build_windows),
        ("Linux", build_linux),
    ):
        print(name)
        build()


if __name__ == "__main__":
    main()
