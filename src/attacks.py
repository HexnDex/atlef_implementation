"""
ATLEF: Attack Simulation Functions

Implements the 18 channel-distortion conditions applied during
robustness evaluation: lossless passthrough, PNG re-encoding, JPEG
compression at three quality levels, resize at four scale factors,
additive Gaussian noise at three levels, Gaussian blur at three radii,
and median blur at three kernel sizes. Matches Table 3.4 (ATLEF attack
suite) and Appendix A.4 of the thesis.
"""

import io
from copy import deepcopy

import numpy as np
from PIL import Image, ImageFilter


def _safe(img):
    """Defensive copy so an attack function never mutates the caller's image."""
    return img.copy() if hasattr(img, "copy") else deepcopy(img)


# ------------------------------------------------------------
# Attack functions
# ------------------------------------------------------------


def atk_identity(img):
    """Lossless / clean channel, no distortion."""
    return _safe(img)


def atk_png(img):
    """PNG re-encode. Lossless compression, but still a save/reload
    round trip through a real image codec."""
    img = _safe(img)
    b = io.BytesIO()
    img.save(b, "PNG")
    b.seek(0)
    return Image.open(b).convert("RGB")


def atk_jpeg(img, quality=90):
    """JPEG compression at the given quality factor."""
    img = _safe(img)
    b = io.BytesIO()
    img.save(b, "JPEG", quality=quality)
    b.seek(0)
    return Image.open(b).convert("RGB")


def atk_resize(img, scale=1.0):
    """Downscale or upscale by `scale`, then resize back to the original
    dimensions (bicubic), simulating a resize round trip."""
    img = _safe(img)
    w, h = img.size
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    return img.resize((w, h), Image.BICUBIC)


def atk_gnoise(img, sigma=0.02):
    """Additive white Gaussian noise at standard deviation `sigma`
    (image normalized to [0, 1] before noise is added)."""
    img = _safe(img)
    arr = np.array(img).astype("float32") / 255.0
    noise = np.random.normal(0, sigma, arr.shape).astype("float32")
    arr = np.clip(arr + noise, 0.0, 1.0)
    arr = (arr * 255).astype("uint8")
    return Image.fromarray(arr)


def atk_gblur(img, r=1.0):
    """Gaussian blur at PIL radius `r`."""
    return _safe(img).filter(ImageFilter.GaussianBlur(radius=r))


def atk_mblur(img, kernel=3):
    """Median blur at the given (odd) kernel size."""
    k = int(kernel)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return _safe(img).filter(ImageFilter.MedianFilter(size=k))


# ------------------------------------------------------------
# Attack grid: the 18 named conditions used throughout evaluation
# ------------------------------------------------------------

# Kernel-to-radius mapping for Gaussian blur uses the common approximation
# 3x3 -> r=1.0, 5x5 -> r=2.0, 7x7 -> r=3.0, so results are labeled and
# compared by kernel size even though PIL's GaussianBlur takes a radius.
ATTACKS = {
    "clean": (atk_identity, {}),
    "png": (atk_png, {}),

    "jpeg90": (atk_jpeg, {"quality": 90}),
    "jpeg70": (atk_jpeg, {"quality": 70}),
    "jpeg50": (atk_jpeg, {"quality": 50}),

    "resize0.5": (atk_resize, {"scale": 0.5}),
    "resize0.75": (atk_resize, {"scale": 0.75}),
    "resize1.25": (atk_resize, {"scale": 1.25}),
    "resize1.5": (atk_resize, {"scale": 1.5}),

    "gnoise01": (atk_gnoise, {"sigma": 0.01}),
    "gnoise05": (atk_gnoise, {"sigma": 0.05}),
    "gnoise10": (atk_gnoise, {"sigma": 0.10}),

    "gblur3": (atk_gblur, {"r": 1.0}),
    "gblur5": (atk_gblur, {"r": 2.0}),
    "gblur7": (atk_gblur, {"r": 3.0}),

    "mblur3": (atk_mblur, {"kernel": 3}),
    "mblur5": (atk_mblur, {"kernel": 5}),
    "mblur7": (atk_mblur, {"kernel": 7}),
}


if __name__ == "__main__":
    print(f"{len(ATTACKS)} attack conditions registered:")
    for name in ATTACKS:
        print(f"  - {name}")
