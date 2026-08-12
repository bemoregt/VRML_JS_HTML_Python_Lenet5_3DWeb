import json
import os
import random

from PIL import Image, ImageOps
from torchvision import datasets

# Exports a small, fixed pool of real MNIST *test*-set digits (not the
# training set - these are images the model never trained on, a fairer
# demo of real generalization) as PNGs the "재추론"(re-infer) button in the
# generated viewer can pick from at random on every click.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "mnist_raw")
OUT_DIR = os.path.join(ROOT_DIR, "output", "mnist-samples")
MANIFEST_PATH = os.path.join(ROOT_DIR, "data", "mnist_sample_manifest.json")

PER_DIGIT = 4  # -> 40 sample images total, every digit reachable

random.seed(7)
os.makedirs(OUT_DIR, exist_ok=True)

test_set = datasets.MNIST(DATA_DIR, train=False, download=True)

by_digit = {d: [] for d in range(10)}
for idx in range(len(test_set)):
    _, label = test_set[idx]
    if len(by_digit[label]) < PER_DIGIT:
        by_digit[label].append(idx)
    if all(len(v) >= PER_DIGIT for v in by_digit.values()):
        break

manifest = []
for digit in range(10):
    for n, idx in enumerate(by_digit[digit]):
        img, label = test_set[idx]
        # Real MNIST is white-digit-on-black (28x28) - upscaled 4x with
        # NEAREST (not smoothed) to stay crisp/blocky at display size, then
        # inverted to the dark-stroke-on-light-paper convention
        # lib/vrml.js's buildInputGrayscale expects (it un-inverts via
        # 1-luminance, matching a hand-drawn attachment's polarity).
        img128 = img.resize((128, 128), Image.NEAREST)
        img128 = ImageOps.invert(img128.convert("L")).convert("RGB")
        filename = "mnist-%d-%d.png" % (digit, n)
        img128.save(os.path.join(OUT_DIR, filename))
        manifest.append({"digit": digit, "url": "mnist-samples/" + filename})

with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("Exported %d sample images to %s" % (len(manifest), OUT_DIR))
print("Manifest written to", MANIFEST_PATH)
