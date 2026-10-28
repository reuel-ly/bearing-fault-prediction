"""Shared paths and signal settings. Paths are absolute, so scripts run from any directory."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DE_ROOT = ROOT / "data" / "12k_DE"
NORMAL_ROOT = ROOT / "data" / "Normal"
ART_DIR = ROOT / "artifacts"
MODELS_DIR = ROOT / "models"

FS, SIZE = 12_000, 2048
N_LOADS = 4
SEED = 42

# Every record is low-passed with the same steep filter (-80 dB by ~4.5 kHz). Normal records
# are decimated from 48 kHz, so above ~4.8 kHz their spectrum follows decimate()'s anti-alias
# roll-off rather than the 12 kHz recorder's (about -39 dB vs -58 dB in the top band); without
# a common cutoff, that processing difference alone identifies the Normal class.
LOWPASS = 4000
USE_BANDPASS = False          # optional extra 500 Hz high-pass edge on top of the low-pass
BAND = (500, LOWPASS)

PAT = re.compile(r"^(Normal|IR|B|OR)(\d{3})?(?:@(\d+))?_(\d)\.mat$")
