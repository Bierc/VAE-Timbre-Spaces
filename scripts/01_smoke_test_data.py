import json
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


# ----------------------------
# Paths
# ----------------------------
DATA_ROOT = Path("data/nsynth-valid.jsonwav/nsynth-valid")
AUDIO_DIR = DATA_ROOT / "audio"
JSON_PATH = DATA_ROOT / "examples.json"
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Load metadata
# ----------------------------
with open(JSON_PATH, "r") as f:
    examples = json.load(f)

print(f"Total examples: {len(examples)}")

# Pick one example
example_key = list(examples.keys())[0] # keys: 
example = examples[example_key]

print("Example key:", example_key)
print("Pitch:", example["pitch"])
print("Instrument family:", example["instrument_family"])
print("Instrument source:", example["instrument_source"])

# ----------------------------
# Load audio
# ----------------------------
wav_path = AUDIO_DIR / f"{example_key}.wav"
audio, sr = librosa.load(wav_path, sr=16000, mono=True)

print("Audio shape:", audio.shape)
print("Sample rate:", sr)

# Save raw audio
sf.write(OUTPUT_DIR / "example.wav", audio, sr)

# ----------------------------
# Log-mel spectrogram
# ----------------------------
mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=1024,
    hop_length=256,
    n_mels=80
)

log_mel = librosa.power_to_db(mel, ref=np.max)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8, 4))
librosa.display.specshow(
    log_mel,
    sr=sr,
    hop_length=256,
    x_axis="time",
    y_axis="mel"
)
plt.colorbar(format="%+2.0f dB")
plt.title("Log-Mel Spectrogram (NSynth example)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "example_logmel.png")
plt.close()

print("Saved outputs to:", OUTPUT_DIR)
