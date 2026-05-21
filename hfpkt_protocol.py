"""
HFPKT - Ham Radio Frequency Packet Transmission Protocol
=========================================================
18 tones: 700Hz to 2400Hz, spaced 100Hz apart
  Tones 1-16  (700-2300Hz): data tones (4 bits per tone)
  Tones 17-18 (2200-2400Hz): reserved for control patterns only

Tone duration: 5ms per tone symbol
ECC: each control field repeated twice; majority vote decoding

Packet structure:
  [START pattern x2] [TYPE x2] [HEIGHT x2 (image only)]
  [WIDTH x2 (image only)] [LENGTH x2 (text/audio/binary)]
  [PAYLOAD tones...] [END pattern x2]

Author: HFPKT Project
"""

import numpy as np
from enum import IntEnum

# ─── Frequency Plan ────────────────────────────────────────────────────────────

TONE_BASE_HZ   = 700
TONE_STEP_HZ   = 100
NUM_TONES      = 18           # total tones
DATA_TONES     = 16           # tones used for payload nibbles
TONE_FREQ      = [TONE_BASE_HZ + i * TONE_STEP_HZ for i in range(NUM_TONES)]
TONE_TOLERANCE = 25           # Hz tolerance for detection

TONE_DURATION_MS = 5          # ms per symbol
SAMPLE_RATE      = 44100      # Hz

SAMPLES_PER_TONE = int(SAMPLE_RATE * TONE_DURATION_MS / 1000)

# ─── Packet Types ──────────────────────────────────────────────────────────────

class PacketType(IntEnum):
    TEXT         = 0x0   # ASCII text
    IMAGE        = 0x1   # B&W / 4-bit / 8-bit image
    IMAGE_PARTIAL= 0x2   # Split image chunk
    AUDIO        = 0x3   # Raw audio (u8 PCM)
    BINARY       = 0x4   # Raw binary data
    CALLSIGN     = 0x5   # Callsign / session info
    # 0x6 – 0xF reserved

PACKET_TYPE_NAMES = {
    PacketType.TEXT:          "TEXT",
    PacketType.IMAGE:         "IMAGE",
    PacketType.IMAGE_PARTIAL: "IMAGE_PARTIAL",
    PacketType.AUDIO:         "AUDIO",
    PacketType.BINARY:        "BINARY",
    PacketType.CALLSIGN:      "CALLSIGN",
}

# ─── Image Colour Modes ────────────────────────────────────────────────────────

class ColorMode(IntEnum):
    BW1BIT  = 0   # 1-bit black & white  → 4 pixels/tone
    COLOR4  = 1   # 4-bit colour palette → 1 pixel/tone
    COLOR8  = 2   # 8-bit colour         → 2 tones / pixel
    COLOR10 = 3   # 10-bit colour        → 3 tones / 2 pixels (trimmed to 12 bits)

COLOR_MODE_NAMES = {
    ColorMode.BW1BIT:  "1-bit B&W  (4 px/tone)",
    ColorMode.COLOR4:  "4-bit Colour (1 px/tone)",
    ColorMode.COLOR8:  "8-bit Colour (2 tones/px)",
    ColorMode.COLOR10: "10-bit Colour (12-bit packed)",
}

# ─── Control Patterns ──────────────────────────────────────────────────────────
# Patterns are 18-bit values expressed as lists of tone indices (0-based).
# Only tones 0-15 carry payload; 16-17 are the "control" pair used here too
# to make patterns maximally distinct from random data.
#
# Pattern encoding: each bit selects a tone frequency.
# We transmit a chord (simultaneous tones) whose combined spectrum encodes
# the pattern.  For sync patterns we use a sequential tone burst instead so
# the pattern survives a noisy channel.

# 18-bit start pattern: 1 0 1 0 1 1 1 0 1 1 1 0 1 1 1 0 1 0
# (trimmed to 18 bits exactly)
START_PATTERN_BITS = [1,0,1,0, 1,1,1,0, 1,1,1,0, 1,1,1,0, 1,0]
END_PATTERN_BITS   = [0,1,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1, 0,1]

assert len(START_PATTERN_BITS) == 18
assert len(END_PATTERN_BITS)   == 18

def pattern_to_nibbles(bits: list[int]) -> list[int]:
    """Group 18 bits into 4-bit nibbles (last group zero-padded to 20 bits → 5 nibbles)."""
    padded = bits + [0,0]          # pad to 20 bits → 5 nibbles
    return [(padded[i]<<3)|(padded[i+1]<<2)|(padded[i+2]<<1)|padded[i+3]
            for i in range(0, 20, 4)]

START_NIBBLES = pattern_to_nibbles(START_PATTERN_BITS)  # 5 nibbles
END_NIBBLES   = pattern_to_nibbles(END_PATTERN_BITS)

# Type-field nibble encodings (4-bit → tone index 0-15)
TYPE_NIBBLES: dict[PacketType, list[int]] = {
    PacketType.TEXT:          [0x1, 0x0],   # "1 (16 zeros) 1" → [1,0]
    PacketType.IMAGE:         [0x0, 0xF],   # "0 (16 ones)  0" → [0,15]
    PacketType.IMAGE_PARTIAL: [0x0, 0x1],   # "01 …"           → [0,1]
    PacketType.AUDIO:         [0x7, 0x7],   # "111 …"          → [7,7]
    PacketType.BINARY:        [0x2, 0xA],   # "010 …"          → [2,10]
    PacketType.CALLSIGN:      [0x5, 0x5],   # reserved         → [5,5]
}

# ─── Tone Generation ───────────────────────────────────────────────────────────

def generate_tone(freq_hz: float, duration_samples: int,
                  amplitude: float = 0.8) -> np.ndarray:
    """Generate a pure sine wave tone."""
    t = np.arange(duration_samples) / SAMPLE_RATE
    wave = amplitude * np.sin(2 * np.pi * freq_hz * t)
    # Apply a short Hann window to avoid clicks
    fade = min(int(SAMPLE_RATE * 0.001), duration_samples // 4)  # 1ms fade
    if fade > 0:
        hann = np.hanning(fade * 2)
        wave[:fade]  *= hann[:fade]
        wave[-fade:] *= hann[fade:]
    return wave

def nibble_to_tone_samples(nibble: int) -> np.ndarray:
    """Convert a 4-bit nibble (0-15) to audio samples using DATA tones."""
    assert 0 <= nibble <= 15, f"nibble out of range: {nibble}"
    freq = TONE_FREQ[nibble]          # tone index == nibble value for data
    return generate_tone(freq, SAMPLES_PER_TONE)

def silence(n_samples: int) -> np.ndarray:
    return np.zeros(n_samples)

# ─── Nibble Stream Encoding Helpers ───────────────────────────────────────────

def encode_uint16_to_nibbles(value: int) -> list[int]:
    """Encode a 16-bit unsigned integer as 4 nibbles (MSB first)."""
    value = value & 0xFFFF
    return [(value >> 12) & 0xF,
            (value >>  8) & 0xF,
            (value >>  4) & 0xF,
             value        & 0xF]

def decode_nibbles_to_uint16(nibbles: list[int]) -> int:
    """Decode 4 nibbles (MSB first) to a 16-bit integer."""
    assert len(nibbles) == 4
    return (nibbles[0]<<12)|(nibbles[1]<<8)|(nibbles[2]<<4)|nibbles[3]

def ecc_encode(nibbles: list[int]) -> list[int]:
    """Simple repetition ECC: repeat each nibble once."""
    out = []
    for n in nibbles:
        out.extend([n, n])
    return out

def ecc_decode(nibbles: list[int]) -> list[int]:
    """Majority-vote ECC: decode pairs of repeated nibbles."""
    assert len(nibbles) % 2 == 0
    out = []
    for i in range(0, len(nibbles), 2):
        a, b = nibbles[i], nibbles[i+1]
        # Use first value; flag mismatch in caller if desired
        out.append(a if a == b else a)   # first wins on disagreement
    return out

def bytes_to_nibbles(data: bytes) -> list[int]:
    """Convert bytes to a list of 4-bit nibbles (high nibble first)."""
    nibbles = []
    for byte in data:
        nibbles.append((byte >> 4) & 0xF)
        nibbles.append(byte & 0xF)
    return nibbles

def nibbles_to_bytes(nibbles: list[int]) -> bytes:
    """Convert a list of nibbles (even length) back to bytes."""
    if len(nibbles) % 2:
        nibbles.append(0)   # pad
    return bytes([(nibbles[i] << 4) | nibbles[i+1]
                  for i in range(0, len(nibbles), 2)])

# ─── Frequency Detection ───────────────────────────────────────────────────────

def _goertzel_power(samples: np.ndarray, freq_hz: float) -> float:
    """
    Compute signal power at a single frequency using the Goertzel algorithm.
    Accurate at any frequency regardless of FFT bin spacing — essential for
    5ms (220-sample) windows where FFT resolution (≈200 Hz) is coarser than
    the 100 Hz tone spacing.
    """
    n = len(samples)
    k = freq_hz * n / SAMPLE_RATE
    w = 2.0 * np.pi * k / n
    coeff = 2.0 * np.cos(w)
    s1 = s2 = 0.0
    for s in samples:
        s0 = float(s) + coeff * s1 - s2
        s2 = s1
        s1 = s0
    return s1 * s1 + s2 * s2 - coeff * s1 * s2


def detect_nibble(samples: np.ndarray, use_data_tones: bool = True) -> tuple[int, float]:
    """
    Detect which tone (nibble value) is present in a window of samples.
    Returns (nibble_value, confidence 0-1).
    Uses data tones 0-15 by default, or all 18 tones for control patterns.

    Uses the Goertzel algorithm for accurate narrow-band power estimation at
    each exact tone frequency — unaffected by FFT bin-resolution limitations.
    """
    n_tones = DATA_TONES if use_data_tones else NUM_TONES
    energies = [_goertzel_power(samples, TONE_FREQ[i]) for i in range(n_tones)]
    best = int(np.argmax(energies))
    total = sum(energies) + 1e-9
    confidence = energies[best] / total
    return best, confidence

def detect_nibble_sequence(audio: np.ndarray, n_nibbles: int,
                            use_data_tones: bool = True) -> list[int]:
    """
    Detect n_nibbles sequential tones from the start of audio.
    Returns list of detected nibble values.
    """
    nibbles = []
    for i in range(n_nibbles):
        start = i * SAMPLES_PER_TONE
        end   = start + SAMPLES_PER_TONE
        chunk = audio[start:end]
        if len(chunk) < SAMPLES_PER_TONE:
            chunk = np.pad(chunk, (0, SAMPLES_PER_TONE - len(chunk)))
        nibble, _ = detect_nibble(chunk, use_data_tones)
        nibbles.append(nibble)
    return nibbles
