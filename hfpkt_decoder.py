#!/usr/bin/env python3
"""
HFPKT Decoder
=============
Decodes HFPKT audio (.wav) back into text, images, audio, or binary files.

Usage:
    python hfpkt_decoder.py input.wav
    python hfpkt_decoder.py input.wav --out-dir ./decoded
    python hfpkt_decoder.py input.wav --verbose
    python hfpkt_decoder.py input.wav --threshold 0.25
"""

import argparse
import sys
import os
import struct
import time
import numpy as np
import soundfile as sf
from PIL import Image

from hfpkt_protocol import (
    PacketType, ColorMode, COLOR_MODE_NAMES, PACKET_TYPE_NAMES,
    START_NIBBLES, END_NIBBLES, TYPE_NIBBLES,
    SAMPLES_PER_TONE, SAMPLE_RATE, TONE_FREQ, DATA_TONES, NUM_TONES,
    TONE_TOLERANCE, TONE_DURATION_MS,
    encode_uint16_to_nibbles, ecc_decode, decode_nibbles_to_uint16,
    nibbles_to_bytes, detect_nibble, detect_nibble_sequence,
)

# ─── Sync Pattern Matching ────────────────────────────────────────────────────

def nibbles_match(detected: list[int], expected: list[int],
                  tolerance: int = 1) -> bool:
    """Return True if detected nibble list matches expected within tolerance."""
    if len(detected) != len(expected):
        return False
    mismatches = sum(1 for a, b in zip(detected, expected) if a != b)
    return mismatches <= tolerance

# ─── Frame Scanner ────────────────────────────────────────────────────────────

class HFPKTDecoder:
    def __init__(self, audio: np.ndarray, sr: int, verbose: bool = False,
                 energy_threshold: float = 0.05):
        if sr != SAMPLE_RATE:
            # Simple nearest-neighbour resample
            ratio = SAMPLE_RATE / sr
            n_out = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio)-1, n_out).astype(int)
            audio = audio[indices]
        self.audio   = audio.astype(np.float32)
        self.verbose = verbose
        self.threshold = energy_threshold
        self.cursor  = 0           # current sample position
        self.packets : list[dict] = []

    def log(self, *args):
        if self.verbose:
            print(" ", *args)

    def remaining_samples(self) -> int:
        return len(self.audio) - self.cursor

    def read_nibble(self) -> tuple[int, float] | None:
        """Read one tone symbol at cursor. Returns (nibble, confidence) or None."""
        if self.remaining_samples() < SAMPLES_PER_TONE:
            return None
        chunk = self.audio[self.cursor: self.cursor + SAMPLES_PER_TONE]
        nibble, conf = detect_nibble(chunk)
        self.cursor += SAMPLES_PER_TONE
        return nibble, conf

    def read_nibbles(self, count: int) -> list[int] | None:
        """Read `count` consecutive nibbles from cursor. Returns None if EOF."""
        result = []
        for _ in range(count):
            r = self.read_nibble()
            if r is None:
                return None
            result.append(r[0])
        return result

    def check_energy(self) -> bool:
        """Return True if the next tone window has enough signal energy."""
        if self.remaining_samples() < SAMPLES_PER_TONE:
            return False
        chunk = self.audio[self.cursor: self.cursor + SAMPLES_PER_TONE]
        rms = float(np.sqrt(np.mean(chunk**2)))
        return rms > self.threshold

    def find_start(self) -> bool:
        """
        Advance cursor until START pattern is found.
        Uses coarse one-tone steps through silence, then fine (10-sample) sub-tone
        alignment search on near-matches to handle inter-packet boundary drift.
        Returns True if found.
        """
        pattern_len = len(START_NIBBLES)          # 5 nibbles
        FINE_STEP = 10                             # sub-sample alignment step

        while self.remaining_samples() >= SAMPLES_PER_TONE * pattern_len:
            if not self.check_energy():
                self.cursor += SAMPLES_PER_TONE   # skip silence in coarse steps
                continue

            # Coarse check: peek pattern_len nibbles from current cursor
            save = self.cursor
            nib = self.read_nibbles(pattern_len)
            if nib is None:
                return False

            matches = sum(1 for a, b in zip(nib, START_NIBBLES) if a == b)

            if matches >= pattern_len - 1:        # allow 1 mismatch → confirmed
                self.log(f"START pattern found at sample {save} "
                         f"({save/SAMPLE_RATE*1000:.1f}ms)")
                # Consume ECC copy
                nib2 = self.read_nibbles(pattern_len)
                if nib2 and sum(1 for a,b in zip(nib2,START_NIBBLES) if a==b) >= pattern_len-2:
                    self.log("START ECC copy confirmed")
                return True

            elif matches >= pattern_len - 2:      # near-miss: try fine alignment
                # Search within ±SAMPLES_PER_TONE/2 in FINE_STEP increments
                best_save   = save
                best_match  = matches
                half_tone   = SAMPLES_PER_TONE // 2
                for fine in range(FINE_STEP, half_tone, FINE_STEP):
                    for sign in (1, -1):
                        trial = save + sign * fine
                        if trial < 0 or trial + SAMPLES_PER_TONE * pattern_len > len(self.audio):
                            continue
                        self.cursor = trial
                        trial_nibs = self.read_nibbles(pattern_len)
                        if trial_nibs is None:
                            continue
                        m = sum(1 for a,b in zip(trial_nibs, START_NIBBLES) if a==b)
                        if m > best_match:
                            best_match = m
                            best_save  = trial
                        if m >= pattern_len - 1:
                            break
                    if best_match >= pattern_len - 1:
                        break

                if best_match >= pattern_len - 1:
                    self.cursor = best_save + SAMPLES_PER_TONE * pattern_len
                    self.log(f"START pattern found (fine-aligned) at sample {best_save} "
                             f"({best_save/SAMPLE_RATE*1000:.1f}ms)  match={best_match}/5")
                    nib2 = self.read_nibbles(pattern_len)
                    if nib2 and sum(1 for a,b in zip(nib2,START_NIBBLES) if a==b) >= pattern_len-2:
                        self.log("START ECC copy confirmed")
                    return True
                else:
                    # Still no luck — advance one tone and continue coarse scan
                    self.cursor = save + SAMPLES_PER_TONE
            else:
                # Clear mismatch — rewind to one tone after save and continue
                self.cursor = save + SAMPLES_PER_TONE

        return False

    def read_type(self) -> PacketType | None:
        """Read the TYPE field (2 nibbles × 2 for ECC)."""
        nib1 = self.read_nibbles(2)
        nib2 = self.read_nibbles(2)
        if nib1 is None or nib2 is None:
            return None

        # Match against known type nibble patterns
        best_type   = None
        best_score  = -1
        for ptype, pattern in TYPE_NIBBLES.items():
            # Try both copies and pick best match
            score1 = sum(1 for a,b in zip(nib1, pattern) if a==b)
            score2 = sum(1 for a,b in zip(nib2, pattern) if a==b)
            score  = max(score1, score2)
            if score > best_score:
                best_score = score
                best_type  = ptype

        self.log(f"Packet type: {PacketType(best_type).name} (match score {best_score}/2)")
        return PacketType(best_type)

    def read_color_mode(self) -> ColorMode:
        """Read colour mode nibble (×2 ECC)."""
        a = self.read_nibble()
        b = self.read_nibble()
        val = a[0] if a else 0
        return ColorMode(min(val, 3))

    def read_uint16_ecc(self) -> int:
        """Read a 16-bit integer stored as 4 nibbles × 2 (ECC)."""
        raw = self.read_nibbles(8)   # 4 nibbles × 2
        if raw is None:
            return 0
        decoded = ecc_decode(raw)    # 4 nibbles
        return decode_nibbles_to_uint16(decoded)

    def read_payload(self, n_nibbles: int) -> list[int]:
        """Read exactly n_nibbles of payload data."""
        result = []
        for _ in range(n_nibbles):
            r = self.read_nibble()
            if r is None:
                break
            result.append(r[0])
        return result

    def consume_end(self) -> bool:
        """
        Consume the END pattern (and its ECC copy) from the current cursor.
        Used after reading a known-length payload. Returns True if confirmed.
        """
        pattern_len = len(END_NIBBLES)
        save = self.cursor
        nib1 = self.read_nibbles(pattern_len)
        if nib1 and nibbles_match(nib1, END_NIBBLES, tolerance=1):
            self.log("END pattern found")
            # Try ECC copy
            save2 = self.cursor
            nib2  = self.read_nibbles(pattern_len)
            if nib2 is None or not nibbles_match(nib2, END_NIBBLES, tolerance=2):
                self.cursor = save2   # rewind if ECC copy not confirmed
            return True
        # END not where expected — rewind and fall back to scan
        self.cursor = save
        self.log("END not at expected position — scanning")
        return self._scan_for_end()

    def _scan_for_end(self) -> bool:
        """
        Fallback: scan forward up to 20 extra nibbles looking for END pattern.
        Does NOT modify payload; just advances cursor past END.
        """
        pattern_len = len(END_NIBBLES)
        window: list[int] = []
        for _ in range(20 + pattern_len):
            r = self.read_nibble()
            if r is None:
                return False
            window.append(r[0])
            if len(window) >= pattern_len:
                tail = window[-pattern_len:]
                if nibbles_match(tail, END_NIBBLES, tolerance=1):
                    self.log("END pattern found (scan fallback)")
                    save = self.cursor
                    nib2 = self.read_nibbles(pattern_len)
                    if nib2 is None or not nibbles_match(nib2, END_NIBBLES, tolerance=2):
                        self.cursor = save
                    return True
        return False

    def find_end(self, payload_nibbles: list[int]) -> list[int]:
        """
        Legacy greedy scan — only used when payload length is truly unknown.
        Reads forward until END pattern appears and strips it from payload.
        """
        pattern_len = len(END_NIBBLES)
        window: list[int] = []
        result: list[int] = list(payload_nibbles)

        while True:
            r = self.read_nibble()
            if r is None:
                break
            window.append(r[0])
            result.append(r[0])

            if len(window) >= pattern_len:
                tail = window[-pattern_len:]
                if nibbles_match(tail, END_NIBBLES, tolerance=1):
                    self.log("END pattern found")
                    save = self.cursor
                    nib2 = self.read_nibbles(pattern_len)
                    if nib2 is None or not nibbles_match(nib2, END_NIBBLES, tolerance=2):
                        self.cursor = save
                    result = result[:-pattern_len]
                    break

        return result

    # ─── Packet Decoders ──────────────────────────────────────────────────────

    def decode_text_packet(self, bit_length: int) -> str:
        n_nibbles = (bit_length + 3) // 4
        self.log(f"Reading {n_nibbles} text nibbles ({bit_length} bits)")
        payload = self.read_payload(n_nibbles)
        self.consume_end()
        data = nibbles_to_bytes(payload)
        n_bytes = bit_length // 8
        data = data[:n_bytes]
        return data.decode("ascii", errors="replace")

    def decode_binary_packet(self, bit_length: int) -> bytes:
        n_nibbles = (bit_length + 3) // 4
        self.log(f"Reading {n_nibbles} binary nibbles")
        payload = self.read_payload(n_nibbles)
        self.consume_end()
        return nibbles_to_bytes(payload)[: bit_length // 8]

    def decode_audio_packet(self, bit_length: int) -> np.ndarray:
        raw = self.decode_binary_packet(bit_length)
        u8  = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        pcm = (u8 / 255.0 - 0.5) * 2.0   # back to float [-1, 1]
        return pcm

    def decode_image_packet(self, color_mode: ColorMode,
                            height: int, width: int) -> Image.Image | None:
        if height == 0 or width == 0:
            self.log("Image has zero dimensions — skipping")
            return None

        if color_mode == ColorMode.BW1BIT:
            n_pixels = height * width
            n_nibbles = (n_pixels + 3) // 4
            payload = self.read_payload(n_nibbles)
            self.consume_end()
            # Unpack 4 pixels per nibble
            pixels = []
            for nib in payload:
                pixels.extend([(nib>>3)&1, (nib>>2)&1, (nib>>1)&1, nib&1])
            pixels = pixels[:n_pixels]
            arr = np.array(pixels, dtype=np.uint8).reshape(height, width) * 255
            return Image.fromarray(arr, mode="L").convert("1")

        elif color_mode == ColorMode.COLOR4:
            n_nibbles = height * width
            payload = self.read_payload(n_nibbles)
            self.consume_end()
            payload = payload[:height*width]
            # Map 4-bit palette to greyscale (0-15 → 0-255)
            arr = np.array(payload, dtype=np.uint8).reshape(height, width)
            arr = (arr * 17).astype(np.uint8)   # 15*17=255
            return Image.fromarray(arr, mode="L")

        elif color_mode in (ColorMode.COLOR8, ColorMode.COLOR10):
            n_nibbles = height * width * 2
            payload = self.read_payload(n_nibbles)
            self.consume_end()
            data = nibbles_to_bytes(payload)[:height*width]
            arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
            return Image.fromarray(arr, mode="L")

    # ─── Main Scan Loop ───────────────────────────────────────────────────────

    def decode_all(self) -> list[dict]:
        packets = []
        pkt_idx = 0

        while self.remaining_samples() >= SAMPLES_PER_TONE * 10:
            self.log(f"\n--- Searching for packet {pkt_idx+1} "
                     f"(cursor {self.cursor}/{len(self.audio)}) ---")

            if not self.find_start():
                self.log("No more START patterns found.")
                break

            ptype = self.read_type()
            if ptype is None:
                self.log("Could not decode packet type — skipping")
                self.cursor += SAMPLES_PER_TONE
                continue

            pkt: dict = {"type": ptype, "index": pkt_idx}

            if ptype in (PacketType.IMAGE, PacketType.IMAGE_PARTIAL):
                color_mode = self.read_color_mode()
                height     = self.read_uint16_ecc()
                width      = self.read_uint16_ecc()
                self.log(f"Image {width}×{height}, mode={color_mode.name}")
                pkt["color_mode"] = color_mode
                pkt["height"]     = height
                pkt["width"]      = width
                pkt["image"]      = self.decode_image_packet(color_mode, height, width)

            elif ptype in (PacketType.TEXT, PacketType.CALLSIGN):
                bit_length = self.read_uint16_ecc()
                self.log(f"Text/callsign bit_length={bit_length}")
                pkt["bit_length"] = bit_length
                pkt["text"]       = self.decode_text_packet(bit_length)

            elif ptype == PacketType.AUDIO:
                bit_length = self.read_uint16_ecc()
                self.log(f"Audio bit_length={bit_length}")
                pkt["bit_length"] = bit_length
                pkt["audio_pcm"]  = self.decode_audio_packet(bit_length)

            elif ptype == PacketType.BINARY:
                bit_length = self.read_uint16_ecc()
                self.log(f"Binary bit_length={bit_length}")
                pkt["bit_length"] = bit_length
                pkt["data"]       = self.decode_binary_packet(bit_length)

            packets.append(pkt)
            pkt_idx += 1

        self.packets = packets
        return packets

# ─── Output Writers ───────────────────────────────────────────────────────────

def save_packet(pkt: dict, out_dir: str, base_name: str):
    os.makedirs(out_dir, exist_ok=True)
    idx   = pkt["index"]
    ptype = pkt["type"]
    stamp = f"{base_name}_pkt{idx:03d}"

    if ptype in (PacketType.TEXT, PacketType.CALLSIGN):
        path = os.path.join(out_dir, f"{stamp}_text.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(pkt.get("text", ""))
        print(f"  [{idx}] TEXT  → {path}")
        preview = pkt.get("text","")[:80].replace("\n"," ")
        print(f"       Preview : {preview!r}")

    elif ptype in (PacketType.IMAGE, PacketType.IMAGE_PARTIAL):
        img = pkt.get("image")
        if img:
            path = os.path.join(out_dir, f"{stamp}_image.png")
            img.save(path)
            print(f"  [{idx}] IMAGE → {path}  "
                  f"({pkt['width']}×{pkt['height']} {pkt['color_mode'].name})")
        else:
            print(f"  [{idx}] IMAGE → (empty/invalid)")

    elif ptype == PacketType.AUDIO:
        pcm  = pkt.get("audio_pcm")
        if pcm is not None and len(pcm):
            path = os.path.join(out_dir, f"{stamp}_audio.wav")
            sf.write(path, pcm.astype(np.float32), 8000)
            dur  = len(pcm) / 8000
            print(f"  [{idx}] AUDIO → {path}  ({dur:.2f}s @ 8kHz)")
        else:
            print(f"  [{idx}] AUDIO → (empty)")

    elif ptype == PacketType.BINARY:
        data = pkt.get("data", b"")
        path = os.path.join(out_dir, f"{stamp}_data.bin")
        with open(path, "wb") as f:
            f.write(data)
        print(f"  [{idx}] BIN   → {path}  ({len(data)} bytes)")

# ─── Spectrum Analyser (text art) ─────────────────────────────────────────────

def print_spectrum(audio: np.ndarray, title: str = "Spectrum", n_points: int = 60):
    """Print a crude text-art FFT of the first 0.5s of audio."""
    segment = audio[:int(SAMPLE_RATE * 0.5)]
    if len(segment) == 0:
        return
    n = len(segment)
    freqs   = np.fft.rfftfreq(n, 1 / SAMPLE_RATE)
    magnitudes = np.abs(np.fft.rfft(segment * np.hanning(n)))

    # Sample at tone frequencies
    bar_max = 30
    print(f"\n  {title}")
    print(f"  {'Hz':>6}  {'Tone':>4}  Level")
    for i, f in enumerate(TONE_FREQ[:DATA_TONES]):
        lo = np.searchsorted(freqs, f - TONE_TOLERANCE)
        hi = np.searchsorted(freqs, f + TONE_TOLERANCE)
        energy = magnitudes[lo:hi+1].max() if lo < hi else 0.0
        norm   = min(int(energy / (magnitudes.max()+1e-9) * bar_max), bar_max)
        bar    = "█" * norm + "░" * (bar_max - norm)
        print(f"  {f:>6}Hz  T{i+1:02d}  {bar}")
    print()

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="hfpkt_decoder",
        description="HFPKT Ham Radio Frequency Packet Decoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Decodes a HFPKT WAV file produced by hfpkt_encoder.py.
All detected packets are saved to --out-dir with auto-named files.

Examples:
  %(prog)s signal.wav
  %(prog)s signal.wav --out-dir ./decoded --verbose
  %(prog)s signal.wav --spectrum
  %(prog)s signal.wav --threshold 0.03
        """)

    parser.add_argument("input",          help="Input WAV file")
    parser.add_argument("--out-dir",      default="decoded",
                        help="Output directory for decoded files (default: ./decoded)")
    parser.add_argument("--threshold",    type=float, default=0.05,
                        help="Minimum RMS energy to consider a tone present (default: 0.05)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose decoder output")
    parser.add_argument("--spectrum",     action="store_true",
                        help="Print a text-art FFT spectrum of the first 0.5s")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[HFPKT Decoder]")
    print(f"  Input file    : {args.input}")
    print(f"  Output dir    : {args.out_dir}")
    print(f"  Energy thresh : {args.threshold}")
    print(f"  Verbose       : {args.verbose}")

    audio, sr = sf.read(args.input, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)   # mix to mono
    duration = len(audio) / sr
    print(f"  Duration      : {duration:.2f}s  ({len(audio)} samples @ {sr}Hz)")

    if args.spectrum:
        print_spectrum(audio, title="Input Signal Spectrum (first 0.5s)")

    t0 = time.time()
    decoder = HFPKTDecoder(audio, sr,
                           verbose=args.verbose,
                           energy_threshold=args.threshold)
    packets = decoder.decode_all()
    elapsed = time.time() - t0

    print(f"\n  Decoded {len(packets)} packet(s) in {elapsed:.2f}s\n")

    if not packets:
        print("  No HFPKT packets found in audio.")
        print("  Tips:")
        print("    • Lower --threshold if signal is quiet")
        print("    • Use --verbose to trace the scanner")
        print("    • Use --spectrum to check tone presence")
        sys.exit(0)

    base = os.path.splitext(os.path.basename(args.input))[0]
    for pkt in packets:
        save_packet(pkt, args.out_dir, base)

    print(f"\n  All files saved to: {args.out_dir}/\n")

if __name__ == "__main__":
    main()
