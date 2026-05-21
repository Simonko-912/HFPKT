#!/usr/bin/env python3
"""
HFPKT Stream Builder
====================
Takes a playlist (JSON or plain text) describing a sequence of packets
and assembles them into a single WAV file — a continuous transmission stream.

JSON format (recommended):
    [
        { "type": "callsign", "callsign": "W1ABC" },
        { "type": "text",     "text": "Hello from W1ABC, sending image next." },
        { "type": "image",    "file": "photo.png", "color": "bw" },
        { "type": "image",    "file": "photo.png", "color": "4",  "chunk": 512 },
        { "type": "audio",    "file": "voice.wav", "chunk": 1000 },
        { "type": "binary",   "file": "data.bin" },
        { "type": "callsign", "callsign": "W1ABC" }
    ]

Plain text format (one entry per line):
    callsign W1ABC
    text Hello from W1ABC
    image photo.png bw
    image photo.png 4 chunk=512
    audio voice.wav chunk=1000
    binary data.bin
    callsign W1ABC

Usage:
    python hfpkt_stream.py playlist.json -o stream.wav
    python hfpkt_stream.py playlist.txt  -o stream.wav
    python hfpkt_stream.py playlist.json -o stream.wav --gap 200 --verbose
    python hfpkt_stream.py --example     # print an example JSON playlist and exit
"""

import argparse
import json
import os
import sys
import numpy as np
import soundfile as sf

# Import encoder internals directly so we don't shell out
from hfpkt_protocol import (
    PacketType, ColorMode, COLOR_MODE_NAMES,
    START_NIBBLES, END_NIBBLES, TYPE_NIBBLES,
    SAMPLES_PER_TONE, SAMPLE_RATE, TONE_DURATION_MS,
    encode_uint16_to_nibbles, ecc_encode,
    bytes_to_nibbles, nibble_to_tone_samples, silence,
)
from hfpkt_encoder import (
    render_nibbles_to_audio,
    build_start, build_end,
    encode_text_payload, encode_binary_payload,
    encode_image_payload, encode_audio_payload,
    assemble_packet, add_inter_packet_gap,
    print_packet_summary,
)

# ─── Colour mode helper ───────────────────────────────────────────────────────

COLOR_MAP = {"bw": ColorMode.BW1BIT, "1": ColorMode.BW1BIT,
             "4":  ColorMode.COLOR4,
             "8":  ColorMode.COLOR8,
             "10": ColorMode.COLOR10}

# ─── Single-entry encoder ─────────────────────────────────────────────────────

def encode_entry(entry: dict, verbose: bool) -> np.ndarray:
    """
    Encode one playlist entry to audio samples.
    Handles all packet types; returns float32 ndarray.
    Raises ValueError on bad config.
    """
    ptype_raw = entry.get("type", "").lower().strip()

    # ── TEXT ──────────────────────────────────────────────────────────────────
    if ptype_raw == "text":
        text = entry.get("text", "")
        if not text:
            raise ValueError("text entry missing 'text' field")
        nibbles, bit_len = encode_text_payload(text)
        header = build_start(PacketType.TEXT, bit_length=bit_len)
        end    = build_end()
        if verbose:
            print_packet_summary(PacketType.TEXT, len(nibbles), bit_length=bit_len)
        return assemble_packet(header, nibbles, end).astype(np.float32)

    # ── CALLSIGN ──────────────────────────────────────────────────────────────
    elif ptype_raw == "callsign":
        call = entry.get("callsign", entry.get("text", ""))
        if not call:
            raise ValueError("callsign entry missing 'callsign' field")
        text = f"CALLSIGN:{call}"
        nibbles, bit_len = encode_text_payload(text)
        header = build_start(PacketType.CALLSIGN, bit_length=bit_len)
        end    = build_end()
        if verbose:
            print_packet_summary(PacketType.CALLSIGN, len(nibbles), bit_length=bit_len)
        return assemble_packet(header, nibbles, end).astype(np.float32)

    # ── IMAGE ─────────────────────────────────────────────────────────────────
    elif ptype_raw in ("image", "image_partial"):
        path = entry.get("file", "")
        if not path or not os.path.exists(path):
            raise ValueError(f"image entry: file not found: {path!r}")
        color_mode = COLOR_MAP.get(str(entry.get("color", "bw")), ColorMode.BW1BIT)
        chunk_size = int(entry.get("chunk", 0))

        nibbles, h, w = encode_image_payload(path, color_mode)

        if chunk_size and len(nibbles) > chunk_size:
            chunks = [nibbles[i:i+chunk_size]
                      for i in range(0, len(nibbles), chunk_size)]
            if verbose:
                print(f"    Splitting into {len(chunks)} partial image packets "
                      f"({chunk_size} nibbles each)")
            parts = []
            for idx, chunk in enumerate(chunks):
                pkt_type = PacketType.IMAGE if idx == 0 else PacketType.IMAGE_PARTIAL
                header = build_start(pkt_type, height=h, width=w,
                                     color_mode=color_mode)
                end    = build_end()
                if verbose:
                    print_packet_summary(pkt_type, len(chunk), height=h, width=w,
                                         color_mode=color_mode)
                parts.append(assemble_packet(header, chunk, end).astype(np.float32))
            # inter-chunk gap (shorter than inter-packet gap)
            gap = silence(int(SAMPLE_RATE * 0.05)).astype(np.float32)
            return np.concatenate([seg for p in parts for seg in [p, gap]])
        else:
            header = build_start(PacketType.IMAGE, height=h, width=w,
                                 color_mode=color_mode)
            end    = build_end()
            if verbose:
                print_packet_summary(PacketType.IMAGE, len(nibbles),
                                     height=h, width=w, color_mode=color_mode)
            return assemble_packet(header, nibbles, end).astype(np.float32)

    # ── AUDIO ─────────────────────────────────────────────────────────────────
    elif ptype_raw == "audio":
        path = entry.get("file", "")
        if not path or not os.path.exists(path):
            raise ValueError(f"audio entry: file not found: {path!r}")
        chunk_size = int(entry.get("chunk", 0))
        nibbles, bit_len = encode_audio_payload(path)

        if chunk_size and len(nibbles) > chunk_size:
            chunks = [nibbles[i:i+chunk_size]
                      for i in range(0, len(nibbles), chunk_size)]
            if verbose:
                print(f"    Splitting into {len(chunks)} audio packets")
            parts = []
            for chunk in chunks:
                bl     = len(chunk) // 2 * 8
                header = build_start(PacketType.AUDIO, bit_length=bl)
                end    = build_end()
                parts.append(assemble_packet(header, chunk, end).astype(np.float32))
            gap = silence(int(SAMPLE_RATE * 0.03)).astype(np.float32)
            return np.concatenate([seg for p in parts for seg in [p, gap]])
        else:
            header = build_start(PacketType.AUDIO, bit_length=bit_len)
            end    = build_end()
            if verbose:
                print_packet_summary(PacketType.AUDIO, len(nibbles), bit_length=bit_len)
            return assemble_packet(header, nibbles, end).astype(np.float32)

    # ── BINARY ────────────────────────────────────────────────────────────────
    elif ptype_raw == "binary":
        path = entry.get("file", "")
        if not path or not os.path.exists(path):
            raise ValueError(f"binary entry: file not found: {path!r}")
        chunk_size = int(entry.get("chunk", 0))
        with open(path, "rb") as f:
            data = f.read()
        nibbles, bit_len = encode_binary_payload(data)

        if chunk_size and len(nibbles) > chunk_size:
            chunks = [nibbles[i:i+chunk_size]
                      for i in range(0, len(nibbles), chunk_size)]
            if verbose:
                print(f"    Splitting into {len(chunks)} binary packets")
            parts = []
            for chunk in chunks:
                bl     = len(chunk) // 2 * 8
                header = build_start(PacketType.BINARY, bit_length=bl)
                end    = build_end()
                parts.append(assemble_packet(header, chunk, end).astype(np.float32))
            gap = silence(int(SAMPLE_RATE * 0.03)).astype(np.float32)
            return np.concatenate([seg for p in parts for seg in [p, gap]])
        else:
            header = build_start(PacketType.BINARY, bit_length=bit_len)
            end    = build_end()
            if verbose:
                print_packet_summary(PacketType.BINARY, len(nibbles), bit_length=bit_len)
            return assemble_packet(header, nibbles, end).astype(np.float32)

    # ── SILENCE (optional spacer) ─────────────────────────────────────────────
    elif ptype_raw == "silence":
        ms = int(entry.get("ms", 500))
        if verbose:
            print(f"    Silence gap: {ms}ms")
        return silence(int(SAMPLE_RATE * ms / 1000)).astype(np.float32)

    else:
        raise ValueError(f"Unknown packet type: {ptype_raw!r}  "
                         f"(valid: text, callsign, image, audio, binary, silence)")

# ─── Playlist Parsers ─────────────────────────────────────────────────────────

def load_json_playlist(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "packets" in data:
        return data["packets"]
    raise ValueError("JSON playlist must be a top-level array of packet objects, "
                     "or a dict with a 'packets' key.")

def load_text_playlist(path: str) -> list[dict]:
    """
    Parse a plain-text playlist. One packet per non-blank, non-comment line.

    Line formats:
        callsign W1ABC
        text Some text here (rest of line = the text)
        image path/to/file.png [bw|4|8|10] [chunk=N]
        audio path/to/file.wav [chunk=N]
        binary path/to/file.bin [chunk=N]
        silence [ms=N]
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)   # split on first whitespace
            ptype = parts[0].lower()
            rest  = parts[1] if len(parts) > 1 else ""

            entry: dict = {"type": ptype}

            if ptype == "callsign":
                entry["callsign"] = rest.strip()

            elif ptype == "text":
                entry["text"] = rest.strip()

            elif ptype in ("image", "image_partial"):
                tokens = rest.split()
                if not tokens:
                    raise ValueError(f"Line {lineno}: image entry needs a file path")
                entry["file"] = tokens[0]
                # Optional color mode
                color_tokens = [t for t in tokens[1:] if not t.startswith("chunk=")]
                if color_tokens:
                    entry["color"] = color_tokens[0]
                # Optional chunk=N
                for t in tokens[1:]:
                    if t.startswith("chunk="):
                        entry["chunk"] = int(t.split("=",1)[1])

            elif ptype in ("audio", "binary"):
                tokens = rest.split()
                if not tokens:
                    raise ValueError(f"Line {lineno}: {ptype} entry needs a file path")
                entry["file"] = tokens[0]
                for t in tokens[1:]:
                    if t.startswith("chunk="):
                        entry["chunk"] = int(t.split("=",1)[1])

            elif ptype == "silence":
                tokens = rest.split()
                for t in tokens:
                    if t.startswith("ms="):
                        entry["ms"] = int(t.split("=",1)[1])

            else:
                raise ValueError(f"Line {lineno}: unknown packet type {ptype!r}")

            entries.append(entry)

    return entries

def load_playlist(path: str) -> list[dict]:
    """Auto-detect JSON vs plain text by extension and content."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".jsonl"):
        return load_json_playlist(path)
    # Try JSON anyway (might be .txt that contains JSON)
    try:
        return load_json_playlist(path)
    except (json.JSONDecodeError, ValueError):
        return load_text_playlist(path)

# ─── Example playlist printer ─────────────────────────────────────────────────

EXAMPLE_JSON = """\
[
    { "type": "callsign", "callsign": "W1ABC" },
    {
        "type": "text",
        "text": "Hello from W1ABC. Sending image and audio. 73."
    },
    {
        "type": "image",
        "file": "photo.png",
        "color": "bw",
        "comment": "colour modes: bw, 4, 8, 10"
    },
    {
        "type": "image",
        "file": "large_photo.png",
        "color": "4",
        "chunk": 512,
        "comment": "chunk splits into IMAGE_PARTIAL packets"
    },
    {
        "type": "audio",
        "file": "voice.wav",
        "chunk": 1000,
        "comment": "chunk recommended for real-time voice"
    },
    {
        "type": "binary",
        "file": "data.bin",
        "comment": "any file — firmware, compressed data, etc."
    },
    { "type": "silence", "ms": 500, "comment": "optional pause between bursts" },
    { "type": "callsign", "callsign": "W1ABC" }
]
"""

EXAMPLE_TEXT = """\
# HFPKT plain-text playlist
# Lines starting with # are comments. Blank lines ignored.

callsign W1ABC
text Hello from W1ABC. Sending image and audio. 73.
image photo.png bw
image large_photo.png 4 chunk=512
audio voice.wav chunk=1000
binary data.bin
silence ms=500
callsign W1ABC
"""

# ─── Stream assembler ─────────────────────────────────────────────────────────

def build_stream(entries: list[dict], gap_ms: int, verbose: bool) -> np.ndarray:
    gap_samples = int(SAMPLE_RATE * gap_ms / 1000)
    segments: list[np.ndarray] = []
    total_tones = 0

    for idx, entry in enumerate(entries):
        ptype = entry.get("type", "?").upper()
        label = entry.get("callsign") or entry.get("text","")[:40] or entry.get("file","")
        print(f"  [{idx+1:02d}/{len(entries):02d}] {ptype:15s}  {label}")

        try:
            audio = encode_entry(entry, verbose)
        except ValueError as e:
            print(f"         ⚠  Skipping: {e}")
            continue

        segments.append(audio)
        # Inter-packet gap (silence)
        segments.append(silence(gap_samples).astype(np.float32))

        n_tones = len(audio) // SAMPLES_PER_TONE
        total_tones += n_tones

    if not segments:
        return np.array([], dtype=np.float32)

    stream = np.concatenate(segments).astype(np.float32)
    return stream

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="hfpkt_stream",
        description="HFPKT Stream Builder — assemble a packet playlist into one WAV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s playlist.json -o stream.wav
  %(prog)s playlist.txt  -o stream.wav --gap 300
  %(prog)s playlist.json -o stream.wav --verbose
  %(prog)s --example-json   > my_playlist.json
  %(prog)s --example-text   > my_playlist.txt

Packet types supported in playlist:
  callsign   Your ham callsign — send at start, end, and periodically
  text       ASCII text message
  image      Image file (PNG, JPEG, BMP …) with optional color mode
  audio      Audio file (WAV, FLAC, OGG …) with optional chunk split
  binary     Any binary file
  silence    Optional silence gap (ms=N)
        """)

    parser.add_argument("playlist", nargs="?",
                        help="Playlist file (.json or .txt)")
    parser.add_argument("-o", "--output", default="stream.wav",
                        help="Output WAV file (default: stream.wav)")
    parser.add_argument("--gap", type=int, default=150,
                        help="Silence gap between packets in ms (default: 150)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-packet tone/timing details")
    parser.add_argument("--example-json", action="store_true",
                        help="Print an example JSON playlist and exit")
    parser.add_argument("--example-text", action="store_true",
                        help="Print an example plain-text playlist and exit")

    args = parser.parse_args()

    if args.example_json:
        print(EXAMPLE_JSON)
        sys.exit(0)
    if args.example_text:
        print(EXAMPLE_TEXT)
        sys.exit(0)
    if not args.playlist:
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.playlist):
        print(f"Error: playlist file not found: {args.playlist}", file=sys.stderr)
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n[HFPKT Stream Builder]")
    print(f"  Playlist      : {args.playlist}")
    print(f"  Output        : {args.output}")
    print(f"  Gap between   : {args.gap}ms\n")

    try:
        entries = load_playlist(args.playlist)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing playlist: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"  {len(entries)} packet(s) in playlist\n")
    print("  Building stream:")
    print("  " + "─" * 52)

    # ── Build ─────────────────────────────────────────────────────────────────
    stream = build_stream(entries, args.gap, args.verbose)

    if len(stream) == 0:
        print("\n  No packets were encoded. Check your playlist.")
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────────────────────
    sf.write(args.output, stream, SAMPLE_RATE)
    duration = len(stream) / SAMPLE_RATE

    print("  " + "─" * 52)
    print(f"\n  Total duration : {duration:.2f}s  ({duration*1000:.0f}ms)")
    print(f"  Total samples  : {len(stream):,}")
    print(f"  Approx tones   : {len(stream) // SAMPLES_PER_TONE:,}")
    print(f"  Saved to       : {args.output}\n")
    print("  Decode with:")
    print(f"    python hfpkt_decoder.py {args.output} --out-dir ./decoded\n")

if __name__ == "__main__":
    main()
