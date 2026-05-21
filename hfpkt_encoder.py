#!/usr/bin/env python3
"""
HFPKT Encoder
=============
Encodes text, images, audio, or binary files into HFPKT audio (.wav).

Usage:
    python hfpkt_encoder.py --help
    python hfpkt_encoder.py text  "Hello, 73 de W1ABC" -o out.wav
    python hfpkt_encoder.py image photo.png --color bw -o out.wav
    python hfpkt_encoder.py audio voice.wav -o out.wav
    python hfpkt_encoder.py binary data.bin -o out.wav
    python hfpkt_encoder.py callsign "W1ABC" -o out.wav
"""

import argparse
import sys
import os
import struct
import numpy as np
import soundfile as sf
from PIL import Image

from hfpkt_protocol import (
    PacketType, ColorMode, COLOR_MODE_NAMES,
    START_NIBBLES, END_NIBBLES, TYPE_NIBBLES,
    SAMPLES_PER_TONE, SAMPLE_RATE, TONE_FREQ, DATA_TONES,
    encode_uint16_to_nibbles, ecc_encode,
    bytes_to_nibbles, nibble_to_tone_samples, silence,
    generate_tone, TONE_DURATION_MS,
)

# ─── Tone Rendering ────────────────────────────────────────────────────────────

def render_nibbles_to_audio(nibbles: list[int]) -> np.ndarray:
    """Render a list of nibble values (0-15) into audio samples."""
    chunks = [nibble_to_tone_samples(n) for n in nibbles]
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)

def render_control_nibbles(nibbles: list[int]) -> np.ndarray:
    """Render control-pattern nibbles (same tones, just labelled separately)."""
    return render_nibbles_to_audio(nibbles)

# ─── Header Builders ──────────────────────────────────────────────────────────

def build_start(pkt_type: PacketType, height: int = 0, width: int = 0,
                bit_length: int = 0, color_mode: ColorMode = ColorMode.BW1BIT
                ) -> list[int]:
    """
    Build the header nibble sequence (before payload) with ECC.
    Returns list of nibbles ready to render.
    """
    stream: list[int] = []

    # START pattern × 2
    stream.extend(START_NIBBLES)
    stream.extend(START_NIBBLES)

    # TYPE × 2 (ECC)
    type_nib = TYPE_NIBBLES[pkt_type]
    stream.extend(type_nib)
    stream.extend(type_nib)

    if pkt_type in (PacketType.IMAGE, PacketType.IMAGE_PARTIAL):
        # Color mode nibble × 2
        cm = [int(color_mode), int(color_mode)]
        stream.extend(cm)

        # HEIGHT 16-bit → 4 nibbles × 2
        hn = encode_uint16_to_nibbles(height)
        stream.extend(ecc_encode(hn))

        # WIDTH 16-bit → 4 nibbles × 2
        wn = encode_uint16_to_nibbles(width)
        stream.extend(ecc_encode(wn))
    else:
        # BIT_LENGTH 16-bit (max 65535 nibbles ≈ 32767 bytes) × 2
        ln = encode_uint16_to_nibbles(bit_length)
        stream.extend(ecc_encode(ln))

    return stream

def build_end() -> list[int]:
    """END pattern × 2."""
    stream: list[int] = []
    stream.extend(END_NIBBLES)
    stream.extend(END_NIBBLES)
    return stream

# ─── Payload Encoders ─────────────────────────────────────────────────────────

def encode_text_payload(text: str) -> tuple[list[int], int]:
    """Encode text as ASCII bytes → nibbles. Returns (nibbles, bit_length)."""
    data = text.encode("ascii", errors="replace")
    nibbles = bytes_to_nibbles(data)
    return nibbles, len(data) * 8

def encode_binary_payload(data: bytes) -> tuple[list[int], int]:
    """Encode raw bytes → nibbles. Returns (nibbles, bit_length)."""
    nibbles = bytes_to_nibbles(data)
    return nibbles, len(data) * 8

def encode_image_payload(path: str, color_mode: ColorMode
                         ) -> tuple[list[int], int, int]:
    """
    Encode an image file. Returns (nibbles, height, width).
    """
    img = Image.open(path)

    if color_mode == ColorMode.BW1BIT:
        img = img.convert("1")
        arr = np.array(img, dtype=np.uint8)
        h, w = arr.shape
        # Pack 4 pixels per nibble (MSB = first pixel)
        flat = arr.flatten()
        # Pad to multiple of 4
        pad = (4 - len(flat) % 4) % 4
        flat = np.pad(flat, (0, pad))
        nibbles = []
        for i in range(0, len(flat), 4):
            n = (int(flat[i])<<3)|(int(flat[i+1])<<2)|(int(flat[i+2])<<1)|int(flat[i+3])
            nibbles.append(n)
        return nibbles, h, w

    elif color_mode == ColorMode.COLOR4:
        img = img.convert("P")
        arr = np.array(img, dtype=np.uint8) & 0xF   # keep low 4 bits
        h, w = arr.shape
        flat = arr.flatten()
        return flat.tolist(), h, w

    elif color_mode == ColorMode.COLOR8:
        img = img.convert("L")   # 8-bit greyscale
        arr = np.array(img, dtype=np.uint8)
        h, w = arr.shape
        flat = arr.flatten()
        nibbles = bytes_to_nibbles(bytes(flat.tolist()))
        return nibbles, h, w

    elif color_mode == ColorMode.COLOR10:
        # Encode as 8-bit (trimmed from 10-bit concept, packed as 2 nibbles/pixel)
        img = img.convert("L")
        arr = np.array(img, dtype=np.uint8)
        h, w = arr.shape
        flat = arr.flatten()
        nibbles = bytes_to_nibbles(bytes(flat.tolist()))
        return nibbles, h, w

    else:
        raise ValueError(f"Unknown colour mode: {color_mode}")

def encode_audio_payload(path: str) -> tuple[list[int], int]:
    """
    Load an audio file, convert to 8-bit unsigned PCM mono, encode as nibbles.
    Returns (nibbles, bit_length).
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)                        # mix to mono
    # Resample naively to 8000Hz for compact transmission
    target_sr = 8000
    ratio = target_sr / sr
    n_out = int(len(mono) * ratio)
    indices = np.linspace(0, len(mono) - 1, n_out).astype(int)
    resampled = mono[indices]
    # Convert to u8
    u8 = ((resampled * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    nibbles = bytes_to_nibbles(bytes(u8.tolist()))
    return nibbles, len(u8) * 8

# ─── Packet Assembler ─────────────────────────────────────────────────────────

def assemble_packet(header_nibbles: list[int], payload_nibbles: list[int],
                    end_nibbles: list[int]) -> np.ndarray:
    """Render the full packet to audio samples."""
    all_nibbles = header_nibbles + payload_nibbles + end_nibbles
    return render_nibbles_to_audio(all_nibbles)

def add_inter_packet_gap(audio: np.ndarray, gap_ms: int = 100) -> np.ndarray:
    """Add a silence gap after a packet."""
    gap = silence(int(SAMPLE_RATE * gap_ms / 1000))
    return np.concatenate([audio, gap])

# ─── Print Summary ────────────────────────────────────────────────────────────

def print_packet_summary(pkt_type: PacketType, n_payload_nibbles: int,
                          height: int = 0, width: int = 0,
                          color_mode: ColorMode = ColorMode.BW1BIT,
                          bit_length: int = 0):
    tones_total = (len(START_NIBBLES)*2 + len(TYPE_NIBBLES[pkt_type])*2 +
                   n_payload_nibbles + len(END_NIBBLES)*2)
    duration_ms = tones_total * TONE_DURATION_MS
    print("─" * 50)
    print(f"  Packet type   : {PacketType(pkt_type).name}")
    if pkt_type in (PacketType.IMAGE, PacketType.IMAGE_PARTIAL):
        print(f"  Colour mode   : {COLOR_MODE_NAMES[color_mode]}")
        print(f"  Dimensions    : {width} × {height} px")
    else:
        print(f"  Bit length    : {bit_length} bits")
    print(f"  Payload tones : {n_payload_nibbles}")
    print(f"  Total tones   : {tones_total}")
    print(f"  Air time      : {duration_ms}ms  ({duration_ms/1000:.2f}s)")
    print("─" * 50)

# ─── CLI ──────────────────────────────────────────────────────────────────────

def cmd_text(args):
    text = args.text
    print(f"\n[HFPKT Encoder] TEXT packet")
    print(f"  Content       : {text[:60]}{'…' if len(text)>60 else ''}")
    nibbles, bit_len = encode_text_payload(text)
    header = build_start(PacketType.TEXT, bit_length=bit_len)
    end    = build_end()
    print_packet_summary(PacketType.TEXT, len(nibbles), bit_length=bit_len)
    audio  = assemble_packet(header, nibbles, end)
    audio  = add_inter_packet_gap(audio)
    sf.write(args.output, audio, SAMPLE_RATE)
    print(f"  Saved to      : {args.output}\n")

def cmd_callsign(args):
    call = args.callsign
    print(f"\n[HFPKT Encoder] CALLSIGN packet")
    text = f"CALLSIGN:{call}"
    nibbles, bit_len = encode_text_payload(text)
    header = build_start(PacketType.CALLSIGN, bit_length=bit_len)
    end    = build_end()
    print_packet_summary(PacketType.CALLSIGN, len(nibbles), bit_length=bit_len)
    audio  = assemble_packet(header, nibbles, end)
    audio  = add_inter_packet_gap(audio)
    sf.write(args.output, audio, SAMPLE_RATE)
    print(f"  Saved to      : {args.output}\n")

def cmd_image(args):
    color_map = {"bw": ColorMode.BW1BIT, "4": ColorMode.COLOR4,
                 "8": ColorMode.COLOR8,  "10": ColorMode.COLOR10}
    color_mode = color_map.get(args.color, ColorMode.BW1BIT)
    print(f"\n[HFPKT Encoder] IMAGE packet")
    print(f"  Source file   : {args.file}")

    # Optional split into chunks
    nibbles, h, w = encode_image_payload(args.file, color_mode)
    chunk_size = args.chunk  # nibbles per chunk (0 = no split)

    if chunk_size and len(nibbles) > chunk_size:
        # Split into multiple IMAGE_PARTIAL packets
        chunks = [nibbles[i:i+chunk_size] for i in range(0, len(nibbles), chunk_size)]
        print(f"  Split into    : {len(chunks)} partial packets ({chunk_size} nibbles each)")
        all_audio = []
        for idx, chunk in enumerate(chunks):
            ptype = PacketType.IMAGE if idx == 0 else PacketType.IMAGE_PARTIAL
            header = build_start(ptype, height=h, width=w, color_mode=color_mode)
            end    = build_end()
            pkt_audio = assemble_packet(header, chunk, end)
            pkt_audio  = add_inter_packet_gap(pkt_audio, gap_ms=50)
            all_audio.append(pkt_audio)
        audio = np.concatenate(all_audio)
    else:
        header = build_start(PacketType.IMAGE, height=h, width=w, color_mode=color_mode)
        end    = build_end()
        print_packet_summary(PacketType.IMAGE, len(nibbles), height=h, width=w,
                             color_mode=color_mode)
        audio  = assemble_packet(header, nibbles, end)
        audio  = add_inter_packet_gap(audio)

    sf.write(args.output, audio.astype(np.float32), SAMPLE_RATE)
    print(f"  Saved to      : {args.output}\n")

def cmd_audio(args):
    print(f"\n[HFPKT Encoder] AUDIO packet")
    print(f"  Source file   : {args.file}")
    nibbles, bit_len = encode_audio_payload(args.file)
    chunk_size = args.chunk or 0

    if chunk_size and len(nibbles) > chunk_size:
        chunks = [nibbles[i:i+chunk_size] for i in range(0, len(nibbles), chunk_size)]
        print(f"  Split into    : {len(chunks)} audio packets")
        all_audio = []
        for chunk in chunks:
            bit_len_chunk = len(chunk) // 2 * 8
            header = build_start(PacketType.AUDIO, bit_length=bit_len_chunk)
            end    = build_end()
            pkt    = assemble_packet(header, chunk, end)
            all_audio.append(add_inter_packet_gap(pkt, 30))
        audio = np.concatenate(all_audio)
    else:
        header = build_start(PacketType.AUDIO, bit_length=bit_len)
        end    = build_end()
        print_packet_summary(PacketType.AUDIO, len(nibbles), bit_length=bit_len)
        audio  = assemble_packet(header, nibbles, end)
        audio  = add_inter_packet_gap(audio)

    sf.write(args.output, audio.astype(np.float32), SAMPLE_RATE)
    print(f"  Saved to      : {args.output}\n")

def cmd_binary(args):
    print(f"\n[HFPKT Encoder] BINARY packet")
    print(f"  Source file   : {args.file}")
    with open(args.file, "rb") as f:
        data = f.read()
    nibbles, bit_len = encode_binary_payload(data)
    chunk_size = args.chunk or 0

    if chunk_size and len(nibbles) > chunk_size:
        chunks = [nibbles[i:i+chunk_size] for i in range(0, len(nibbles), chunk_size)]
        print(f"  Split into    : {len(chunks)} binary packets")
        all_audio = []
        for chunk in chunks:
            bl = len(chunk) // 2 * 8
            header = build_start(PacketType.BINARY, bit_length=bl)
            end    = build_end()
            pkt    = assemble_packet(header, chunk, end)
            all_audio.append(add_inter_packet_gap(pkt, 30))
        audio = np.concatenate(all_audio)
    else:
        header = build_start(PacketType.BINARY, bit_length=bit_len)
        end    = build_end()
        print_packet_summary(PacketType.BINARY, len(nibbles), bit_length=bit_len)
        audio  = assemble_packet(header, nibbles, end)
        audio  = add_inter_packet_gap(audio)

    sf.write(args.output, audio.astype(np.float32), SAMPLE_RATE)
    print(f"  Saved to      : {args.output}\n")

def main():
    parser = argparse.ArgumentParser(
        prog="hfpkt_encoder",
        description="HFPKT Ham Radio Frequency Packet Encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s text "Hello 73 de W1ABC" -o hello.wav
  %(prog)s callsign W1ABC -o id.wav
  %(prog)s image photo.png --color bw -o photo.wav
  %(prog)s image photo.png --color 4  --chunk 512 -o photo.wav
  %(prog)s audio voice.wav --chunk 1000 -o voice_pkt.wav
  %(prog)s binary file.bin --chunk 2048 -o data.wav

Colour modes (--color):
  bw  = 1-bit B&W  (4 pixels per tone)   [default]
  4   = 4-bit colour (1 pixel per tone)
  8   = 8-bit greyscale (2 tones per pixel)
  10  = 10-bit colour (packed 12-bit, 2 tones per pixel)

Tone Map (18 tones, 700-2400 Hz, 100 Hz steps, 5ms each):
  Tones 1-16  : data payload (nibbles 0x0-0xF)
  Tones 17-18 : control/pattern use only
        """)

    sub = parser.add_subparsers(dest="mode", required=True)

    # text
    p_text = sub.add_parser("text", help="Encode a text string")
    p_text.add_argument("text", help="Text string to encode")
    p_text.add_argument("-o", "--output", default="out_text.wav", help="Output WAV file")

    # callsign
    p_call = sub.add_parser("callsign", help="Encode a callsign / session header")
    p_call.add_argument("callsign", help="Your ham callsign, e.g. W1ABC")
    p_call.add_argument("-o", "--output", default="out_callsign.wav", help="Output WAV file")

    # image
    p_img = sub.add_parser("image", help="Encode an image file")
    p_img.add_argument("file", help="Image file (PNG, JPEG, BMP, …)")
    p_img.add_argument("--color", choices=["bw","4","8","10"], default="bw",
                       help="Colour mode (default: bw)")
    p_img.add_argument("--chunk", type=int, default=0,
                       help="Split into partial packets of N nibbles (0=no split)")
    p_img.add_argument("-o", "--output", default="out_image.wav", help="Output WAV file")

    # audio
    p_aud = sub.add_parser("audio", help="Encode an audio file")
    p_aud.add_argument("file", help="Audio file (WAV, FLAC, OGG, …)")
    p_aud.add_argument("--chunk", type=int, default=0,
                       help="Split into packets of N nibbles (0=no split)")
    p_aud.add_argument("-o", "--output", default="out_audio.wav", help="Output WAV file")

    # binary
    p_bin = sub.add_parser("binary", help="Encode a binary file")
    p_bin.add_argument("file", help="Binary file path")
    p_bin.add_argument("--chunk", type=int, default=0,
                       help="Split into packets of N nibbles (0=no split)")
    p_bin.add_argument("-o", "--output", default="out_binary.wav", help="Output WAV file")

    args = parser.parse_args()

    dispatch = {
        "text":     cmd_text,
        "callsign": cmd_callsign,
        "image":    cmd_image,
        "audio":    cmd_audio,
        "binary":   cmd_binary,
    }
    dispatch[args.mode](args)

if __name__ == "__main__":
    main()
