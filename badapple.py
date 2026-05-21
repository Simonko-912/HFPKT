import json
import os
import subprocess

video_file = "badapple.mp4"
output_dir = "frames"
fps = 1

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Extract frames using ffmpeg at 1 FPS, 24x24, grayscale, B&W
cmd = [
    "ffmpeg",
    "-i", video_file,
    "-vf", "fps=1,scale=24:24,format=gray",
    "-vframes", "82",  # Bad Apple is 82 seconds, so 82 frames at 1 FPS
    os.path.join(output_dir, "frame_%03d.png")
]

print(f"Extracting frames from {video_file}...")
subprocess.run(cmd)

# Get all frame files and sort them
frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])

# Build JSON stream
stream = []
stream.append({"type": "callsign", "callsign": "EXAMPLE"})

for frame in frames:
    stream.append({"type": "image", "file": os.path.join(output_dir, frame), "color": "bw"})

stream.append({"type": "callsign", "callsign": "EXAMPLE"})

# Write to file
with open('bad_apple_stream.json', 'w') as f:
    json.dump(stream, f, indent=2)

print(f"Generated stream with {len(frames)} frames")
print(f"Stream saved to bad_apple_stream.json")
