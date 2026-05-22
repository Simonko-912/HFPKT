<img width="800" height="450" alt="HFPKT's name and a waterfall image of it in fm" src="https://github.com/user-attachments/assets/74ff8128-e28d-4cc0-b044-22110e82f74d" /> <br>
<img width="1774" height="129" alt="bad apple in hfpkt on spectogram" src="https://github.com/user-attachments/assets/a3a24829-cb7c-4b05-b4c4-7c8fa74459ab" /> <br> 
<br>
<img width="1777" height="463" alt="image" src="https://github.com/user-attachments/assets/6149e0a5-b647-41c2-a476-f97c5fa39ce3" />


# Architecture explained

## Basic info

HFPKT is a new digital mode simular to RTTY or FSK, It uses 18 diffrent frequencies, from 700hz to 2400hz, each 100hz away from each other.<br>
16 are used for data, and the other 2 for headers, HFPKT uses a special combination of frequencies (tones) to represent diffrent bits. <br>
Data uses 16 frequencies, so you can transmit 4 bits at the same time. The other bits (2) are used to identify the metadata from data, thats why its longer. <br>
HFPKT is made for HF, but should work on VHF, and other frequencies, Since its audio based, it should work on every mode (am, fm, lsb, usb etc. expect modes like cw.)<br>
HFPKT transmits at 800bps, using a mix of tones, that mix is played for 5 ms, and packets are made from multiple combinations of these.<br>

## Packets

### Anatomy of a packet


Packets are made by transmitting sections each taking 5ms. (aka, a single mix of tones is on for 5ms) <br>
How a packet is made: <br>
1. Start section - Transmits a specific pattern 2 times for ECC. This defines the start of the packet.
2. Type section - Transmits a specific pattern for each type (Audio, Image, Partial image, Text, Binary), 2 times for ECC.
3. (Image only) Height - Transmits the height of the image as a 16 bit number, (8 times transmits, 4 without ECC).
4. (Image only) Width - Simular concept like height, just with width.
5. (Text, bin etc.) Binary length - The size of the content in bits.
6. The content - The data transmited in 4 bit chunks, this does NOT have ECC
7. End section - Transmits a specific patern 2 times for ECC. This defines the end of the packet


### How packets are used

Its recommended to transmit a packet multiple times if its important. You dont loose a lot of bps. <br>
If you transmit a packet 2 times for ECC, it will still be around 400bps. Transmiting 4 times would be 200bps. <br>


#### Recommended way to send multiple packets

First transmit your callsign using morse, or voice, this mode isnt officialy recognized. <br>
Now you can transmit some text, images, etc, then its recommended to end by transmitting your callsign using morse or voice again <br>
Theres also a app to help you make this, the stream app. <br>


# Software

## Prepare
First install requirements using pip or apt. (Or use a venv and first clone the repo) <br>
```
pip install numpy scipy pillow soundfile
```
or using apt <br>
```
sudo apt install python3-numpy python3-scipy python3-pillow python3-soundfile
```

Now clone the repository <br>
```
git clone https://github.com/Simonko-912/HFPKT
cd HFPKT
```
Now you can use the encoder, decoder and etc!<br>

## Examples

### Without rx / tx

For this example we will encode and decode "Hello world!". <br>
```
python3 hfpkt_encoder.py text "Hello world!"  -o helloworld.wav
```
You will now see some info like packet type, length in bits, total tones, and length it takes to transmit this.<br>
This is how to decode this now. <br>
```
python3 hfpkt_decoder.py helloworld.wav --out-dir ./decoded
```
Now the packet is saved here ./decoded/helloworld_pkt000_text.txt<br>
You can see its correctly decoded there!<br>

### With rx / tx

First prepare your transmitter and reciever, tune them on the same frequencies, now run:<br>
```
python3 hfpkt_encoder.py text "Hello world!"  -o helloworld.wav
```
Then record on your reciever and send your callsign in morse or voice, and then you can play the wave file.<br>
Now get the wave file / mp3 from the reciever and run: <br>
```
python3 hfpkt_decoder.py example.wav --spectrum --out-dir ./decoded --verbose --threshold 0.01
```
Replace example.wav with the file you recieved, and the value for threshold (Lower value = more sensitive), a ammount like 0.01 works most of the time.<br>
We run with --spectrum to see what types of frequencies we reciever and --verbose to see more info about the decoding.<br>
If it doesnt work, tune the threshold, and if that doesnt help, try removing noise using programs, The decoder is still in development.<br>

### More example commands
Encoder:<br>
```
python hfpkt_encoder.py text "Hello 73 de W1ABC"  -o out.wav
python hfpkt_encoder.py callsign W1ABC             -o id.wav
python hfpkt_encoder.py image photo.png --color bw -o img.wav
python hfpkt_encoder.py image photo.png --color 4  --chunk 512 -o img_split.wav
python hfpkt_encoder.py audio voice.wav --chunk 1000 -o voice.wav
python hfpkt_encoder.py binary data.bin            -o data.wav
```

Decoder:<br>
```
python hfpkt_decoder.py signal.wav
python hfpkt_decoder.py signal.wav --out-dir ./decoded --verbose
python hfpkt_decoder.py signal.wav --spectrum        # text-art FFT display
python hfpkt_decoder.py signal.wav --threshold 0.03  # lower for weak signals
```


<br>

## Image examples
Note that --color stands for color depth, chosing bw uses a single bit for a pixel.<br>
Scottie 1 is around 9 seconds slower when comparing scottie 1 and hfpkt (HFPKT 102.48s, Scottie 1 111s)<br>
<img width="320" height="256" alt="img_pkt000_image" src="https://github.com/user-attachments/assets/9672ce1a-4869-4b01-b4be-087ab010c657" /> <img width="320" height="256" alt="monalisasstv" src="https://github.com/user-attachments/assets/7a607c5b-583c-4c5e-9ce9-1940f57e45d3" /> <br>
Scottie 1 has simulated noise, as you can see, sstv is analog, so its more blurry, but HFPKT is digital, even if it is only bw, its sharp. This was made with `--color bw`<br>
It took HFPKT 26.16s to decode this. <br>

<br>

<img width="320" height="256" alt="img_pkt000_image" src="https://github.com/user-attachments/assets/fb3e5ae6-f038-4599-9b32-a781179827a0" /> <br>
The wave file for this was 817.65s long, it took 154.18s to decode, but its really high quality. This was made with `--color 8`<br>
Note: HFPKT was not tested with noise since the current decoder is not the best. But it was tested with text over fm, what was sometimes intact and sometimes corrupted.<br>

<br>

## HFPKT Stream
You can use hfpkt stream to make a stream of packets into a wave file. <br>
Commands: <br>
```
# Generate an example playlist
python hfpkt_stream.py --example-json > my_stream.json
python hfpkt_stream.py --example-text > my_stream.txt

# Build the stream
python hfpkt_stream.py my_stream.json -o stream.wav
python hfpkt_stream.py my_stream.txt  -o stream.wav --gap 300 --verbose
```
Gap sets the gap between packets, i recommend something like 10ms or 5ms if you want it as short as possible.<br>
The examples built in are diffrent than the stream_example.json, you can use any you want.<br>
