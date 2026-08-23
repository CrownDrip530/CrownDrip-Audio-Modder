# CrownDrip Audio Modder 👑🔥

A Windows mic-modding soundboard app: routes your microphone through a virtual
audio cable so apps like Discord, OBS, and games see it as your mic — while
you apply live effects (Mic Gain, **Deep Fried Mode**) and blast MP3s into the
mix from a built-in soundboard. Gold & black themed GUI. All your settings and
uploaded sounds are remembered between sessions.

## Features

- 🎙️ **Mic passthrough** to a virtual input device (via VB-Audio Virtual Cable)
- 🔊 **Mic gain boost/cut** (-20 dB to +20 dB)
- 🍟 **Deep Fried Mode** — one-click toggle for a crunchy, distorted, bitcrushed,
  mid-boosted "deep fried meme audio" mic effect
- 🎵 **Soundboard** — upload MP3s, give them custom names, play them mixed into
  your mic output live
- 💾 **Persistent memory** — your device selection, gain, Deep Fry setting, and
  entire soundboard library (with custom names) are saved and reloaded every
  time you open the app, stored in `%APPDATA%\CrownDripAudioModder\`
- 🧩 **Extensible effects chain** — built so new effects (echo, robot/pitch
  shift, reverb, etc.) can be dropped in later with minimal code

## Requirements

- **Windows 10/11**
- **Python 3.10+**
- **[VB-Audio Virtual Cable](https://vb-audio.com/Cable/)** installed
  (free) — this creates the "CABLE Input" / "CABLE Output" devices that let
  CrownDrip's processed audio show up as a selectable microphone elsewhere.
- **ffmpeg** on your PATH (required by `pydub` to decode MP3 files) —
  download from [ffmpeg.org](https://ffmpeg.org/download.html) or
  `winget install ffmpeg`

## Setup

```bash
git clone https://github.com/yourusername/CrownDrip-Audio-Modder.git
cd CrownDrip-Audio-Modder
pip install -r requirements.txt
