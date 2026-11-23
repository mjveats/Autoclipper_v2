# Auto Clipper v2.0 — AI Viral Video Clipping Engine

[![Owner](https://img.shields.io/badge/Developed%20By-MJ%20Mark%20Jonel-7c5cff.svg)](https://www.facebook.com/search/top/?q=MJ%20Mark%20Jonel)
[![License](https://img.shields.io/badge/License-Free%20Community%20Edition-green.svg)](#-anti-resale-notice--ownership)
[![Status](https://img.shields.io/badge/Resale-STRICTLY%20PROHIBITED-red.svg)](#-anti-resale-notice--ownership)

> **Auto Clipper** is an open-source, local AI-powered video clipping suite that transforms long YouTube videos, Twitch streams, and podcast VODs into high-CTR, viral vertical short-form content (TikTok, Instagram Reels, YouTube Shorts).

---

## 🛡️ Anti-Resale Notice & Ownership
* **Software Owner & Lead Developer**: **MJ Mark Jonel**
* **Facebook Profile**: [MJ Mark Jonel on Facebook](https://www.facebook.com/search/top/?q=MJ%20Mark%20Jonel)
* **Distribution Policy**: **FREE COMMUNITY EDITION — NOT FOR SALE.**
  * If someone charged you money for this software, you have been scammed. Report the seller directly to MJ Mark Jonel on Facebook.
  * Rebranding, selling, or charging for access to this tool without written authorization from MJ Mark Jonel is strictly prohibited.

---

## 🚀 Key Features

* **⚡ Instant First-Clip Live Delivery**: Clip #1 is delivered and playable in the built-in studio player the moment it finishes—no waiting for all clips to render!
* **🎬 In-App Video Player Studio**: Watch, scrub, replay, and preview generated clips with synchronized audio, waveforms, and 1-click title/description copy.
* **✏️ Interactive Results Studio (Editable Output Before MP4 Export)**:
  * Adjust the **Hook Banner Text** directly in an editable field.
  * Switch between any of the **12 Viral Fonts** in real time.
  * Switch **Caption Animation Styles** (Beast, Hormozi, Abdaal, Neon, Podcast, Karaoke).
  * Customize **Brand Accent Color** and **Font Size Scale** (Compact, Medium, Large, Huge).
  * Reposition subtitles (**Bottom Safe Zone**, **Middle Cut Seam**, or **Top-Third**).
  * Click `[ 🔄 Apply / Update Preview ]` to refresh the video draft instantly.
  * Click `[ 🚀 Export Final MP4 ]` to render high-definition final production MP4s on demand.
* **🔤 12 Viral Fonts Included**:
  * Montserrat, Poppins, Roboto, Bebas Neue, Proxima Nova, SF Pro Display, Helvetica, TikTok Sans, Playfair Display, Garamond, Luckiest Guy, and Fredoka.
* **🌊 Smooth Open-Source Kinetic Subtitles (ASS Transforms)**:
  * Open-source ASS easing curves with acceleration physics (`\t(0,90,0.7,\fscx118...)` and `\t(90,200,1.3,\fscx100...)`) for elastic spring bounce.
* **💻 Hardware Aware & Low-End Friendly**:
  * **Laptop / iGPU (0GB VRAM • 4-8GB RAM)**: Ultra-lightweight `qwen2.5:1.5b` + Whisper `small`.
  * **Fast / Budget GPU (4GB VRAM • 8GB RAM)**: Fast `qwen2.5:3b`.
  * **Recommended / Balanced (6-8GB VRAM • 16GB RAM)**: Recommended `qwen2.5:7b`.
  * **Heavy / High-End (12+GB VRAM • 32GB RAM)**: High-end `qwen2.5:14b`.
* **👤 Smart Facecam Tracking (YuNet)**:
  * Persistent facecam tracking with split-screen lock and streamer corner webcam fallback.
* **🤖 Automatic Local AI Setup**:
  * Automatically detects, downloads, and runs Ollama and AI models with zero manual technical setup.

---

## 💻 Quick Start for Testers

### Option A: Standalone Executable (.exe) — Recommended
1. Download `AutoClipper_v2.0_Windows.zip` from the [Releases](../../releases) tab.
2. Extract the ZIP to your computer.
3. Double-click `AutoClipper.exe` to launch!
   *(Optional: You can double-click `Install_Ollama_Automatic.bat` to pre-install the local AI engine with 1 click).*

### Option B: Run from Source Code (For Developers)
1. Clone this repository:
   ```bash
   git clone https://github.com/mjveats/AutoClipper.git
   cd AutoClipper
   ```
2. Double-click `INSTALL_AND_RUN.bat` to install the environment and launch.

---

## 🔨 Compiling Standalone Executable (.exe)
To compile a fresh standalone `.exe` at any time, run:
```bash
build_exe.bat
```
The compiled output is automatically packaged into `1. COMPILED (Share With Testers)/AutoClipper/` and zipped to `AutoClipper_v2.0_Windows.zip`.

---

© 2026 MJ Mark Jonel. All Rights Reserved.
