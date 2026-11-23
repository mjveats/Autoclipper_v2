"""Auto Clipper v3 - fully local clipping pipeline (no AI subscriptions).

yt-dlp -> full transcript (Whisper) -> story judging + editor review (Ollama) -> dead-air cuts,
punch-in zooms, 3-second text hook, captions, profanity bleep, background music (ffmpeg)
-> SEO title/description per clip -> results page ranked by score.
"""
import argparse
import colorsys
import functools
import gc
import json
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
try:
    os.chdir(ROOT)
except Exception:
    pass

def get_bundle_dir() -> Path:
    """Return the base directory for bundled PyInstaller resources (_MEIPASS, _internal, or ROOT)."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS).resolve()
    internal = ROOT / "_internal"
    if internal.exists():
        return internal
    return ROOT

def resolve_resource(*subpaths) -> Path:
    """Find a resource file/folder across ROOT, PyInstaller bundle, and script locations."""
    rel = Path(*subpaths)
    candidates = [
        ROOT / rel,
        get_bundle_dir() / rel,
        ROOT / "_internal" / rel,
        Path(__file__).resolve().parent / rel,
    ]
    for c in candidates:
        if c.exists():
            return c
    return ROOT / rel

def get_tool_path(tool_name: str) -> str:
    """Resolve binary (ffmpeg, ffprobe, ffplay, yt-dlp) to bundled executable or system fallback."""
    raw = str(tool_name).strip()
    exe_name = raw if raw.lower().endswith(".exe") else f"{raw}.exe"
    candidates = [
        ROOT / "tools" / "ffmpeg" / "bin" / exe_name,
        get_bundle_dir() / "tools" / "ffmpeg" / "bin" / exe_name,
        ROOT / "_internal" / "tools" / "ffmpeg" / "bin" / exe_name,
        ROOT / "tools" / "bin" / exe_name,
        ROOT / "tools" / exe_name,
        Path(__file__).resolve().parent / "tools" / "ffmpeg" / "bin" / exe_name,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    found = shutil.which(raw) or shutil.which(exe_name)
    if found:
        return found
    return raw

def prepare_env():
    """Make bundled ffmpeg / ffprobe / ffplay / yt-dlp / CUDA DLLs visible to this process and subprocesses."""
    extra = []
    for p in [
        ROOT / "tools" / "ffmpeg" / "bin",
        get_bundle_dir() / "tools" / "ffmpeg" / "bin",
        ROOT / "_internal" / "tools" / "ffmpeg" / "bin",
        ROOT / "tools" / "bin",
        ROOT / "tools",
        Path(sys.executable).resolve().parent,
        get_bundle_dir(),
    ]:
        if p.is_dir() and str(p) not in extra:
            extra.append(str(p))
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(p))
                except Exception:
                    pass

    for base in [Path(sys.prefix), get_bundle_dir(), ROOT, ROOT / "_internal"]:
        nv = base / "Lib" / "site-packages" / "nvidia"
        if nv.exists():
            for b in nv.glob("*/bin"):
                if b.is_dir() and str(b) not in extra:
                    extra.append(str(b))
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(str(b))
                        except Exception:
                            pass
        if hasattr(os, "add_dll_directory") and base.is_dir():
            try:
                os.add_dll_directory(str(base))
            except Exception:
                pass

    os.environ["PATH"] = os.pathsep.join(extra + [os.environ.get("PATH", "")])

# Run immediately at startup so all imports and subprocesses find bundled tools
prepare_env()

OUT_DIR = ROOT / "output"
WORK_DIR = ROOT / "work"
MODEL_DIR = resolve_resource("models")
MUSIC_DIR = resolve_resource("music")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_PATH = resolve_resource("models", "yunet.onnx")
# Models labeled with explicit hardware & VRAM/RAM specs for low-end GPUs, APUs & laptops:
MODELS = {
    "Laptop / iGPU (No GPU • 0GB VRAM • 4-8GB RAM) - qwen2.5:1.5b": "qwen2.5:1.5b",
    "Fast / Budget GPU (4GB VRAM • 8GB RAM) - qwen2.5:3b": "qwen2.5:3b",
    "Recommended / Balanced (6-8GB VRAM • 16GB RAM) - qwen2.5:7b": "qwen2.5:7b",
    "Heavy / High-End (12+GB VRAM • 32GB RAM) - qwen2.5:14b": "qwen2.5:14b",
}
MODEL_CHOICES = list(MODELS.keys())
DEFAULT_LABEL = "Laptop / iGPU (No GPU • 0GB VRAM • 4-8GB RAM) - qwen2.5:1.5b"
DEFAULT_MODEL = "qwen2.5:1.5b"
FAST_MODEL = "qwen2.5:1.5b"
FALLBACK_MODEL = "qwen2.5:1.5b"


def parse_model_name(name):
    if name in MODELS:
        return MODELS[name]
    for tag in ("qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b"):
        if tag in name:
            return tag
    return name.strip()


# 12 Viral Fonts requested for TikTok/Shorts:
FONTS = [
    "Montserrat",
    "Poppins",
    "Roboto",
    "Bebas Neue",
    "Proxima Nova",
    "SF Pro Display",
    "Helvetica",
    "TikTok Sans",
    "Playfair Display",
    "Garamond",
    "Luckiest Guy",
    "Fredoka",
]
DEFAULT_FONT = "Montserrat"
FONTS_DIR = resolve_resource("models", "fonts")
RESOURCES_FONTS_DIR = resolve_resource("resources", "fonts")


def resolve_font_path(font_name):
    """Finds a matching font path on disk or returns a safe Windows system fallback."""
    try:
        FONTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    for fdir in (FONTS_DIR, RESOURCES_FONTS_DIR):
        if not fdir.exists():
            continue
        for ext in (".ttf", ".otf"):
            p = fdir / f"{font_name}{ext}"
            if p.exists():
                return str(p)
            p = fdir / f"{font_name.replace(' ', '')}{ext}"
            if p.exists():
                return str(p)
            p = fdir / f"{font_name.lower()}{ext}"
            if p.exists():
                return str(p)

    win_fonts = Path("C:/Windows/Fonts")
    local_fonts = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts"
    for sdir in (win_fonts, local_fonts):
        if not sdir.exists():
            continue
        for ext in (".ttf", ".otf", ".TTF"):
            for cand in [sdir / f"{font_name}{ext}",
                         sdir / f"{font_name.replace(' ', '')}{ext}",
                         sdir / f"{font_name.lower()}{ext}",
                         sdir / f"{font_name.lower().replace(' ', '')}{ext}"]:
                if cand.exists():
                    return str(cand)

    name_l = (font_name or "").lower()
    if any(k in name_l for k in ("garamond", "playfair")):
        for fb in ("GARABD.TTF", "GARA.TTF", "georgiab.ttf", "georgia.ttf"):
            if (win_fonts / fb).exists():
                return str(win_fonts / fb)
    if any(k in name_l for k in ("bebas", "luckiest")):
        for fb in ("impact.ttf", "arialbd.ttf"):
            if (win_fonts / fb).exists():
                return str(win_fonts / fb)

    for fb in ("segoeuib.ttf", "arialbd.ttf", "impact.ttf"):
        if (win_fonts / fb).exists():
            return str(win_fonts / fb)
    return None


CAP_POSITIONS = {
    "Bottom Safe Zone (~28%)": "bottom",
    "Middle Cut Seam (~46%)": "middle",
    "Top-Third (~68%)": "top",
}
CAP_POS_CHOICES = list(CAP_POSITIONS.keys())
DEFAULT_CAP_POS_LABEL = "Bottom Safe Zone (~28%)"


def parse_cap_pos(p):
    return CAP_POSITIONS.get(p, "bottom")


FONT_SCALES = {
    "Medium (100% - Default)": 1.0,
    "Compact (85%)": 0.85,
    "Large (115%)": 1.15,
    "Viral Huge (130%)": 1.30,
}
FONT_SCALE_CHOICES = list(FONT_SCALES.keys())
DEFAULT_FONT_SCALE_LABEL = "Medium (100% - Default)"


def parse_font_scale(s):
    if isinstance(s, (int, float)):
        return float(s)
    return FONT_SCALES.get(s, 1.0)


# Aspect Ratios:
RATIOS = {
    "9:16 (Shorts / TikTok / Reels)": "9:16",
    "3:4 (Whop Cards / Feed)": "3:4",
    "1:1 (Square / Feed)": "1:1",
    "16:9 (Landscape / YouTube)": "16:9",
}
RATIO_CHOICES = list(RATIOS.keys())
DEFAULT_RATIO_LABEL = "9:16 (Shorts / TikTok / Reels)"


def parse_ratio(r):
    return RATIOS.get(r, r.split()[0] if r else "9:16")


# Layout options:
LAYOUTS = {
    "auto (smart auto-detect)": "auto",
    "fit (whole video in frame)": "fit",
    "split (facecam + game)": "split",
    "fill (crop to full screen)": "fill",
}
LAYOUT_CHOICES = list(LAYOUTS.keys())
DEFAULT_LAYOUT_LABEL = "auto (smart auto-detect)"


def parse_layout(l):
    return LAYOUTS.get(l, l.split()[0] if l else "auto")


# Facecam placement:
CAM_POSITIONS = {
    "Top (Facecam Top / Game Bottom)": "top",
    "Bottom (Game Top / Facecam Bottom)": "bottom",
    "PIP Top-Right (Overlay)": "pip-tr",
    "PIP Top-Left (Overlay)": "pip-tl",
    "PIP Bottom-Right (Overlay)": "pip-br",
    "PIP Bottom-Left (Overlay)": "pip-bl",
}
CAM_POS_CHOICES = list(CAM_POSITIONS.keys())
DEFAULT_CAM_POS_LABEL = "Top (Facecam Top / Game Bottom)"


def parse_cam_pos(c):
    return CAM_POSITIONS.get(c, "top")


# Live stream capture modes:
LIVE_MODES = {
    "Off (Full Video / VOD)": 0,
    "🔴 Live (Last 15 min buffer)": 15,
    "🔴 Live (Last 30 min buffer)": 30,
    "🔴 Live (Last 45 min buffer)": 45,
}
LIVE_CHOICES = list(LIVE_MODES.keys())
DEFAULT_LIVE_LABEL = "Off (Full Video / VOD)"


def parse_live_mins(val):
    if isinstance(val, int):
        return val
    return LIVE_MODES.get(val, 0)


# Clip count limits:
TOP_CLIPS = {
    "5 (Recommended - ~4 mins)": 5,
    "3 (Fastest - ~2 mins)": 3,
    "10": 10,
    "All detected clips": 999,
}
TOP_CLIPS_CHOICES = list(TOP_CLIPS.keys())
DEFAULT_TOP_CLIPS_LABEL = "5 (Recommended - ~4 mins)"


def parse_top_clips(val):
    if isinstance(val, int):
        return val
    return TOP_CLIPS.get(val, 5)


# Brand color palettes:
BRAND_COLORS = {
    "yellow": {
        "name": "High-Voltage Yellow (#FFDD00)",
        "hex": "#FFDD00",
        "ass": "&H0000DDFF&",
        "rgb": (255, 221, 0),
    },
    "neon_green": {
        "name": "Neon Green (#00FF66 - Beast)",
        "hex": "#00FF66",
        "ass": "&H0066FF00&",
        "rgb": (0, 255, 102),
    },
    "electric_cyan": {
        "name": "Electric Cyan (#00F0FF - Viral)",
        "hex": "#00F0FF",
        "ass": "&H00FFF000&",
        "rgb": (0, 240, 255),
    },
    "hot_pink": {
        "name": "Hot Pink (#FF2E93 - Pop)",
        "hex": "#FF2E93",
        "ass": "&H00932EFF&",
        "rgb": (255, 46, 147),
    },
    "purple": {
        "name": "Royal Purple (#8A2BE2 - Twitch)",
        "hex": "#8A2BE2",
        "ass": "&H00E22B8A&",
        "rgb": (138, 43, 226),
    },
    "orange": {
        "name": "Flame Orange (#FF6B00 - Fire)",
        "hex": "#FF6B00",
        "ass": "&H00006BFF&",
        "rgb": (255, 107, 0),
    },
    "pure_white": {
        "name": "Pure White (#FFFFFF - Minimal)",
        "hex": "#FFFFFF",
        "ass": "&H00FFFFFF&",
        "rgb": (255, 255, 255),
    },
}
BRAND_COLOR_CHOICES = [b["name"] for b in BRAND_COLORS.values()]
DEFAULT_BRAND_COLOR_LABEL = BRAND_COLORS["yellow"]["name"]


def parse_brand_color(val):
    if not val:
        return BRAND_COLORS["yellow"]
    for k, v in BRAND_COLORS.items():
        if val == k or val == v["name"] or val.lower() == v["hex"].lower():
            return v
    # Check if a custom hex was passed: e.g. #123456
    if isinstance(val, str) and val.startswith("#") and len(val) == 7:
        try:
            r = int(val[1:3], 16)
            g = int(val[3:5], 16)
            b = int(val[5:7], 16)
            ass_code = f"&H00{b:02X}{g:02X}{r:02X}&"
            return {"name": f"Custom ({val})", "hex": val, "ass": ass_code, "rgb": (r, g, b)}
        except Exception:
            pass
    return BRAND_COLORS["yellow"]


# --------------------------------------------------------------- OWNER & LICENSING
OWNER_NAME = "MJ Mark Jonel"
OWNER_FACEBOOK = "https://www.facebook.com/search/top/?q=MJ%20Mark%20Jonel"
# Remote Kill-Switch endpoint: point to any raw JSON URL (e.g. GitHub Gist or Pastebin)
# Example JSON: {"status": "active", "disabled": false, "notice": "Auto Clipper by MJ Mark Jonel"}
REMOTE_KILLSWITCH_URL = ""


def verify_remote_license(timeout=2.0):
    """Checks remote killswitch if REMOTE_KILLSWITCH_URL is configured."""
    if not REMOTE_KILLSWITCH_URL:
        return True, "Active (Community Edition)"
    try:
        import urllib.request
        req = urllib.request.Request(REMOTE_KILLSWITCH_URL, headers={"User-Agent": "AutoClipper-MJ"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("disabled") is True or data.get("status") == "revoked":
                msg = data.get("notice", f"This software has been disabled by owner {OWNER_NAME}.")
                return False, msg
    except Exception:
        pass
    return True, "Active"


# Caption styling options (Viral Open-Source ASS Templates):
CAPTION_STYLES = {
    "auto": "auto (smart niche preset)",
    "beast": "beast (mrbeast kinetic bounce - 1-2 words)",
    "hormozi": "hormozi (alex hormozi yellow pop - 2-3 words)",
    "abdaal": "abdaal (ali abdaal clean pill highlighter - 3-4 words)",
    "neon": "neon (cyberpunk electric glow - 2 words)",
    "podcast": "podcast (clean minimal box - 3-5 words)",
    "karaoke": "karaoke (kinetic wave bounce - 2 words)",
}
CAPTION_STYLE_CHOICES = list(CAPTION_STYLES.values())
DEFAULT_CAPTION_STYLE_LABEL = CAPTION_STYLES["auto"]


def parse_caption_style(val):
    if not val:
        return "auto"
    for k, v in CAPTION_STYLES.items():
        if val == k or val == v:
            return k
    return "auto"


EMOJI_DIR = resolve_resource("models", "emojis")
EMOJI_MAP = {
    "💀": "1f480", "😭": "1f62d", "😂": "1f602", "🤣": "1f923",
    "😱": "1f631", "🤯": "1f92f", "🔥": "1f525", "🚨": "1f6a8",
    "💸": "1f4b8", "🤡": "1f921", "🐐": "1f410", "🤐": "1f910",
    "🤫": "1f92b", "😈": "1f608", "👀": "1f440", "🎯": "1f3af",
    "💯": "1f4af", "😳": "1f633", "🥶": "1f976", "😡": "1f621",
    "🎮": "1f3ae", "💣": "1f4a3", "💰": "1f4b0", "👑": "1f451",
    "🤦": "1f926", "✨": "2728", "⚡": "26a1"
}


def is_emoji_char(ch):
    c = ord(ch)
    return (0x1F300 <= c <= 0x1FAFF or 0x2600 <= c <= 0x27BF or 0xFE00 <= c <= 0xFE0F
            or 0x1F600 <= c <= 0x1F64F or 0x1F680 <= c <= 0x1F6FF or 0x1F900 <= c <= 0x1F9FF)


def extract_emojis(text):
    """Extracts emojis and strips them from the text so no font fallback boxes appear."""
    if not text:
        return "", []
    clean_chars = []
    codes = []
    for ch in text:
        if ch in EMOJI_MAP:
            codes.append(EMOJI_MAP[ch])
        elif is_emoji_char(ch):
            hex_code = f"{ord(ch):x}"
            if (EMOJI_DIR / f"{hex_code}.png").exists():
                codes.append(hex_code)
            elif EMOJI_MAP.get(ch):
                codes.append(EMOJI_MAP[ch])
            else:
                codes.append("1f480")  # Default to skull if unknown
        else:
            clean_chars.append(ch)
    return "".join(clean_chars).strip(), codes


def find_emoji(text):
    _, codes = extract_emojis(text)
    return (None, codes[0]) if codes else (None, None)


OLLAMA_URL = "http://127.0.0.1:11434"
SR = 16000
NOWIN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
STATE = dict(model=DEFAULT_MODEL, cookies="")
FLOOR_PRE = 4.5          # window score needed to get an editor review
FLOOR = 5.0              # final score needed to be rendered (red = 5-6, green = 7.5+)
ZOOMS = [1.12, 1.0, 1.18, 1.0, 1.10, 1.0]

PROFILES = {
    "gaming":        dict(min=30, max=65, cam=True,  captions=True,  story=True,  scan="deep", gap=1.8, zoom=False,
                          cap_style="beast", brand="neon_green",
                          rec="Gaming: 9:16 Split | 30s+ Complete play with buildup, tension, and funny freakout",
                          hint="clutch plays with context, funny team banter, rage buildup and freakout, surprising fails"),
    "streaming":     dict(min=30, max=65, cam=True,  captions=True,  story=True,  scan="deep", gap=1.5, zoom=True,
                          cap_style="beast", brand="electric_cyan",
                          rec="Streaming: 9:16 Split or Fit | 30s+ Complete story to chat, drama, relatable punchlines",
                          hint="funny stories told to chat, strong reactions with buildup, hilarious debates, punchlines"),
    "finance_crypto":dict(min=30, max=70, cam=False, captions=True,  story=True,  scan="deep", gap=1.4, zoom=True,
                          cap_style="hormozi", brand="yellow",
                          rec="Whop / Finance: 9:16 Fit with Captions | 30s+ Actionable advice, trading calls, case studies",
                          hint="crypto signals, trading strategies, income reveal, business breakdowns, actionable money tips"),
    "motivation":    dict(min=30, max=65, cam=False, captions=True,  story=True,  scan="deep", gap=1.3, zoom=True,
                          cap_style="hormozi", brand="yellow",
                          rec="Motivation: 9:16 Fit or Fill | 30s+ Intense speeches, mindset shifts, discipline truth",
                          hint="powerful mindset shifts, discipline advice, raw truth, emotional wake-up calls, gym motivation"),
    "education_tech":dict(min=30, max=65, cam=False, captions=True,  story=True,  scan="deep", gap=1.4, zoom=True,
                          cap_style="hormozi", brand="electric_cyan",
                          rec="Tech / SaaS: 9:16 Fit (Whole screen visible) | 30s+ AI tools, websites, step-by-step hacks",
                          hint="insane AI tools, useful websites, secret shortcuts, tech hacks, step-by-step tutorials"),
    "podcast":       dict(min=35, max=75, cam=False, captions=True,  story=True,  scan="deep", gap=1.5, zoom=True,
                          cap_style="podcast", brand="pure_white",
                          rec="Podcasts: 9:16 Fit | 35s+ Complete argument, debate, storytelling with payoff",
                          hint="bold opinions, personal stories with a payoff, surprising facts, quotable lessons"),
    "interviews":    dict(min=35, max=75, cam=False, captions=True,  story=True,  scan="deep", gap=1.5, zoom=True,
                          cap_style="podcast", brand="pure_white",
                          rec="Interviews: 9:16 Fit | 35s+ Controversial revelations, celebrity stories, hot takes",
                          hint="revealing answers, controversy, emotional stories, quotable lines"),
    "irl":           dict(min=30, max=65, cam=False, captions=True,  story=True,  scan="deep", gap=1.5, zoom=True,
                          cap_style="beast", brand="hot_pink",
                          rec="IRL: 9:16 Fill or Fit | 30s+ Street encounter, funny conversation, public reactions",
                          hint="awkward or funny encounters, surprises, personal stories, strong reactions"),
    "entertainment": dict(min=30, max=65, cam=False, captions=True,  story=True,  scan="deep", gap=1.5, zoom=True,
                          cap_style="beast", brand="yellow",
                          rec="Comedy & Entertainment: 30s+ Setup, joke progression, and viral punchline",
                          hint="jokes with setup and punchline, twists, memorable lines, shocking beats"),
    "music":         dict(min=30, max=60, cam=False, captions=False, story=False, scan="fast", gap=None, zoom=False,
                          music=True, cap_style="karaoke", brand="purple",
                          rec="Music: 9:16 Fit or 16:9 | 30s+ High-energy beat drops, chorus peaks, crowd hype",
                          hint="drops, chorus peaks, crowd energy"),
}

log = print
STAGES = ["download", "read", "transcribe", "judge", "copy", "render"]
STAGE_NAMES = dict(download="Downloading audio", read="Reading audio", transcribe="Transcribing speech",
                   judge="Judging stories", copy="Writing titles and descriptions", render="Editing and rendering clips")
STAGE_WEIGHT = dict(download=0.12, read=0.03, transcribe=0.30, judge=0.28, copy=0.05, render=0.22)
CANCEL = threading.Event()


def prog(stage, frac=0.0, text=""):
    """Progress hook. The GUI replaces this to drive its progress bars."""


class Cancelled(Exception):
    pass


def check_cancel():
    if CANCEL.is_set():
        raise Cancelled("Stopped by you.")


# ----------------------------------------------------------------- helpers
def strip_cookie_args(cmd_list):
    """Strip cookie-related flags from cmd_list for clean automatic fallback."""
    clean_cmd = []
    skip = False
    for c in cmd_list:
        if skip:
            skip = False
            continue
        if c in ("--cookies-from-browser", "--cookies"):
            skip = True
            continue
        clean_cmd.append(c)
    return clean_cmd


def run(cmd, cwd=None, check=True):
    cmd_list = [str(c) for c in cmd]
    if cmd_list:
        cmd_list[0] = get_tool_path(cmd_list[0])
    try:
        p = subprocess.run(cmd_list, cwd=cwd or str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", creationflags=NOWIN)
    except FileNotFoundError as e:
        tool_name = Path(cmd_list[0]).name if cmd_list else "command"
        raise RuntimeError(f"Required binary '{tool_name}' was not found at '{cmd_list[0]}'. Please ensure the AutoClipper folder is complete. (Error: {e})")
    if p.returncode != 0 and any(c in cmd_list for c in ("--cookies-from-browser", "--cookies")):
        log("   (Browser cookies failed or locked — retrying download automatically without cookies...)")
        STATE["cookies"] = ""
        return run(strip_cookie_args(cmd_list), cwd=cwd, check=check)
    if check and p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"{Path(str(cmd_list[0])).name} failed:\n{err[-1500:]}")
    return p


def run_live(cmd, label, stage="download"):
    """Run yt-dlp, report download progress, and allow Stop."""
    cmd_list = [str(c) for c in cmd]
    if cmd_list:
        cmd_list[0] = get_tool_path(cmd_list[0])
    try:
        p = subprocess.Popen(cmd_list, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", creationflags=NOWIN)
    except FileNotFoundError as e:
        tool_name = Path(cmd_list[0]).name if cmd_list else "command"
        raise RuntimeError(f"Required binary '{tool_name}' not found at '{cmd_list[0]}'. (Error: {e})")
    tail, all_output, last = [], [], -10.0
    for line in p.stdout:
        if CANCEL.is_set():
            p.kill()
            raise Cancelled("Stopped by you.")
        all_output.append(line)
        tail = (tail + [line])[-40:]
        m = re.search(r"\[download\]\s+([\d.]+)% of\s+~?\s*([\d.]+\S+)", line)
        if m:
            pct = float(m.group(1))
            extra = re.search(r"at\s+(\S+)\s+ETA\s+(\S+)", line)
            info = f"{pct:.0f}% of {m.group(2)}" + (f"  |  {extra.group(1)}  |  ETA {extra.group(2)}" if extra else "")
            prog(stage, pct / 100.0, f"{label}: {info}")
            if pct >= last + 5:
                last = pct
                log(f"   {label} {pct:.0f}% (of {m.group(2)})")
    p.wait()
    if p.returncode != 0:
        if any(c in cmd_list for c in ("--cookies-from-browser", "--cookies")):
            log("   (Browser cookies failed or locked — retrying download automatically without cookies...)")
            STATE["cookies"] = ""
            return run_live(strip_cookie_args(cmd_list), label, stage=stage)
        err_text = "".join(tail).strip()
        if not err_text:
            err_text = "".join(all_output).strip()
        if not err_text:
            err_text = f"Process exited with error code {p.returncode}."
        raise RuntimeError(f"yt-dlp failed (code {p.returncode}):\n{err_text[-1500:]}")


@functools.lru_cache(maxsize=1)
def has_nvidia():
    return bool(shutil.which("nvidia-smi")) and run(["nvidia-smi"], check=False).returncode == 0


def ensure_ytdlp():
    """Ensure standalone yt-dlp binary exists; download official standalone binary if missing."""
    tool = get_tool_path("yt-dlp")
    if Path(tool).is_file():
        return tool
    dest = ROOT / "tools" / "ffmpeg" / "bin" / "yt-dlp.exe"
    dest.parent.mkdir(parents=True, exist_ok=True)
    log("   Standalone yt-dlp.exe not found. Downloading official binary automatically...")
    try:
        url = "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        log("   yt-dlp.exe ready.")
        return str(dest)
    except Exception as e:
        log(f"   (Automatic download of yt-dlp failed: {e})")
        return "yt-dlp"


def ytdlp_cmd():
    # 1. Bundled standalone yt-dlp binary
    tool = get_tool_path("yt-dlp")
    if Path(tool).is_file():
        cmd = [tool]
    elif not getattr(sys, "frozen", False) and Path(sys.executable).name.lower().startswith("python"):
        cmd = [sys.executable, "-m", "yt_dlp"]
    else:
        # Download standalone binary or locate
        downloaded = ensure_ytdlp()
        cmd = [downloaded]

    # Proven flags to prevent bot-detection, JS challenge failures and DPAPI crashes
    cmd += [
        "--no-playlist",
        "--no-check-certificates",
        "--extractor-args", "youtube:player_client=android,web",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    ]

    # Cookie handling: support cookies.txt file or selected browser
    cookie_file = ROOT / "cookies.txt"
    c_choice = (STATE.get("cookies") or "").strip()
    if cookie_file.exists() and (c_choice.lower().startswith("cookies.txt") or c_choice in ("", "(None - Recommended)")):
        cmd += ["--cookies", str(cookie_file)]
    elif c_choice and not c_choice.startswith("(") and c_choice.lower() not in ("none", "off", "0", "false"):
        browser = c_choice.split()[0].lower()
        cmd += ["--cookies-from-browser", browser]

    return cmd


def ollama_bin():
    p = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if p.exists():
        return str(p)
    sys_ollama = shutil.which("ollama") or shutil.which("ollama.exe")
    if sys_ollama:
        return sys_ollama
    for pf in [os.environ.get("ProgramFiles", "C:\\Program Files"), os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")]:
        p = Path(pf) / "Ollama" / "ollama.exe"
        if p.exists():
            return str(p)
    return str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe")


def ensure_ollama():
    import requests
    try:
        requests.get(OLLAMA_URL, timeout=2)
        return
    except Exception:
        pass

    bin_path = ollama_bin()
    if not Path(bin_path).exists() and not shutil.which(bin_path):
        log("   Ollama not found on this PC. Downloading Ollama automatically (one time only)...")
        setup_exe = ROOT / "tools" / "OllamaSetup.exe"
        setup_exe.parent.mkdir(parents=True, exist_ok=True)
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            with requests.get("https://ollama.com/download/OllamaSetup.exe", headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(setup_exe, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            log("   Installing Ollama silently in the background...")
            subprocess.run([str(setup_exe), "/SILENT", "/NORESTART"], check=True)
            time.sleep(4)
            bin_path = ollama_bin()
        except Exception as e:
            log(f"   (Automatic Ollama installation failed: {e})")
            raise RuntimeError(f"Ollama is required for local AI story judging. Please install free Ollama from https://ollama.com\nError: {e}")

    if not Path(bin_path).exists() and not shutil.which(bin_path):
        raise RuntimeError(f"Ollama binary not found at '{bin_path}'. Please install Ollama from https://ollama.com and restart.")

    try:
        subprocess.Popen([bin_path, "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=NOWIN)
    except FileNotFoundError as e:
        raise RuntimeError(f"Failed to start Ollama at '{bin_path}': {e}")
    for _ in range(40):
        time.sleep(1)
        try:
            requests.get(OLLAMA_URL, timeout=2)
            return
        except Exception:
            pass
    raise RuntimeError("Ollama server did not start. If Ollama was just installed, please restart the app.")


def clean_work_dir():
    """Remove leftover temp folders from previously crashed or killed jobs to reclaim disk space."""
    if WORK_DIR.exists():
        for p in WORK_DIR.glob("job_*"):
            try:
                shutil.rmtree(p, ignore_errors=True)
            except Exception:
                pass


def ensure_model(name):
    """Make sure the chosen Ollama model exists locally; fall back to lighter models or available models if download fails."""
    import requests
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
        have = {m.get("name") for m in tags} | {m.get("model") for m in tags}
    except Exception:
        have = set()
    if name in have or f"{name}:latest" in have:
        return name
    # Check if we already have a suitable fallback installed locally to avoid blocking downloads
    for candidate in ("qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"):
        if candidate in have or f"{candidate}:latest" in have:
            log(f"   Model '{name}' not found locally, using available model '{candidate}'...")
            return candidate
    log(f"   Downloading AI model {name} (one time only, can take a few minutes)...")
    p = run([ollama_bin(), "pull", name], check=False)
    if p.returncode != 0:
        if name != FAST_MODEL:
            log(f"   Could not download {name}; falling back to {FAST_MODEL}...")
            return ensure_model(FAST_MODEL)
        elif name != FALLBACK_MODEL:
            log(f"   Could not download {name}; falling back to {FALLBACK_MODEL}...")
            return ensure_model(FALLBACK_MODEL)
        raise RuntimeError("Could not download the AI model:\n" + (p.stderr or p.stdout)[-800:])
    return name


def unload_llm():
    try:
        import requests
        requests.post(f"{OLLAMA_URL}/api/generate", json=dict(model=STATE["model"], keep_alive=0), timeout=5)
    except Exception:
        pass


def llm(prompt, as_json=True):
    import requests
    target_model = STATE.get("model") or DEFAULT_MODEL
    # Try with json format; if the model or Ollama throws 500, retry with plain text prompt and parse json with regex
    body = dict(model=target_model, prompt=prompt, stream=False, keep_alive="30m",
                options=dict(temperature=0.2, num_ctx=4096))
    if as_json:
        body["format"] = "json"

    r = None
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=900)
    except Exception as e:
        log(f"   [LLM warning] Request failed ({e}), checking server...")
        ensure_ollama()
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=900)

    if r.status_code == 500 and as_json:
        log("   [LLM note] Server 500 on json grammar constraint, retrying with raw prompt...")
        body.pop("format", None)
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=900)

    if r.status_code != 200:
        # Check if model issue
        err_msg = r.text
        log(f"   [LLM note] HTTP {r.status_code} ({err_msg[:200]}), attempting fallback to {FALLBACK_MODEL}...")
        body["model"] = FALLBACK_MODEL
        body.pop("format", None)
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=900)

    r.raise_for_status()
    return r.json().get("response", "")


def parse_json(txt):
    try:
        d = json.loads(txt)
        return d if isinstance(d, dict) else {}
    except Exception:
        m = re.search(r"\{.*\}", txt or "", re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


def num(v, default, lo=1.0, hi=10.0):
    try:
        return max(lo, min(hi, float(v)))
    except Exception:
        return default


def truthy(v):
    return v in (True, 1, "true", "True", "yes")


def safe_name(s):
    return re.sub(r"[^\w\-]+", "_", s).strip("_")[:50] or "video"


def ev(v):
    return int(v) // 2 * 2


def hms(t):
    t = int(max(0, t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


# ------------------------------------------------------------ focus parsing
TS_RE = re.compile(r"(?<!\d)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?!\d)")


def _ts(m):
    return int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + int(m.group(3))


def parse_focus(text):
    """'find the funniest part from 1:00:00 to 1:30:00' -> ('find the funniest part', [3600, 5400])."""
    text = (text or "").strip()
    ms = list(TS_RE.finditer(text))
    rng = [None, None]
    topic = text
    if len(ms) >= 2:
        a, b = _ts(ms[0]), _ts(ms[1])
        rng = [min(a, b), max(a, b)]
        topic = TS_RE.sub(" ", text)
    elif len(ms) == 1:
        t = _ts(ms[0])
        before = text[:ms[0].start()].lower()
        rng = [None, t] if re.search(r"(until|till|before|up to|ending(?: at)?|end(?: at)?)\s*$", before) else [t, None]
        topic = TS_RE.sub(" ", text)
    glue = re.sub(r"\b(from|to|until|till|between|and|starting|start|ending|end|at|up|after|before|since|only|the|"
                  r"part|timestamp|time)\b|[-–—,.]", " ", topic, flags=re.I)
    if not re.search(r"[A-Za-z]{3,}", glue):
        topic = ""
    topic = re.sub(r"\s+", " ", topic).strip()
    edge = r"(?:from|to|until|till|between|and|starting|start|ending|end|at|after|before|since|[-–—,.])"
    prev = None
    while prev != topic:                      # drop dangling connectors left behind by the removed timestamps
        prev = topic
        topic = re.sub(rf"^\s*{edge}\s+|\s+{edge}\s*$", " ", topic, flags=re.I).strip()
    return topic, rng


# --------------------------------------------------------------- profanity
# Edit/extend with a file named censor_words.txt next to this script (one word per line).
BAD_ANY = re.compile(r"f+u+c+k|f+u+k+|sh[i1]+t|b[i1]tch|c[u*]nt|n[i1*]gg|motherf|\*{2,}", re.I)
BAD_EXACT = re.compile(
    r"^(?:a+ss?|a+ss?holes?|bastards?|dicks?|dickheads?|pussy|pussies|sluts?|whores?|fags?|faggots?|"
    r"retards?|retarded|cocks?|dumbass(?:es)?|jackass(?:es)?)$", re.I)
try:
    EXTRA_BAD = [w.strip().lower() for w in (ROOT / "censor_words.txt").read_text(encoding="utf-8").splitlines()
                 if w.strip() and not w.startswith("#")]
except Exception:
    EXTRA_BAD = []


def is_bad(tok):
    t = re.sub(r"[^A-Za-z0-9*']", "", tok or "").lower()
    if not t:
        return False
    return bool(BAD_ANY.search(t) or BAD_EXACT.match(t) or any(t.startswith(w) for w in EXTRA_BAD))


def mask_word(tok):
    core = re.sub(r"[^A-Za-z0-9*']", "", tok)
    return core[0] + "*" * (len(core) - 1) if len(core) > 1 else "*"


def mask_text(text):
    return re.sub(r"[A-Za-z0-9*']+", lambda m: mask_word(m.group(0)) if is_bad(m.group(0)) else m.group(0),
                  text or "")


def bleep_intervals(words, dur):
    """words are relative to the (edited) clip timeline."""
    iv = []
    for x in words:
        if is_bad(x["w"]):
            a = max(0.0, x["s"] - 0.06)
            b = min(dur, max(x["e"] + 0.10, x["s"] + 0.30))
            if b > a:
                iv.append([a, b])
    iv.sort()
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


# ------------------------------------------------------------------- setup
def load_whisper():
    from faster_whisper import WhisperModel
    if has_nvidia():
        try:
            return WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
        except Exception as e:
            log(f"GPU Whisper unavailable ({e}); using CPU.")
    threads = max(1, min(4, (os.cpu_count() or 4) - 1))
    return WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=threads)


def setup_models():
    prepare_env()
    MODEL_DIR.mkdir(exist_ok=True)
    if not YUNET_PATH.exists():
        print("Downloading YuNet face detector...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    print("Downloading Whisper model...")
    m = load_whisper()
    del m
    gc.collect()
    print("Models ready.")


# --------------------------------------------------------- 1. audio fetch
def fetch_audio(src, job, live_mins=0):
    is_live = False
    if re.match(r"https?://", src):
        log("   reading video info...")
        info = run(ytdlp_cmd() + ["--print", "%(title)s\t%(is_live)s", "--skip-download", src], check=False).stdout.strip().splitlines()
        title = "video"
        if info:
            parts = info[-1].split("\t")
            title = parts[0].strip() or "video"
            if len(parts) > 1 and parts[1].strip().lower() in ("true", "1"):
                is_live = True

        if is_live or live_mins > 0:
            capture_mins = live_mins if live_mins > 0 else 20
            log(f"   🔴 LIVE STREAM: {title}")
            log(f"   capturing the last {capture_mins} minutes of live broadcast buffer...")
            live_sec = int(capture_mins * 60)
            cmd = ytdlp_cmd() + [
                "--newline", "-f", "bestaudio/worst/best",
                "--downloader", "ffmpeg",
                "--downloader-args", f"ffmpeg_i:-t {live_sec}",
                "-o", str(job / "audio_raw.%(ext)s"), src
            ]
            run_live(cmd, "live audio capture")
        else:
            log(f"   downloading audio of: {title}  (long VODs can take several minutes)")
            run_live(ytdlp_cmd() + ["--newline", "-N", "8", "-f", "bestaudio/worst/best",
                                    "-o", str(job / "audio_raw.%(ext)s"), src], "audio download")
        raw = next(job.glob("audio_raw.*"))
    else:
        raw = Path(src)
        title = raw.stem
    probe = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                 "-of", "csv=p=0", raw], check=False).stdout.strip()
    if not probe:
        raise RuntimeError("This video file has NO AUDIO track (it looks like a video-only download), so there is "
                           "nothing to transcribe. Download it again with sound, or paste the YouTube/Twitch link "
                           "into the app instead.")
    wav = job / "audio16k.wav"
    run(["ffmpeg", "-y", "-i", raw, "-vn", "-ac", "1", "-ar", str(SR), wav])
    return title, raw, wav


# ------------------------------------------------------- 2. loudness peaks
def z(a):
    import numpy as np
    a = np.asarray(a, dtype=float)
    return (a - a.mean()) / (a.std() + 1e-9)


def rms_profile(y):
    import librosa
    import numpy as np
    chunk_len = 30 * 60 * SR  # 30-minute chunks to prevent huge memory spikes
    if len(y) <= chunk_len:
        return librosa.feature.rms(y=y, frame_length=3200, hop_length=1600)[0]
    parts = []
    for i in range(0, len(y), chunk_len):
        chunk = y[i:i + chunk_len]
        parts.append(librosa.feature.rms(y=chunk, frame_length=3200, hop_length=1600)[0])
    return np.concatenate(parts)


def find_peaks(rms, n, sep, t0=0.0, t1=1e12):
    import numpy as np
    ker = np.ones(30) / 30
    sm = np.convolve(rms, ker, mode="same")
    flux = np.convolve(np.abs(np.diff(rms, prepend=rms[0])), ker, mode="same")
    score = z(sm) + 0.5 * z(flux)
    picked = []
    for i in np.argsort(-score):
        t = i / 10.0
        if t < max(3.0, t0) or t > t1:
            continue
        if all(abs(t - p[0]) >= sep for p in picked):
            picked.append((float(t), float(score[i])))
            if len(picked) >= n:
                break
    return picked


def hype_0_10(score):
    return round(max(0.0, min(10.0, 4 + 1.2 * score)), 1)


# --------------------------------------------------------- 3. transcription
def seg_dict(sg, off=0.0):
    return dict(s=off + sg.start, e=off + sg.end, t=sg.text.strip(),
                words=[dict(w=w.word.strip(), s=off + w.start, e=off + w.end) for w in (sg.words or [])])


def transcribe(model, y, s, e, lang):
    chunk = y[int(s * SR):int(e * SR)]
    segs, _ = model.transcribe(chunk, language=lang, word_timestamps=True, vad_filter=True,
                               beam_size=1, condition_on_previous_text=False)
    return [seg_dict(sg, s) for sg in segs]


def transcribe_full(model, y, t0, t1, lang, chunk_sec=300):
    """Transcribe audio in 5-minute chunks to prevent faster-whisper from allocating
    multi-gigabyte FFT arrays in RAM on long videos/livestreams (which causes 16GB RAM exhaustion
    and forces heavy 100% HDD pagefile swapping)."""
    out = []
    total = max(1.0, t1 - t0)
    cur = t0
    last_log = t0
    while cur < t1:
        check_cancel()
        cur_end = min(t1, cur + chunk_sec)
        chunk = y[int(cur * SR):int(cur_end * SR)]
        if len(chunk) > int(0.5 * SR):
            segs, _ = model.transcribe(chunk, language=lang, word_timestamps=True, vad_filter=True,
                                       beam_size=1, condition_on_previous_text=False)
            for sg in segs:
                check_cancel()
                out.append(seg_dict(sg, cur))
        cur = cur_end
        frac = min(1.0, (cur - t0) / total)
        prog("transcribe", frac, f"{cur / 60:.0f} of {total / 60:.0f} min of audio")
        if cur - last_log >= 600 or cur >= t1:
            log(f"   transcribed {cur / 60:.0f} of {(t1 - t0) / 60:.0f} min")
            last_log = cur
    return out


LAUGHTER_REGEX = re.compile(
    r"\[laughter\]|\(laughter\)|\*laughs\*|\b(haha|hahaha|hehe|hehehe|lmao|lmfao|rofl|wheeze|giggle)\b", re.I)


def detect_laughter_segments(segs):
    """Detects segments containing genuine laughter, giggles, or wheezes."""
    laughs = []
    for i, s in enumerate(segs):
        txt = s.get("t", "")
        if LAUGHTER_REGEX.search(txt):
            laughs.append(dict(index=i, s=s["s"], e=s["e"], text=txt))
    return laughs


def deep_windows(rms, segs, t0, t1, span, story, limit=400):
    """Anchor-and-Expand narrative window generator:
    1. Slides windows over the transcript with standard coverage.
    2. Builds dedicated Anchor Windows around detected Laughter & Hype Climaxes (Setup -> Payoff)."""
    hz, step, wins, t = 10, span / 2, [], t0
    while t < max(t1 - span * 0.3, t0 + 1.0):
        ws, we = t, min(t1, t + span)
        a, b = int(ws * hz), max(int(ws * hz) + 1, int(we * hz))
        ss = [x for x in segs if ws <= x["s"] < we]
        wins.append(dict(ws=ws, we=we, loud=float(rms[a:b].mean()) if b > a else 0.0,
                         words=sum(len(x["words"]) for x in ss), segs=ss, has_laugh=False))
        t += step

    # Anchor Windows for Laughter & Climaxes:
    # A laugh/climax at T needs the narrative setup before it (T - 25s) and the aftermath (T + 15s)
    laugh_events = detect_laughter_segments(segs)
    for le in laugh_events:
        lt = (le["s"] + le["e"]) / 2
        if t0 <= lt <= t1:
            ws = max(t0, lt - span * 0.65)  # 65% before the laugh for the setup!
            we = min(t1, ws + span)
            ss = [x for x in segs if ws <= x["s"] < we]
            a, b = int(ws * hz), max(int(ws * hz) + 1, int(we * hz))
            wins.append(dict(ws=ws, we=we, loud=float(rms[a:b].mean()) if b > a else 0.0,
                             words=sum(len(x["words"]) for x in ss), segs=ss, has_laugh=True))

    if not wins:
        return []
    zl, zw = z([w["loud"] for w in wins]), z([w["words"] for w in wins])
    for w, a, b in zip(wins, zl, zw):
        laugh_bonus = 1.2 if w.get("has_laugh") else 0.0
        w["heat"] = float(a + 0.6 * b + laugh_bonus)
        w["hype"] = hype_0_10(float(a + laugh_bonus))
        w["peak"] = (w["ws"] + w["we"]) / 2
    wins = [w for w in wins if w["words"] >= (20 if story else 5)]
    wins.sort(key=lambda w: -w["heat"])
    return wins[:limit]


# ------------------------------------------------------------ 4. scoring
def pick_range(segs, a, b, lo, hi, dur):
    """Snap a clip to whole transcript segments: starts at a sentence start, ends on a sentence end.
    Guarantees clip length is AT LEAST lo seconds (minimum 30s) and does not cut off context."""
    lo = max(30.0, float(lo))
    n = len(segs)
    a = max(0, min(a, n - 1))
    b = max(a, min(b, n - 1))
    span = lambda i, j: segs[j]["e"] - segs[i]["s"]

    # If too long, trim from end
    while span(a, b) > hi and b > a:
        b -= 1

    # If too short, expand in both directions to reach >= lo (30s)
    while span(a, b) < lo:
        if b + 1 < n and (a == 0 or span(a, b + 1) <= hi):
            b += 1
        elif a > 0 and span(a - 1, b) <= hi:
            a -= 1
        elif b + 1 < n:
            b += 1
        else:
            break

    cs = max(0.0, segs[a]["s"] - 0.30)
    ce = min(dur, segs[b]["e"] + 0.60)
    # Guarantee minimum 30 seconds span
    if ce - cs < lo and dur >= lo:
        diff = lo - (ce - cs)
        cs = max(0.0, cs - diff / 2)
        ce = min(dur, ce + diff / 2)
        if ce - cs < lo and ce == dur:
            cs = max(0.0, ce - lo)
        elif ce - cs < lo and cs == 0.0:
            ce = min(dur, cs + lo)

    return a, b, cs, ce


def seg_lines(segs):
    return "\n".join(f"#{i} ({x['s']:.0f}s-{x['e']:.0f}s) {x['t']}" for i, x in enumerate(segs))


def focus_block(topic, scoring=True):
    if not topic:
        return ""
    if not scoring:
        return f'The viewer asked for clips about: "{topic}". The clip must stay on that request.\n'
    return (f'\nUSER REQUEST: "{topic}"\nOnly choose a clip whose content matches this request. If nothing in this '
            f'window matches it, set "matches" to false and give every score 1.\n')


def score_window(kind, prof, w, dur, topic):
    segs = w["segs"]
    if not segs:
        return None
    win_span = segs[-1]["e"] - segs[0]["s"]
    if win_span < 28.0:
        return None

    laugh_hint = ("\nIMPORTANT CONTEXT NOTE: This window contains a detected laughter/hype reaction! "
                  "Make sure start_seg includes the setup/premise leading up to this moment so the payoff makes complete sense."
                  if w.get("has_laugh") else "")

    goal = (
        "CRITICAL RULES FOR SELECTING A VIRAL STANDALONE CLIP (TIER-1 US AUDIENCE):\n"
        "1. ZERO-CONTEXT STRANGER TEST: A viewer scrolling on TikTok/Shorts who has NEVER seen this streamer must "
        "understand what is happening within the first 3 seconds without having watched any other part of the video. "
        "DO NOT start mid-sentence, mid-conversation, or with unresolved pronouns ('and then he did it...'). "
        "start_seg MUST be a clean sentence that introduces who is talking, what the challenge is, or the topic.\n"
        "2. COMPLETE NARRATIVE ARC: The clip must have a complete beginning (setup/context), middle (escalation, tension, debate, or funny banter), "
        "and end (punchline, crazy reaction, shocking outcome, or actionable takeaway).\n"
        "3. TIER-1 US SLANG & HUMOR: Target US TikTok & Reels audience. Focus on dramatic irony, relatable gamer pain, hilarious fails, "
        "and unexpected chaos. Titles must use natural US slang ('bro really thought...', 'aint no way he did that 💀', 'caught in 4k', 'he actually cooked').\n"
        "4. DURATION: Total clip length MUST BE AT LEAST 30 SECONDS (ideal range: 35 to 65 seconds). "
        "Never pick short 5-15 second fragments.\n"
        "5. CLEAN CUTOFF: End immediately after the payoff or reaction is delivered. Do not cut someone off mid-thought.\n"
    )
    prompt = (
        f"You are an elite viral short-form video editor (TikTok, Shorts, Reels) working on {kind} content.\n"
        f"Choose the BEST consecutive segments for a complete, standalone 30s+ viral clip.\n\n"
        f"{goal}\n"
        f"Niche focus: {prof['hint']}.\n{focus_block(topic)}{laugh_hint}\n"
        f"Target clip length: {prof['min']}-{prof['max']} seconds.\n"
        f"If this window has no compelling complete story, set has_story to false.\n"
        f"Be a strict critic: 5 is average, 8 or more is rare viral quality.\n\n"
        f"Reply ONLY with JSON:\n"
        f'{{"matches": true, "has_story": true, "start_seg": 0, "end_seg": 0, "hook": 1-10, '
        f'"story": 1-10, "payoff": 1-10, "context": 1-10, "title": "punchy 4-7 word title", '
        f'"reason": "one sentence explaining the full context and payoff"}}\n\n'
        f"TRANSCRIPT SEGMENTS:\n{seg_lines(segs)}"
    )
    d = parse_json(llm(prompt))
    n = len(segs)
    try:
        a, b = int(d.get("start_seg")), int(d.get("end_seg"))
    except Exception:
        a, b = 0, n - 1
    a, b, cs, ce = pick_range(segs, a, b, prof["min"], prof["max"], dur)

    # Reject anything under 30 seconds
    if ce - cs < 30.0:
        return None

    if topic and not truthy(d.get("matches", True)):
        return dict(score=1.0, start=cs, end=ce, segs=segs[a:b + 1], title="", reason="off-topic")

    base = (0.25 * num(d.get("hook"), 3) + 0.25 * num(d.get("story"), 3) + 0.25 * num(d.get("payoff"), 3)
            + 0.15 * num(d.get("context"), 3) + 0.10 * num(d.get("standalone", 5), 5))
    if not truthy(d.get("has_story")):
        base *= 0.5
    hw = 0.1 if prof["story"] else 0.25
    score = max(1.0, min(10.0, base * (1 - hw) + w.get("hype", 5) * hw))
    return dict(score=score, start=cs, end=ce, segs=segs[a:b + 1],
                title=str(d.get("title") or "Viral Moment"), reason=str(d.get("reason") or ""))


def review_clip(kind, prof, c, dur, topic):
    """Second pass: an 'editor' re-reads the exact clip, ensures standalone context,
    and writes a high-converting on-screen text hook tailored to the niche."""
    segs = c["segs"]
    prompt = (
        f"You are a master viral TikTok & Shorts editor reviewing a {kind} clip.\n"
        f"Clip duration: {c['end'] - c['start']:.1f}s. Segments are numbered.\n\n"
        f"GOALS:\n"
        f"1. Check STANDALONE CONTEXT: Does a stranger immediately understand what this is about? "
        f"If the very first segment starts mid-thought or mid-sentence, increase drop_start (0 to 2) to start cleanly.\n"
        f"2. Write a VIRAL ON-SCREEN TEXT HOOK (displayed for first 3.5 seconds):\n"
        f"   - Must be funny, curious, shocking, or dramatic (4 to 7 words).\n"
        f"   - TIER-1 US SLANG & HUMOR: Authentic US internet culture (e.g. 'Bro thought he was safe 😭', "
        f"'He had NO idea his mic was on 💀', 'Ain\\'t no way he survived this 😭', 'He is NOT recovering from this 💀', "
        f"'Bro is playing with his monitor off 💀', 'Wait for the ending... nah this is wild 💀').\n"
        f"   - Keep it concise (under 28 characters) so it renders cleanly on ONE SINGLE LINE.\n"
        f"   - Match the niche ({kind}): use natural internet slang and relatable hooks.\n"
        f"   - End with 1 relevant emoji (💀, 😭, 🤡, 🔥, 🚨, 💸, 😱, 🤯, 🎮).\n"
        f"{focus_block(topic, scoring=False)}\n"
        f"Reply ONLY with JSON: {{\"standalone\": 1-10, \"hook_ok\": true, \"drop_start\": 0, \"drop_end\": 0, "
        f"\"hook_text\": \"...\"}}\n\n"
        f"CLIP TRANSCRIPT:\n{seg_lines(segs)}"
    )
    d = parse_json(llm(prompt))
    ds, de = int(num(d.get("drop_start"), 0, 0, 2)), int(num(d.get("drop_end"), 0, 0, 2))
    if len(segs) - ds - de >= 1:
        new = segs[ds:len(segs) - de]
        # Never allow trimming to shrink the clip below 30 seconds!
        if new[-1]["e"] - new[0]["s"] >= 30.0:
            segs = new
    standalone = num(d.get("standalone"), 5)
    score = 0.5 * c["score"] + 0.5 * standalone
    if not truthy(d.get("hook_ok", True)):
        score *= 0.95
    # Enforce minimum 30 seconds
    start_t = max(0.0, segs[0]["s"] - 0.30)
    end_t = min(dur, segs[-1]["e"] + 0.60)
    if end_t - start_t < 30.0 and dur >= 30.0:
        end_t = min(dur, start_t + 30.0)
    c.update(segs=segs, start=start_t, end=end_t,
             score=max(1.0, min(10.0, score)), hook=str(d.get("hook_text") or c["title"]))
    return c


def overlap(a, b):
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    return max(0.0, inter) / max(1.0, min(a["end"] - a["start"], b["end"] - b["start"]))


# ------------------------------------------------------ 6. facecam detect
def detect_facecam(video, vstart, dur, job, idx, aspect):
    """Return (x,y,w,h) crop box around a stable small face (aspect = box width/height), or None."""
    import numpy as np
    import cv2
    if not YUNET_PATH.exists():
        return None
    pat = job / f"fc{idx}_%02d.jpg"
    run(["ffmpeg", "-y", "-ss", f"{vstart:.3f}", "-t", f"{dur:.3f}", "-i", video,
         "-vf", f"fps={10 / max(dur, 1):.4f}", "-frames:v", "10", pat], check=False)
    frames = sorted(job.glob(f"fc{idx}_*.jpg"))
    hits, W, H, det = [], 0, 0, None
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        H, W = img.shape[:2]
        if det is None:
            det = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (W, H), 0.7, 0.3, 5000)
        det.setInputSize((W, H))
        _, faces = det.detect(img)
        if faces is None:
            continue
        for fc in faces:
            x, y, w, h = fc[:4]
            if 0.03 * W <= w <= 0.22 * W:          # small face = facecam, not an in-game character
                hits.append((x + w / 2, y + h / 2, w))
    for f in frames:
        f.unlink(missing_ok=True)
    if len(hits) < 3:
        return None
    arr = np.array(hits)
    med = np.median(arr[:, :2], axis=0)
    good = arr[(np.abs(arr[:, 0] - med[0]) < 0.06 * W) & (np.abs(arr[:, 1] - med[1]) < 0.06 * H)]
    if len(good) < 3:
        return None
    cx, cy, fw = good[:, 0].mean(), good[:, 1].mean(), good[:, 2].mean()
    bw = min(W, fw * 3.4)
    bh = bw / aspect
    if bh > H:
        bh = H
        bw = bh * aspect
    x0 = min(max(0, cx - bw / 2), W - bw)
    y0 = min(max(0, cy - bh * 0.45), H - bh)
    return tuple(int(v) // 2 * 2 for v in (x0, y0, bw, bh))


# ---------------------------------------------- editing: dead air + fast cuts
def plan_cuts(words, d, gap, chunk=4.5, split=True):
    """Return [(a,b)] keep-intervals (clip-relative seconds). Gaps longer than `gap` are cut out.
    Uses generous padding to preserve conversational flow and context.
    Guarantees that final rendered clip stays AT LEAST 30 seconds."""
    if not words or gap is None:
        return [(0.0, d)]
    effective_gap = max(1.5, float(gap))
    ws = sorted((max(0.0, x["s"]), min(d, x["e"])) for x in words if x["e"] > 0 and x["s"] < d)
    if not ws:
        return [(0.0, d)]
    spans, (cs, ce) = [], ws[0]
    for a, b in ws[1:]:
        if a - ce <= effective_gap:
            ce = max(ce, b)
        else:
            spans.append((cs, ce))
            cs, ce = a, b
    spans.append((cs, ce))
    padded = []
    for i, (a, b) in enumerate(spans):
        # Generous padding: 0.30s before speech, 0.40s after speech to avoid clipping words/laughs
        a = 0.0 if (i == 0 and a < 0.6) else max(0.0, a - 0.30)
        b = d if (i == len(spans) - 1 and d - b < 0.9) else min(d, b + 0.40)
        padded.append((a, b))
    cuts = [(ws[k][1] + ws[k + 1][0]) / 2 for k in range(len(ws) - 1) if ws[k + 1][0] - ws[k][1] >= 0.04]
    pieces = []
    for a, b in padded:
        cur = a
        while split and b - cur > chunk * 1.4:
            cand = [t for t in cuts if cur + chunk * 0.6 <= t <= cur + chunk * 1.4]
            if not cand:
                break
            t = min(cand, key=lambda t: abs(t - (cur + chunk)))
            pieces.append((cur, t))
            cur = t
        pieces.append((cur, b))
    pieces = [(a, b) for a, b in pieces if b - a >= 0.3]

    # Guarantees that dead-air cuts NEVER shrink a clip below 30 seconds!
    ndur = sum(b - a for a, b in pieces)
    if ndur < 30.0 and d >= 30.0:
        return [(0.0, d)]
    return pieces or [(0.0, d)]


def remap_time(t, pieces):
    off = 0.0
    for a, b in pieces:
        if t <= b:
            return off + max(0.0, t - a)
        off += b - a
    return off


# -------------------------------------------------------------- captions
def ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def create_hook_banner(hook_text, out_path, W=1080, H=1920, brand_dict=None,
                       font_name=DEFAULT_FONT, font_scale=1.0):
    """Renders a sleek, modern, minimal 1-line hook banner with authentic Apple iOS emoji PNG."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None

    if not hook_text:
        return None

    brand = brand_dict or BRAND_COLORS["yellow"]
    clean_text, emoji_codes = extract_emojis(hook_text)
    clean_text = re.sub(r"[{}\\]", "", clean_text).strip().strip('"')
    if not clean_text:
        return None

    font_size = max(26, int(W * 0.052 * font_scale))   # ~56px on 1080p scaled
    font_path = resolve_font_path(font_name)
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    safe_w = int(W * 0.86)
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), clean_text.upper(), font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    emoji_size = int(font_size * 1.15) if emoji_codes else 0
    emoji_spacing = 8
    num_emojis = min(2, len(emoji_codes))
    total_emoji_w = (num_emojis * emoji_size + (num_emojis - 1) * emoji_spacing) if num_emojis else 0

    # User request: Keep it on 1 line unless it nears the edges of the safe zone!
    lines = [clean_text.upper()]
    if tw + total_emoji_w + 60 > safe_w:
        words = clean_text.upper().split()
        mid = max(1, len(words) // 2)
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]

    line_w = [draw.textbbox((0, 0), l, font=font)[2] for l in lines]
    max_lw = max(line_w)
    total_content_w = max_lw + (total_emoji_w + 14 if total_emoji_w else 0)

    pad_x = 28
    pad_y = 14
    line_gap = 6
    num_lines = len(lines)
    total_h = (th * num_lines) + (line_gap * (num_lines - 1)) + (pad_y * 2)
    total_w = total_content_w + (pad_x * 2)

    img = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))
    img_draw = ImageDraw.Draw(img)

    # Dark translucent pill card
    r_col, g_col, b_col = brand.get("rgb", (255, 221, 0))
    bg_col = (18, 18, 22, 235)
    border_col = (r_col, g_col, b_col, 220)

    radius = 16
    img_draw.rounded_rectangle([(1, 1), (total_w - 2, total_h - 2)], radius=radius,
                               fill=bg_col, outline=border_col, width=2)

    # Stylish brand accent bar on left edge of pill
    accent_w = 5
    img_draw.rounded_rectangle([(pad_x - 12, pad_y + 2), (pad_x - 12 + accent_w, total_h - pad_y - 2)],
                               radius=2, fill=(r_col, g_col, b_col, 255))

    cur_y = pad_y
    for i, line in enumerate(lines):
        img_draw.text((pad_x + 2, cur_y + 2), line, font=font, fill=(0, 0, 0, 220))
        img_draw.text((pad_x, cur_y), line, font=font, fill=(255, 255, 255, 255))
        if i == 0 and num_emojis and len(lines) == 1:
            em_x = pad_x + line_w[0] + 12
            for ecode in emoji_codes[:num_emojis]:
                epath = EMOJI_DIR / f"{ecode}.png"
                if epath.exists():
                    try:
                        eimg = Image.open(epath).convert("RGBA").resize((emoji_size, emoji_size), Image.Resampling.LANCZOS)
                        img.paste(eimg, (em_x, cur_y + (th - emoji_size) // 2), eimg)
                        em_x += emoji_size + emoji_spacing
                    except Exception:
                        pass
        cur_y += th + line_gap

    if num_emojis and len(lines) > 1:
        em_x = pad_x + line_w[-1] + 12
        em_y = pad_y + th + line_gap + (th - emoji_size) // 2
        for ecode in emoji_codes[:num_emojis]:
            epath = EMOJI_DIR / f"{ecode}.png"
            if epath.exists():
                try:
                    eimg = Image.open(epath).convert("RGBA").resize((emoji_size, emoji_size), Image.Resampling.LANCZOS)
                    img.paste(eimg, (em_x, em_y), eimg)
                    em_x += emoji_size + emoji_spacing
                except Exception:
                    pass

    img.save(out_path)
    return total_w, total_h


def build_ass(words, path, censor, hook, ndur, show_captions=True, show_hook=True, W=1080, H=1920,
              brand_dict=None, style_name="auto", layout="auto", cam_pos="top", has_hook_image=False,
              font_name=DEFAULT_FONT, font_scale=1.0, cap_pos="bottom"):
    """Generates minimal, non-overlapping viral TikTok/Shorts captions and fallback in-text hook."""
    if not show_captions and not (show_hook and not has_hook_image):
        Path(path).unlink(missing_ok=True)
        return

    brand = brand_dict or BRAND_COLORS["yellow"]
    brand_ass = brand.get("ass", "&H0000DDFF&").rstrip("&") + "&"

    # Medium font sizes (balanced, high visibility on mobile, scaled by user choice)
    cap_size = max(24, int(W * 0.048 * font_scale))    # ~52px at 1080p
    hook_size = max(26, int(W * 0.052 * font_scale))   # ~56px at 1080p

    safe_left = int(W * 0.10)
    safe_right = int(W * 0.12)

    # Position captions according to user choice
    if cap_pos == "middle":
        cap_margin_bottom = int(H * 0.46)
    elif cap_pos == "top":
        cap_margin_bottom = int(H * 0.68)
    else:  # default "bottom"
        cap_margin_bottom = int(H * 0.28)

    # Position hook at middle cut seam if split layout
    if layout == "split":
        if cam_pos == "bottom":
            hook_margin_top = int(2 * H / 3 - hook_size / 2)
        else:
            hook_margin_top = int(H / 3 - hook_size / 2)
    elif str(cam_pos).startswith("pip"):
        hook_margin_top = int(H * 0.12)
    else:
        hook_margin_top = int(H * 0.12)

    # Caption style configs (Open-Source ASS Presets)
    user_font = font_name or DEFAULT_FONT
    border_style = 1
    outline_val = 3
    shadow_val = 1
    back_colour = "&H80000000"
    words_per_chunk = 2
    force_uppercase = True
    anim_type = "bounce"

    if style_name == "podcast":
        border_style = 3   # boxed pill background
        outline_val = 8
        shadow_val = 0
        back_colour = "&H99000000"
        words_per_chunk = 4
        force_uppercase = False
        anim_type = "clean"
    elif style_name == "abdaal":
        border_style = 3
        outline_val = 8
        shadow_val = 0
        back_colour = "&HB0141414"
        words_per_chunk = 3
        force_uppercase = False
        anim_type = "highlight"
    elif style_name == "neon":
        outline_val = 4
        shadow_val = 2
        words_per_chunk = 2
        anim_type = "neon"
    elif style_name == "beast":
        words_per_chunk = 2
        outline_val = 4
        shadow_val = 2
        anim_type = "bounce"
    elif style_name == "karaoke":
        words_per_chunk = 2
        outline_val = 3
        shadow_val = 1
        anim_type = "wave"
    elif style_name == "hormozi":
        words_per_chunk = 2
        outline_val = 3
        shadow_val = 1
        anim_type = "pop"

    head = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,"
        "Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,"
        "MarginL,MarginR,MarginV,Encoding\n"
        # Subtitles
        f"Style: Cap,{user_font},{cap_size},&H00FFFFFF,&H00FFFFFF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline_val},{shadow_val},2,{safe_left},{safe_right},{cap_margin_bottom},1\n"
        # Hook Fallback
        f"Style: Hook,{user_font},{hook_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,3,10,0,8,{safe_left},{safe_right},{hook_margin_top},1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = []
    if show_hook and hook and not has_hook_image:
        # Fallback ASS hook: strip emoji to avoid wireframes
        h_text, _ = extract_emojis(hook)
        h = re.sub(r"[{}\\]", "", mask_text(h_text) if censor else h_text).upper().strip().strip('"')
        wrap_limit = max(26, int(W / (hook_size * 0.72)))
        h = "\\N".join(textwrap.wrap(h, wrap_limit)[:2])
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(3.5, max(1.0, ndur)))},Hook,,0,0,0,,"
                     f"{{\\fad(180,250)}}{h}")

    if show_captions and words:
        ws = []
        for x in words:
            if not x["w"].strip():
                continue
            clean_w, _ = extract_emojis(x["w"])
            if not clean_w.strip():
                continue
            txt = mask_word(clean_w) if (censor and is_bad(clean_w)) else clean_w
            txt = txt.upper() if force_uppercase else txt
            ws.append(dict(w=re.sub(r"[{}\\]", "", txt).strip(), s=x["s"], e=x["e"]))

        # Chunk words
        chunks = []
        for i in range(0, len(ws), words_per_chunk):
            chunks.append(ws[i:i + words_per_chunk])

        # Strict chronological non-overlapping times (CRITICAL FIX FOR FAST TALKING)
        for i, chunk in enumerate(chunks):
            c_start = chunk[0]["s"]
            if i + 1 < len(chunks):
                c_end = min(chunk[-1]["e"] + 0.10, chunks[i + 1][0]["s"] - 0.02)
            else:
                c_end = chunk[-1]["e"] + 0.12
            if c_end <= c_start:
                c_end = c_start + 0.08

            for j, cur in enumerate(chunk):
                w_start = cur["s"]
                if j + 1 < len(chunk):
                    w_end = min(cur["e"], chunk[j + 1]["s"])
                else:
                    w_end = c_end
                if w_end <= w_start:
                    w_end = w_start + 0.06

                # Format line with smooth kinetic typography & open-source ASS acceleration curves
                text_parts = []
                for k, c in enumerate(chunk):
                    if k == j:
                        if anim_type == "bounce":  # MrBeast spring pop with ease-out
                            text_parts.append(f"{{\\c{brand_ass}\\fscx90\\fscy90\\t(0,90,0.7,\\fscx118\\fscy118)\\t(90,200,1.3,\\fscx100\\fscy100)}}{c['w']}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}")
                        elif anim_type == "wave":  # Wave bounce with tilt
                            text_parts.append(f"{{\\c{brand_ass}\\t(0,80,0.7,\\frz-2\\fscy122\\fscx106)\\t(80,190,1.3,\\frz0\\fscy100\\fscy100)}}{c['w']}{{\\c&H00FFFFFF&\\frz0\\fscx100\\fscy100}}")
                        elif anim_type == "neon":  # Cyberpunk shimmer glow
                            text_parts.append(f"{{\\c{brand_ass}\\3c&H00FFF000&\\t(0,90,0.8,\\fscx112\\fscy112\\3c&H00FF00FF&)\\t(90,210,1.2,\\fscx100\\fscy100\\3c&H00000000&)}}{c['w']}{{\\c&H00FFFFFF&\\3c&H00000000&\\fscx100\\fscy100}}")
                        elif anim_type == "highlight":  # Ali Abdaal modern highlighter
                            text_parts.append(f"{{\\c{brand_ass}\\b1\\t(0,100,0.8,\\fscx106\\fscy106)\\t(100,200,1.2,\\fscx100\\fscy100)}}{c['w']}{{\\c&H00FFFFFF&\\b0\\fscx100\\fscy100}}")
                        elif anim_type == "clean":  # Podcast clean subtitle
                            text_parts.append(f"{{\\c{brand_ass}&}}{c['w']}{{\\c&H00FFFFFF&}}")
                        else:  # Hormozi punch pop with settle
                            text_parts.append(f"{{\\c{brand_ass}\\fscx94\\fscy94\\t(0,80,0.8,\\fscx114\\fscy114)\\t(80,180,1.2,\\fscx100\\fscy100)}}{c['w']}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}")
                    else:
                        text_parts.append(c["w"])
                text = " ".join(text_parts)
                lines.append(f"Dialogue: 0,{ass_time(w_start)},{ass_time(w_end)},Cap,,0,0,0,,{text}")

    if lines:
        Path(path).write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
    else:
        Path(path).unlink(missing_ok=True)


# --------------------------------------------------------------- music
def music_tracks():
    MUSIC_DIR.mkdir(exist_ok=True)
    tr = [p for p in MUSIC_DIR.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")]
    if tr:
        return tr
    # No tracks supplied: synthesize a soft ambient bed once (drop real royalty-free tracks in /music for better results)
    chords = [(220, 261.63, 329.63), (174.61, 220, 261.63), (261.63, 329.63, 392), (196, 246.94, 293.66)]
    expr = "+".join(f"between(mod(t,16),{4 * i},{4 * i + 4})*(" + "+".join(f"sin(2*PI*{f}*t)" for f in ch) + ")"
                    for i, ch in enumerate(chords))
    expr = f"0.12*({expr})*(0.55-0.45*cos(2*PI*mod(t,4)/4))"
    p = MUSIC_DIR / "_builtin_ambient.mp3"
    r = run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"aevalsrc=exprs='{expr}':s=44100:d=64", "-af",
             "lowpass=f=1800,aecho=0.8:0.7:500|900:0.35|0.25", "-b:a", "128k", p], check=False)
    return [p] if r.returncode == 0 and p.exists() else []


# --------------------------------------------------------- 7. compose 9:16
def probe_video(path):
    p = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height,r_frame_rate", "-of", "json", path])
    st = json.loads(p.stdout)["streams"][0]
    try:
        a, b = st["r_frame_rate"].split("/")
        fps = float(a) / float(b)
    except Exception:
        fps = 30.0
    return int(st["width"]), int(st["height"]), fps


def resolve_layout(layout, prof, sw, sh):
    if sw / sh < 0.8:                       # already vertical: just fill the screen
        return "fill"
    if layout == "auto":
        return "split" if prof["cam"] else "fit"
    return layout


def canvas_for(layout, sw, sh, ratio="9:16"):
    """Best quality: calculate canvas dimensions according to chosen ratio and layout."""
    if ratio == "16:9":
        w = ev(max(1920, min(sw, 3840)))
        return w, ev(w * 9 / 16)
    elif ratio == "1:1":
        s = ev(max(1080, min(sh, 2160)))
        return s, s
    elif ratio == "3:4":
        w = ev(max(1080, min(sw, 2160)))
        return w, ev(w * 4 / 3)
    else:  # 9:16
        if layout == "fit":
            w = ev(max(1080, min(sw, 2160)))
            return w, ev(w * 16 / 9)
        h = ev(max(1920, min(3840, sh * (1.5 if layout == "split" else 1.0))))
        return ev(h * 9 / 16), h


def encoders():
    out = run(["ffmpeg", "-hide_banner", "-encoders"], check=False).stdout
    names = []
    if has_nvidia() and "h264_nvenc" in out:
        names.append(["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "18", "-b:v", "0",
                      "-profile:v", "high"])
    elif "h264_qsv" in out:
        names.append(["-c:v", "h264_qsv", "-preset", "faster", "-global_quality", "20"])
    elif "h264_amf" in out:
        names.append(["-c:v", "h264_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", "18", "-qp_p", "20"])
    names.append(["-c:v", "libx264", "-preset", "fast", "-crf", "17", "-profile:v", "high"])
    return names


def compose(job, vid, vstart, audio_src, astart, dur, ass, layout, cam, out, W, H, fps,
            pieces, zooms, sw, sh, bleeps, music_track, cam_pos="top",
            hook_img_path=None, hook_banner_dim=None, fast_draft=False):
    N = len(pieces)
    ndur = sum(b - a for a, b in pieces)
    PH = ev(H / 3)
    BH = H - PH
    g = []
    # ---- video: cut into pieces, punch-in zoom per piece, concat
    if N > 1:
        g.append(f"[0:v]split={N}" + "".join(f"[vs{i}]" for i in range(N)))
    for i, (a, b) in enumerate(pieces):
        zoom = ""
        if zooms[i] > 1.001:
            cw, ch = ev(sw / zooms[i]), ev(sh / zooms[i])
            zoom = f"crop={cw}:{ch}:{ev((sw - cw) / 2)}:{ev((sh - ch) / 2)},scale={sw}:{sh}:flags=lanczos,"
        g.append(f"{'[vs%d]' % i if N > 1 else '[0:v]'}trim=start={a:.3f}:end={b:.3f},"
                 f"setpts=PTS-STARTPTS,{zoom}setsar=1[v{i}]")
    g.append(("".join(f"[v{i}]" for i in range(N)) + f"concat=n={N}:v=1:a=0[src]") if N > 1 else "[v0]null[src]")
    # ---- layout
    if layout == "split" and cam:
        x, y, w, h = cam
        if cam_pos == "bottom":
            g.append(f"[src]split=2[a][b];"
                     f"[b]crop=trunc(ih*{W}/{BH}/2)*2:ih,scale={W}:{BH}:flags=lanczos[top];"
                     f"[a]crop={w}:{h}:{x}:{y},scale={W}:{PH}:flags=lanczos[bot];"
                     f"[top][bot]vstack=inputs=2[v0x]")
        elif str(cam_pos).startswith("pip"):
            cw = ev(W * 0.36)
            ch = ev(cw * h / max(1, w))
            pad = 24
            pos_dict = {
                "pip-tr": f"main_w-overlay_w-{pad}:{pad}",
                "pip-tl": f"{pad}:{pad}",
                "pip-br": f"main_w-overlay_w-{pad}:main_h-overlay_h-{pad}",
                "pip-bl": f"{pad}:main_h-overlay_h-{pad}",
            }
            ov_pos = pos_dict.get(cam_pos, f"main_w-overlay_w-{pad}:{pad}")
            bw, bh = ev(W // 6), ev(H // 6)
            g.append(f"[src]split=3[bg_in][fg_in][cam_in];"
                     f"[bg_in]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},boxblur=6:2,scale={W}:{H}[bg];"
                     f"[fg_in]scale={W}:-2:flags=lanczos[fg];"
                     f"[bg][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2[base];"
                     f"[cam_in]crop={w}:{h}:{x}:{y},scale={cw}:{ch}:flags=lanczos[cam_sc];"
                     f"[base][cam_sc]overlay={ov_pos}[v0x]")
        else:  # default top
            g.append(f"[src]split=2[a][b];"
                     f"[a]crop={w}:{h}:{x}:{y},scale={W}:{PH}:flags=lanczos[top];"
                     f"[b]crop=trunc(ih*{W}/{BH}/2)*2:ih,scale={W}:{BH}:flags=lanczos[bot];"
                     f"[top][bot]vstack=inputs=2[v0x]")
    elif layout == "fill":
        g.append(f"[src]scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H}[v0x]")
    else:  # fit: full frame with untouched pixels, soft background only behind the bars
        bw, bh = ev(W // 6), ev(H // 6)
        g.append(f"[src]split=2[a][b];"
                 f"[a]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},boxblur=6:2,"
                 f"scale={W}:{H}[bg];"
                 f"[b]scale={W}:-2:flags=lanczos[fg];"
                 f"[bg][fg]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2[v0x]")

    # ---- inputs setup
    inputs = ["-ss", f"{vstart:.3f}", "-t", f"{dur:.3f}", "-i", vid,
              "-ss", f"{astart:.3f}", "-t", f"{dur:.3f}", "-i", audio_src]
    idx = 2

    # ---- optional hook banner image overlay (iOS emoji + centered on middle cut)
    v_cur = "[v0x]"
    if hook_img_path and Path(hook_img_path).exists() and hook_banner_dim:
        bw, bh = hook_banner_dim
        if layout == "split" and cam:
            if cam_pos == "bottom":
                hook_y = int(2 * H / 3 - bh / 2)
            else:
                hook_y = int(H / 3 - bh / 2)
        elif str(cam_pos).startswith("pip"):
            hook_y = int(H * 0.12)
        else:
            hook_y = int(H * 0.12)

        inputs += ["-loop", "1", "-t", "3.5", "-i", str(hook_img_path)]
        g.append(f"[{idx}:v]fade=in:st=0:d=0.20:alpha=1,fade=out:st=3.2:d=0.30:alpha=1[hook_f]")
        g.append(f"[v0x][hook_f]overlay=(main_w-overlay_w)/2:{hook_y}:enable='between(t,0,3.5)'[v_hooked]")
        v_cur = "[v_hooked]"
        idx += 1

    # ---- captions / subtitles
    # Note: Modern animated word-timed captions are handled by pycaps in Stage 2.
    # Legacy FFmpeg .ass burning is disabled to prevent duplicate overlapping captions.
    g.append(f"{v_cur}null[v]")

    # ---- audio: same cuts, bleeps, music with ducking, loudness
    if N > 1:
        g.append(f"[1:a:0]asplit={N}" + "".join(f"[as{i}]" for i in range(N)))
    for i, (a, b) in enumerate(pieces):
        fo = max(0.0, b - a - 0.015)
        g.append(f"{'[as%d]' % i if N > 1 else '[1:a:0]'}atrim=start={a:.3f}:end={b:.3f},asetpts=PTS-STARTPTS,"
                 f"afade=t=in:d=0.015,afade=t=out:st={fo:.3f}:d=0.015[a{i}]")
    g.append(("".join(f"[a{i}]" for i in range(N)) + f"concat=n={N}:v=0:a=1[vo0]") if N > 1 else "[a0]anull[vo0]")
    if bleeps:
        gate = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in bleeps)
        inputs += ["-f", "lavfi", "-i", f"aevalsrc=exprs='0.35*sin(2*PI*1000*t)*({gate})':s=48000:d={ndur:.3f}"]
        g.append(f"[vo0]aresample=48000,aformat=channel_layouts=stereo,volume=enable='{gate}':volume=0[am]")
        g.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo[bp]")
        g.append("[am][bp]amix=inputs=2:duration=first:dropout_transition=0,volume=2[vo1]")
        idx += 1
    else:
        g.append("[vo0]aresample=48000,aformat=channel_layouts=stereo[vo1]")
    cur = "[vo1]"
    if music_track:
        inputs += ["-stream_loop", "-1", "-t", f"{ndur:.3f}", "-i", str(music_track)]
        g.append("[vo1]asplit=2[vm][vsc]")
        g.append(f"[{idx}:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.18,"
                 f"afade=t=in:d=0.6,afade=t=out:st={max(0.0, ndur - 1.2):.3f}:d=1.2[mus]")
        g.append("[mus][vsc]sidechaincompress=threshold=0.02:ratio=10:attack=15:release=350[md]")
        g.append("[vm][md]amix=inputs=2:duration=first:dropout_transition=0,volume=2[mixed]")
        cur = "[mixed]"
    g.append(f"{cur}loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    graph = ";".join(g)

    last = None
    enc_list = encoders()
    if fast_draft:
        enc_list = [["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"]]
    for enc in enc_list:
        cmd = (["ffmpeg", "-y"] + inputs + ["-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                                              "-t", f"{ndur:.3f}", "-r", f"{fps:g}"]
               + enc + ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out])
        p = run(cmd, cwd=job, check=False)
        if p.returncode == 0:
            return ndur
        last = p.stderr[-1500:]
    raise RuntimeError("ffmpeg compose failed:\n" + str(last))


# ------------------------------------------------------------ 5. video dl
def get_video_section(src, s, e, job, idx):
    """Return (video_path, seek_seconds_inside_that_file). Always the best available quality."""
    if not re.match(r"https?://", src):
        return Path(src), s                              # local file: cut straight from the original, no re-encode
    for f in job.glob(f"clip{idx}_src.*"):
        f.unlink()
    run(ytdlp_cmd() + ["-f", "bestvideo[vcodec^=avc1]/bestvideo/best", "--download-sections",
                       f"*{s:.2f}-{e:.2f}", "--force-keyframes-at-cuts",
                       "-o", str(job / f"clip{idx}_src.%(ext)s"), src])
    return next(job.glob(f"clip{idx}_src.*")), 0.0


# ------------------------------------------------------ 8. SEO title/description
def make_copy(kind, src_title, topic, c, censor):
    tx = " ".join(sg["t"] for sg in c["segs"])
    prompt = (
        f"You are a top 1% viral YouTube Shorts & TikTok strategist writing metadata for a {kind} clip.\n"
        f"Source video: \"{src_title}\"\n"
        + (f"Viewer topic: \"{topic}\"\n" if topic else "")
        + f"Clip context: {c['reason']}\nTranscript: \"{tx[:2000]}\"\n"
        f"RULES FOR VIRAL TITLES & DESCRIPTIONS (TIER-1 US AUDIENCE):\n"
        f"- 'title': High-CTR, curiosity-driven, punchy (max 52 characters). "
        f"MUST use authentic US internet slang & humor (e.g. 'Bro really thought he got away 💀', "
        f"'Ain\\'t no way he survived this 😭', 'Bro got caught in 4K', 'He actually cooked with this one 🔥'). "
        f"Include at most one viral emoji.\n"
        f"- 'description': Exactly 2 snappy sentences matching US TikTok creators. Sentence 1 hooks the reader with the core drama/lesson. "
        f"Sentence 2 asks an engaging question to spark comment debates (e.g., 'Did bro actually cook or is he tweaking? 👇').\n"
        f"- 'hashtags': Exactly 5 hashtags (2 broad like #viral #fyp and 3 specific to {kind}).\n"
        f"- 'keywords': 8 lowercase search phrases US viewers actually search for.\n"
        f"Reply ONLY with JSON:\n"
        f'{{"title": "...", "description": "...", "hashtags": ["#fyp", "..."], "keywords": ["...", "..."]}}'
    )
    d = parse_json(llm(prompt))
    tags = d.get("hashtags")
    tags = " ".join(tags) if isinstance(tags, list) else str(tags or "")
    kws = d.get("keywords")
    kws = ", ".join(kws) if isinstance(kws, list) else str(kws or "")
    title = str(d.get("title") or c.get("hook") or c["title"])[:80]
    out = dict(title=title, description=str(d.get("description") or c["reason"]),
               keywords=kws, hashtags=tags)
    return {k: (mask_text(v) if censor else v) for k, v in out.items()}


def copy_txt(cp):
    return (f"# Auto Clipper v2.0 | Developed by {OWNER_NAME}\n"
            f"# NOT FOR SALE • Free Community Edition (Facebook: {OWNER_NAME})\n"
            f"# ------------------------------------------------------------\n"
            f"TITLE\n{cp['title']}\n\n"
            f"DESCRIPTION\n{cp['description']}\n\n{cp['hashtags']}\n\n"
            f"KEYWORDS / TAGS\n{cp['keywords']}\n")


# -------------------------------------------------------------- pipeline
def run_job(src, kind="gaming", layout="auto", ratio="9:16", cam_pos="top", model=DEFAULT_MODEL,
            captions=True, hook=True, censor=True, music=True, edit=True,
            focus="", cookies="", lang=None, live=0, brand_color="yellow", caption_style="auto",
            top_clips=DEFAULT_TOP_CLIPS_LABEL, font=DEFAULT_FONT,
            layout_mode="auto", caption_template="hormozi", hook_position="top", platform="tiktok"):
    clean_work_dir()
    prepare_env()
    ok, msg = verify_remote_license()
    if not ok:
        raise RuntimeError(f"ACCESS REVOKED BY OWNER ({OWNER_NAME}): {msg}")
    model = parse_model_name(model)
    # layout_mode defaults to layout if not explicitly set
    if layout_mode == "auto" and layout and layout != "auto":
        layout_mode = layout
    layout = parse_layout(layout)
    ratio = parse_ratio(ratio)
    cam_pos = parse_cam_pos(cam_pos)
    live_mins = parse_live_mins(live)
    brand_info = parse_brand_color(brand_color)
    cap_style = parse_caption_style(caption_style)
    # caption_template maps to caption_style if set
    if caption_template and caption_template != "auto":
        cap_style = caption_template
    top_limit = parse_top_clips(top_clips)
    STATE.update(model=model, cookies=cookies or "")
    prof = PROFILES.get(kind, PROFILES["gaming"])
    if cap_style == "auto":
        cap_style = prof.get("cap_style", "hormozi")
    if brand_color in ("yellow", DEFAULT_BRAND_COLOR_LABEL) and "brand" in prof:
        brand_info = parse_brand_color(prof["brand"])
    src = src.strip().strip('"')
    topic, rng = parse_focus(focus)
    job = WORK_DIR / f"job_{int(time.time())}"
    job.mkdir(parents=True, exist_ok=True)
    try:
        log("1/8 Downloading audio only...")
        prog("download", 0.0, "starting...")
        title, raw, wav = fetch_audio(src, job, live_mins=live_mins)
        prog("download", 1.0, "audio ready")
        out_dir = OUT_DIR / f"{safe_name(title)}_{time.strftime('%Y%m%d_%H%M')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        log("2/8 Reading the audio...")
        prog("read", 0.0, "loading audio into memory...")
        import soundfile as sf
        y, _ = sf.read(str(wav), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        dur = len(y) / SR
        rms = rms_profile(y)
        prog("read", 1.0, "done")
        t0 = float(rng[0] or 0.0)
        t1 = min(dur, float(rng[1])) if rng[1] else dur
        if t1 - t0 < 10:
            raise RuntimeError(f"The time range is empty. The video is {hms(dur)} long.")
        if topic:
            log(f'   focus: "{topic}"')
        if rng != [None, None]:
            log(f"   only searching {hms(t0)} - {hms(t1)}")
        span = max(120.0, prof["max"] * 2.2)
        mode = "deep" if (topic or rng != [None, None]) else prof["scan"]
        if prof.get("music"):
            mode = "fast"
        if mode == "deep" and not has_nvidia():
            log("   Warning: no NVIDIA GPU found - reading the whole video on CPU is slow. Be patient.")
        log(f"   scan mode: {mode}  |  searching {(t1 - t0) / 60:.0f} min of audio")

        cands = []
        if prof.get("music"):
            del y
            gc.collect()
            n = max(3, int((t1 - t0) / (prof["max"] * 1.5)))
            for t, sc in find_peaks(rms, n, prof["max"] * 1.5, t0, t1):
                en = min(t1, max(t0, t - prof["max"] * 0.3) + prof["max"])
                cands.append(dict(start=max(t0, en - prof["max"]), end=en, score=hype_0_10(sc), segs=[],
                                  title=title, hook=title, reason="Highest-energy section of the track"))
            prog("transcribe", 1.0, "not needed for music")
            prog("judge", 1.0, "not needed for music")
        else:
            if mode == "deep":
                log("3/8 Transcribing (local Whisper)...")
                m = load_whisper()
                allsegs = transcribe_full(m, y, t0, t1, lang)
                del m
                del y
                gc.collect()
                wins = deep_windows(rms, allsegs, t0, t1, span, prof["story"])
            else:
                wins = []
                n = min(150, max(10, int((t1 - t0) / 90)))
                for t, sc in find_peaks(rms, n, span * 0.5, t0, t1):
                    we = min(t1, max(t0, t - span * 0.62) + span)
                    wins.append(dict(peak=t, hype=hype_0_10(sc), ws=max(t0, we - span), we=we, segs=[]))
                log("3/8 Transcribing the loudest moments (local Whisper)...")
                m = load_whisper()
                for i, w in enumerate(wins, 1):
                    check_cancel()
                    w["segs"] = transcribe(m, y, w["ws"], w["we"], lang)
                    prog("transcribe", i / len(wins), f"moment {i} of {len(wins)}")
                del m
                del y
                gc.collect()
            prog("transcribe", 1.0, "done")
            log(f"   {len(wins)} windows to judge")

            log("4/8 Judging stories and reviewing as an editor (local LLM)... this is the slow step")
            ensure_ollama()
            STATE["model"] = ensure_model(model)
            started = time.time()
            for i, w in enumerate(wins, 1):
                check_cancel()
                res = score_window(kind, prof, w, dur, topic)
                if res and res["score"] >= FLOOR_PRE:
                    res = review_clip(kind, prof, res, dur, topic)
                    cands.append(res)
                eta = (time.time() - started) / i * (len(wins) - i) / 60
                prog("judge", i / len(wins), f"window {i} of {len(wins)}  |  {len(cands)} good clips  |  ~{eta:.0f} min left")
                if i % 10 == 0 or i == len(wins):
                    log(f"   judged {i}/{len(wins)} windows, {len(cands)} good clips so far (~{eta:.0f} min left)")

        cands = [c for c in cands if c["score"] >= FLOOR]
        cands.sort(key=lambda c: -c["score"])
        chosen = []
        for c in cands:
            if all(overlap(c, k) < 0.2 for k in chosen):
                chosen.append(c)
        if not chosen:
            raise RuntimeError("No good clips were found" + (" matching that request." if topic else ".")
                               + " Try a different focus, or a different Content type.")
        if top_limit and len(chosen) > top_limit:
            log(f"   found {len(chosen)} candidate clips -> keeping top {top_limit} best clips")
            chosen = chosen[:top_limit]
        else:
            log(f"   found {len(chosen)} clips (best {chosen[0]['score']:.1f}/10)")

        log("4b/8 Writing SEO titles and descriptions...")
        for k, c in enumerate(chosen, 1):
            check_cancel()
            c["copy"] = make_copy(kind, title, topic, c, censor) if not prof.get("music") else \
                dict(title=title, description=c["reason"], keywords="", hashtags="")
            prog("copy", k / len(chosen), f"clip {k} of {len(chosen)}")
        unload_llm()                        # free VRAM before encoding

        tracks = music_tracks() if music else []
        if music and not tracks:
            log("   (could not create background music, continuing without)")
        results = []
        N = len(chosen)
        cached_cam = None
        for n, c in enumerate(chosen, 1):
            if CANCEL.is_set():
                log("   Stopped. Clips finished so far are saved.")
                break
            s, e = c["start"], c["end"]
            d = e - s
            try:
                prog("render", (n - 1) / N, f"clip {n} of {N}: downloading video")
                log(f"5/8 Clip {n}/{len(chosen)} (score {c['score']:.1f}): downloading video {hms(s)}-{hms(e)}...")
                vid, vstart = get_video_section(src, s, e, job, n)
                sw, sh, sfps = probe_video(str(vid))
                lay = resolve_layout(layout, prof, sw, sh)
                W, H = canvas_for(lay, sw, sh, ratio=ratio)
                fps = min(60.0, round(sfps))
                cam = None
                if lay == "split":
                    if cached_cam is None:
                        log(f"6/8 Clip {n}: detecting facecam...")
                        PH = ev(H / 3)
                        detected = detect_facecam(str(vid), vstart, d, job, n, aspect=W / max(1, PH))
                        if detected:
                            cached_cam = detected
                            log(f"   facecam locked: {cached_cam}")
                        else:
                            # Streamer corner webcam fallback: top-left corner crop so layout stays split!
                            cached_cam = (0, 0, int(0.28 * sw), int(0.32 * sh))
                            log(f"   (facecam locked to top-left streamer corner crop)")
                    cam = cached_cam

                words = [dict(w=x["w"], s=x["s"] - s, e=x["e"] - s)
                         for sg in c["segs"] for x in sg["words"] if x["s"] >= s - 0.1 and x["e"] <= e + 0.3]
                zoom_on = bool(edit and prof["zoom"] and lay != "split")
                pieces = plan_cuts(words, d, prof["gap"] if edit else None, split=zoom_on)
                zooms = [ZOOMS[i % len(ZOOMS)] if zoom_on else 1.0 for i in range(len(pieces))]
                ndur = sum(b - a for a, b in pieces)
                new_words = [dict(w=x["w"], s=remap_time(x["s"], pieces), e=remap_time(x["e"], pieces))
                             for x in words]
                raw_clips_dir = out_dir / "raw_clips"
                raw_clips_dir.mkdir(parents=True, exist_ok=True)
                raw_vid_saved = raw_clips_dir / f"clip_{n:02d}_raw.mp4"
                if not raw_vid_saved.exists() and Path(vid).exists():
                    try:
                        shutil.copyfile(str(vid), str(raw_vid_saved))
                    except Exception:
                        pass
                active_vid_path = str(raw_vid_saved) if raw_vid_saved.exists() else str(vid)

                ass = out_dir / f"clip_{n:02d}.ass"
                hook_text = c.get("hook") if hook else ""

                build_ass(new_words, ass, censor, hook_text, ndur,
                          show_captions=captions, show_hook=hook, W=W, H=H,
                          brand_dict=brand_info, style_name=cap_style, layout=lay, cam_pos=cam_pos,
                          has_hook_image=False, font_name=font)
                bleeps = bleep_intervals(new_words, ndur) if censor else []
                log(f"   {lay} {W}x{H} ({ratio}), {len(pieces)} cuts, cut {d - ndur:.0f}s of dead air"
                    + (f", {len(bleeps)} bleep(s)" if bleeps else ""))

                prog("render", (n - 1 + 0.4) / N, f"clip {n} of {N}: Stage 1 Auto-Reframe (1080x1920)")
                log(f"6/8 Clip {n}: Stage 1 Auto-Reframe & audio mix to 1080x1920...")
                raw_reframed_mp4 = out_dir / f"clip_{n:02d}_reframed.mp4"
                mp4 = out_dir / f"clip_{n:02d}.mp4"
                audio_src = str(raw) if re.match(r"https?://", src) else src
                chosen_track = random.choice(tracks) if tracks else None

                # Compose audio cuts, music ducking, and video base (without old hook overlay)
                ndur = compose(job, active_vid_path, vstart, audio_src, s, d, None, lay, cam, str(raw_reframed_mp4), W, H, fps,
                               pieces, zooms, sw, sh, bleeps, chosen_track,
                               cam_pos=cam_pos, hook_img_path=None,
                               hook_banner_dim=None, fast_draft=True)

                # Ensure guaranteed 1080x1920 standardization through the 4-way layout classifier & reframer
                from app.services.reframer import reframe_to_vertical, detect_layout
                stage1_layout = detect_layout(str(raw_reframed_mp4), layout_override=layout_mode)
                log(f"   [Reframe Router] Selected engine: '{stage1_layout}' (1080x1920 standardized)")
                temp_1080x1920 = out_dir / f"clip_{n:02d}_1080x1920.mp4"
                reframe_to_vertical(str(raw_reframed_mp4), str(temp_1080x1920), layout_mode=stage1_layout)

                # Stage 2: Styled Template Caption & Hook Rendering via pycaps
                hook_text_to_render = c.get("hook") if hook else None
                if (captions and new_words) or hook_text_to_render:
                    log(f"7/8 Clip {n}: Stage 2 pycaps Styled Captions & Hook ({cap_style})...")
                    prog("render", (n - 1 + 0.7) / N, f"clip {n} of {N}: rendering pycaps captions & hook")
                    from app.services.caption_renderer import render_styled_captions
                    pycaps_words = [dict(text=w_item["w"], start=w_item["s"], end=w_item["e"]) for w_item in new_words] if (captions and new_words) else []
                    render_styled_captions(
                        str(temp_1080x1920),
                        str(mp4),
                        pycaps_words,
                        template_id=cap_style,
                        layout_mode=stage1_layout,
                        brand_color=brand_info.get("hex") if isinstance(brand_info, dict) else None,
                        hook_text=hook_text_to_render,
                        hook_position=hook_position,
                    )
                    # Clean intermediate temp reframed file
                    temp_1080x1920.unlink(missing_ok=True)
                    raw_reframed_mp4.unlink(missing_ok=True)
                else:
                    if temp_1080x1920.exists():
                        if mp4.exists():
                            mp4.unlink()
                        temp_1080x1920.rename(mp4)
                    elif raw_reframed_mp4.exists():
                        if mp4.exists():
                            mp4.unlink()
                        raw_reframed_mp4.rename(mp4)
                cp = c["copy"]
                (out_dir / f"clip_{n:02d}.txt").write_text(copy_txt(cp), encoding="utf-8")
                results.append(dict(file=mp4.name, rank=n, score=round(c["score"], 1), title=cp["title"],
                                    description=cp["description"], hashtags=cp["hashtags"],
                                    keywords=cp["keywords"], hook=c.get("hook", ""), reason=c["reason"],
                                    start=round(s, 1), end=round(e, 1), duration=round(ndur, 1),
                                    font=font, cap_style=cap_style, brand_color_name=brand_info.get("name", "Yellow"),
                                    cap_pos="bottom", font_scale=1.0, exported=False,
                                    vstart=vstart, astart=s, d=d, lay=lay, cam=cam, cam_pos=cam_pos,
                                    W=W, H=H, fps=fps, pieces=pieces, zooms=zooms, sw=sw, sh=sh,
                                    bleeps=bleeps, track=str(chosen_track) if chosen_track else None,
                                    audio_src=audio_src, new_words=new_words, raw_vid=str(raw_vid_saved),
                                    hook_dim=list(hook_dim) if hook_dim else None))
                (out_dir / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False),
                                                      encoding="utf-8")
                prog("render", n / N, f"clip {n} of {N} done")
                log(f"   clip_{n:02d} done: {cp['title']}")
                if n == 1:
                    log(f"\x00POP_RESULTS:{out_dir}")
                else:
                    log(f"\x00REFRESH_RESULTS:{out_dir}")
            except Cancelled:
                log("   Stopped. Clips finished so far are saved.")
                break
            except Exception as ex:
                import subprocess as _sp
                tb = traceback.format_exc()
                detail = str(ex)
                # CalledProcessError: also dump stderr so FFmpeg failure is visible
                if isinstance(ex, _sp.CalledProcessError):
                    stderr_txt = (ex.stderr or b"")
                    if isinstance(stderr_txt, bytes):
                        stderr_txt = stderr_txt.decode("utf-8", errors="replace")
                    detail = f"returncode={ex.returncode}\n{stderr_txt[-800:]}"
                err_path = out_dir / f"clip_{n:02d}_error.txt"
                err_path.write_text(f"{tb}\n---\n{detail}", encoding="utf-8")
                log(f"   Clip {n} FAILED, skipping: {detail[-400:]}")
                log(f"   Full error saved to: {err_path.name}")
            finally:
                for f in job.glob(f"clip{n}_src.*"):
                    f.unlink(missing_ok=True)
        log(f"DONE. {len(results)} of {len(chosen)} clips ready in: {out_dir}")
        return out_dir
    finally:
        shutil.rmtree(job, ignore_errors=True)
        unload_llm()


# ------------------------------------------------------------------- GUI
BG, PANEL, PANEL2, FIELD = "#0e1015", "#161922", "#202536", "#0b0d12"
TEXT, MUTED, BORDER = "#e8eaf0", "#8a93a8", "#2a2f40"
ACCENT, ACCENT_H, DANGER, GOOD = "#7c5cff", "#9277ff", "#ef4444", "#22c55e"
FONT = "Segoe UI"


def score_color(s, light=0.30):
    """Red (weak) -> yellow -> green (strong) colour for a 1-10 score, tuned for a dark UI."""
    t = max(0.0, min(1.0, (float(s) - 5.0) / 3.5))
    r, g, b = colorsys.hls_to_rgb(t * 120 / 360, light, 0.60)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def apply_theme(root):
    from tkinter import ttk
    root.configure(bg=BG)
    st = ttk.Style(root)
    st.theme_use("clam")
    st.configure(".", background=BG, foreground=TEXT, font=(FONT, 10), bordercolor=BORDER,
                 lightcolor=BORDER, darkcolor=BORDER, troughcolor=FIELD)
    st.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT, padding=7,
                 bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    st.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)], darkcolor=[("focus", ACCENT)])
    st.configure("TCombobox", fieldbackground=FIELD, background=PANEL2, foreground=TEXT, arrowcolor=TEXT,
                 padding=6, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    st.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)],
           selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)],
           bordercolor=[("focus", ACCENT)])
    root.option_add("*TCombobox*Listbox.background", PANEL2)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.font", (FONT, 10))
    st.configure("Card.TCheckbutton", background=PANEL, foreground=TEXT, focuscolor=PANEL,
                 indicatorbackground=FIELD, indicatorforeground="#ffffff")
    st.map("Card.TCheckbutton", background=[("active", PANEL)], indicatorbackground=[("selected", ACCENT)],
           indicatorcolor=[("selected", ACCENT)])
    st.configure("Main.Horizontal.TProgressbar", troughcolor=FIELD, background=ACCENT, bordercolor=FIELD,
                 lightcolor=ACCENT, darkcolor=ACCENT, thickness=16)
    st.configure("Step.Horizontal.TProgressbar", troughcolor=FIELD, background="#3b82f6", bordercolor=FIELD,
                 lightcolor="#3b82f6", darkcolor="#3b82f6", thickness=9)
    st.configure("Vertical.TScrollbar", background=PANEL2, troughcolor=BG, bordercolor=BG, arrowcolor=TEXT,
                 lightcolor=PANEL2, darkcolor=PANEL2)
    st.configure("Clips.Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT, rowheight=30,
                 borderwidth=0, font=(FONT, 10))
    st.configure("Clips.Treeview.Heading", background=PANEL2, foreground=MUTED, relief="flat",
                 font=(FONT, 9, "bold"), padding=6)
    st.map("Clips.Treeview.Heading", background=[("active", PANEL2)])
    st.map("Clips.Treeview", background=[("selected", ACCENT)], foreground=[("selected", "#ffffff")])
    return st


def flat_button(parent, text, command, kind="secondary", width=None):
    import tkinter as tk
    colors = dict(primary=(ACCENT, ACCENT_H, "#ffffff"), danger=(DANGER, "#f87171", "#ffffff"),
                  secondary=(PANEL2, "#2b3148", TEXT))
    bg, hover, fg = colors[kind]
    b = tk.Button(parent, text=text, command=command, bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
                  relief="flat", bd=0, cursor="hand2", font=(FONT, 10, "bold"), padx=16, pady=9,
                  highlightthickness=0, disabledforeground="#59607a", width=width)
    b.bind("<Enter>", lambda _e: b.config(bg=hover) if str(b["state"]) == "normal" else None)
    b.bind("<Leave>", lambda _e: b.config(bg=bg) if str(b["state"]) == "normal" else None)
    b._base = bg
    return b


def card(parent, title=None):
    import tkinter as tk
    f = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
    inner = tk.Frame(f, bg=PANEL)
    inner.pack(fill="both", expand=True, padx=16, pady=12)
    if title:
        tk.Label(inner, text=title.upper(), bg=PANEL, fg=MUTED, font=(FONT, 8, "bold")).pack(anchor="w", pady=(0, 6))
    return f, inner


def show_results(root, folder):
    import tkinter as tk
    from tkinter import messagebox, ttk
    import cv2
    from PIL import Image, ImageTk
    import subprocess
    import threading

    folder = Path(folder)
    try:
        data = json.loads((folder / "results.json").read_text(encoding="utf-8"))
    except Exception:
        messagebox.showinfo("Results", "No results.json in that folder yet.")
        return None
    data.sort(key=lambda c: -c["score"])

    win = tk.Toplevel(root)
    win.title(f"Clips Studio - {folder.name} • By {OWNER_NAME} (NOT FOR SALE)")
    win.configure(bg=BG)
    win.geometry("1280x800")
    win.minsize(1100, 720)
    root._results_win = win

    head = tk.Frame(win, bg=BG)
    head.pack(fill="x", padx=18, pady=(14, 6))
    count_lbl = tk.Label(head, text=f"{len(data)} clips", font=(FONT, 16, "bold"), bg=BG, fg=TEXT)
    count_lbl.pack(side="left")
    tk.Label(head, text="  best first", font=(FONT, 10), bg=BG, fg=MUTED).pack(side="left", pady=(6, 0))
    tk.Label(head, text=f"  •  Created by {OWNER_NAME} (NOT FOR SALE)", font=(FONT, 9, "bold"), bg=BG, fg="#a99bff").pack(side="left", padx=(12, 0), pady=(6, 0))
    for txt_l, sc in (("strong", 8.5), ("ok", 6.5), ("weak", 5.0)):
        tk.Label(head, text=f"  {txt_l}  ", bg=score_color(sc), fg="#ffffff", font=(FONT, 9, "bold")
                 ).pack(side="right", padx=2)

    body = tk.PanedWindow(win, orient="horizontal", sashwidth=8, bg=BG, bd=0, sashrelief="flat")
    body.pack(fill="both", expand=True, padx=18, pady=(0, 16))

    left = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
    body.add(left, width=460)
    tree = ttk.Treeview(left, columns=("n", "score", "title", "len", "status"), show="headings", selectmode="browse",
                        style="Clips.Treeview")
    for col, txt_c, w in (("n", "#", 36), ("score", "SCORE", 60), ("title", "TITLE", 230), ("len", "LEN", 55), ("status", "STATUS", 75)):
        tree.heading(col, text=txt_c)
        tree.column(col, width=w, anchor="w" if col == "title" else "center")
    sb = ttk.Scrollbar(left, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")
    for i, c in enumerate(data):
        st_text = "Exported" if c.get("exported") else "Draft"
        tree.insert("", "end", iid=str(i), tags=(f"t{i}",),
                    values=(c["rank"], f"{c['score']:.1f}", c["title"], f"{c['duration']:.0f}s", st_text))
        tree.tag_configure(f"t{i}", background=score_color(c["score"]), foreground="#ffffff")

    right = tk.Frame(body, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
    body.add(right)

    r_content = tk.Frame(right, bg=PANEL)
    r_content.pack(fill="both", expand=True, padx=14, pady=12)

    # In-App Video Player Column
    p_box = tk.Frame(r_content, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
    p_box.pack(side="left", fill="y", padx=(0, 14))

    canvas = tk.Canvas(p_box, width=240, height=426, bg="#000000", highlightthickness=0)
    canvas.pack(padx=10, pady=(10, 6))

    ctrl_frame = tk.Frame(p_box, bg=PANEL)
    ctrl_frame.pack(fill="x", padx=10, pady=(2, 4))

    play_btn = flat_button(ctrl_frame, "▶ Play", None, "primary", width=7)
    play_btn.pack(side="left")

    replay_btn = flat_button(ctrl_frame, "⏮", None, "secondary", width=3)
    replay_btn.pack(side="left", padx=(6, 0))

    time_lbl = tk.Label(ctrl_frame, text="00:00 / 00:00", bg=PANEL, fg=MUTED, font=(FONT, 9))
    time_lbl.pack(side="right")

    scrubber_var = tk.DoubleVar(value=0.0)
    scrubber = ttk.Scale(p_box, from_=0.0, to=100.0, variable=scrubber_var, orient="horizontal")
    scrubber.pack(fill="x", padx=10, pady=(4, 10))

    # Details & Editor Column
    info_box = tk.Frame(r_content, bg=PANEL)
    info_box.pack(side="left", fill="both", expand=True)

    tl = tk.Label(info_box, text="Click a clip on the left", font=(FONT, 13, "bold"), wraplength=480, justify="left",
                  anchor="w", bg=PANEL, fg=TEXT, padx=10, pady=8)
    tl.pack(fill="x")
    meta = tk.Label(info_box, text="", fg=MUTED, bg=PANEL, anchor="w", justify="left", wraplength=480, font=(FONT, 9))
    meta.pack(fill="x", pady=(2, 4))

    # Notebook: Tab 1 = Editor Studio (Hook, Fonts, Styles, Export), Tab 2 = SEO Copy
    nb = ttk.Notebook(info_box)
    nb.pack(fill="both", expand=True, pady=(2, 6))

    tab_edit = tk.Frame(nb, bg=FIELD, padx=12, pady=10)
    tab_seo = tk.Frame(nb, bg=FIELD, padx=12, pady=10)
    nb.add(tab_edit, text="  ✏️ Edit Clip & Styling  ")
    nb.add(tab_seo, text="  📝 SEO Copy & Tags  ")

    # --- TAB 1: EDIT CLIP & STYLING
    tk.Label(tab_edit, text="ON-SCREEN HOOK TEXT (EDITABLE BANNER):", bg=FIELD, fg="#9277ff", font=(FONT, 8, "bold")).pack(anchor="w")
    hook_ent = ttk.Entry(tab_edit)
    hook_ent.pack(fill="x", pady=(2, 8))

    e_grid = tk.Frame(tab_edit, bg=FIELD)
    e_grid.pack(fill="x", pady=(0, 6))

    def e_combo(r, c, lbl, vals, default, colspan=1):
        f = tk.Frame(e_grid, bg=FIELD)
        f.grid(row=r, column=c, columnspan=colspan, sticky="we", padx=(0, 8), pady=(0, 6))
        e_grid.columnconfigure(c, weight=1)
        tk.Label(f, text=lbl, bg=FIELD, fg=MUTED, font=(FONT, 8)).pack(anchor="w")
        cb = ttk.Combobox(f, values=vals, state="readonly")
        cb.set(default)
        cb.pack(fill="x", pady=(2, 0))
        return cb

    e_font = e_combo(0, 0, "Font Choice (12 Viral Fonts)", FONTS, DEFAULT_FONT)
    e_style = e_combo(0, 1, "Caption Animation Style", CAPTION_STYLE_CHOICES, DEFAULT_CAPTION_STYLE_LABEL)
    e_color = e_combo(1, 0, "Brand Accent Color", BRAND_COLOR_CHOICES, DEFAULT_BRAND_COLOR_LABEL)
    e_pos = e_combo(1, 1, "Caption Position", CAP_POS_CHOICES, DEFAULT_CAP_POS_LABEL)
    e_scale = e_combo(2, 0, "Text Size Scale", FONT_SCALE_CHOICES, DEFAULT_FONT_SCALE_LABEL)

    status_lbl = tk.Label(tab_edit, text="", bg=FIELD, fg="#4ade80", font=(FONT, 9, "bold"))
    status_lbl.pack(anchor="w", pady=(4, 6))

    e_btn_bar = tk.Frame(tab_edit, bg=FIELD)
    e_btn_bar.pack(fill="x", pady=(4, 0))

    apply_btn = flat_button(e_btn_bar, "🔄 Apply / Update Preview", None, "secondary")
    apply_btn.pack(side="left", padx=(0, 8))

    export_btn = flat_button(e_btn_bar, "🚀 Export Final MP4", None, "primary")
    export_btn.pack(side="left", padx=(0, 8))

    export_all_btn = flat_button(e_btn_bar, "📦 Export All Final MP4s", None, "secondary")
    export_all_btn.pack(side="left")

    # --- TAB 2: SEO COPY & TAGS
    txt = tk.Text(tab_seo, wrap="word", height=10, font=(FONT, 10), bg=FIELD, fg=TEXT, relief="flat", padx=10, pady=8,
                  insertbackground=TEXT, highlightthickness=1, highlightbackground=BORDER, state="disabled")
    txt.pack(fill="both", expand=True, pady=(0, 6))

    seo_bar = tk.Frame(tab_seo, bg=FIELD)
    seo_bar.pack(fill="x")

    def copy_text(what):
        c = cur["c"]
        if c:
            win.clipboard_clear()
            win.clipboard_append(c["title"] if what == "title" else f"{c['description']}\n\n{c['hashtags']}")

    flat_button(seo_bar, "Copy Title", lambda: copy_text("title"), "secondary").pack(side="left", padx=(0, 8))
    flat_button(seo_bar, "Copy Description", lambda: copy_text("desc"), "secondary").pack(side="left")

    # Bottom Actions Bar
    bar = tk.Frame(info_box, bg=PANEL)
    bar.pack(fill="x", pady=(6, 0))

    cur = {"c": None, "i": None}
    play_state = {
        "cap": None,
        "fps": 30.0,
        "dur": 0.0,
        "playing": False,
        "start_wall": 0.0,
        "start_pos": 0.0,
        "audio_proc": None,
        "loop_id": None,
        "cur_file": None,
    }

    def stop_playback():
        if play_state["loop_id"]:
            win.after_cancel(play_state["loop_id"])
            play_state["loop_id"] = None
        play_state["playing"] = False
        play_btn.config(text="▶ Play")
        if play_state["audio_proc"]:
            try:
                play_state["audio_proc"].kill()
            except Exception:
                pass
            play_state["audio_proc"] = None

    def render_frame_at(pos_sec):
        cap = play_state["cap"]
        if not cap or not cap.isOpened():
            return
        cap.set(cv2.CAP_PROP_POS_MSEC, pos_sec * 1000.0)
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(240 / max(1, w), 426 / max(1, h))
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
            img = Image.new("RGB", (240, 426), (0, 0, 0))
            px = (240 - nw) // 2
            py = (426 - nh) // 2
            img.paste(Image.fromarray(rgb), (px, py))
            photo = ImageTk.PhotoImage(img)
            canvas.create_image(120, 213, image=photo)
            canvas.image = photo

    def update_loop():
        if not play_state["playing"]:
            return
        elapsed = time.time() - play_state["start_wall"]
        pos = play_state["start_pos"] + elapsed
        dur = play_state["dur"]
        if pos >= dur:
            stop_playback()
            scrubber_var.set(dur)
            render_frame_at(dur)
            time_lbl.config(text=f"{hms(dur)} / {hms(dur)}")
            return
        scrubber_var.set(pos)
        render_frame_at(pos)
        time_lbl.config(text=f"{hms(pos)} / {hms(dur)}")
        play_state["loop_id"] = win.after(33, update_loop)

    def toggle_play():
        if not cur["c"]:
            return
        if play_state["playing"]:
            stop_playback()
        else:
            curr_pos = scrubber_var.get()
            if curr_pos >= play_state["dur"] - 0.2:
                curr_pos = 0.0
            play_state["playing"] = True
            play_state["start_wall"] = time.time()
            play_state["start_pos"] = curr_pos
            play_btn.config(text="⏸ Pause")
            mp4_path = str(folder / cur["c"]["file"])
            try:
                flags = 0x08000000 if os.name == 'nt' else 0
                ffplay_bin = get_tool_path("ffplay")
                play_state["audio_proc"] = subprocess.Popen(
                    [ffplay_bin, "-nodisp", "-autoexit", "-ss", f"{curr_pos:.2f}", mp4_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags
                )
            except Exception:
                play_state["audio_proc"] = None
            update_loop()

    def replay():
        if not cur["c"]:
            return
        stop_playback()
        scrubber_var.set(0.0)
        render_frame_at(0.0)
        time_lbl.config(text=f"00:00 / {hms(play_state['dur'])}")
        toggle_play()

    def on_scrub(val):
        if not cur["c"] or not play_state["cap"]:
            return
        pos = float(val)
        if play_state["playing"]:
            stop_playback()
        render_frame_at(pos)
        time_lbl.config(text=f"{hms(pos)} / {hms(play_state['dur'])}")

    play_btn.config(command=toggle_play)
    replay_btn.config(command=replay)
    scrubber.config(command=on_scrub)

    def pick(_=None):
        sel = tree.selection()
        if not sel:
            return
        if cur["i"] is not None and cur["i"] < len(data):
            tree.tag_configure(f"t{cur['i']}", background=score_color(data[cur['i']]["score"]))
        cur["i"] = int(sel[0])
        c = data[cur["i"]]
        cur["c"] = c
        tree.tag_configure(f"t{cur['i']}", background=score_color(c["score"], 0.46))
        tl.config(text=c["title"], bg=score_color(c["score"]), fg="#ffffff")
        st_label = "✅ Exported MP4" if c.get("exported") else "📝 Draft Mode (Editable)"
        meta.config(text=f"Score {c['score']:.1f}/10   |   {c['duration']:.0f}s   |   Status: {st_label}\nSource: {hms(c['start'])} - {hms(c['end'])}   |   File: {c['file']}")

        # Populate editor fields with current clip settings
        hook_ent.delete(0, "end")
        hook_ent.insert(0, c.get("hook", ""))
        e_font.set(c.get("font", DEFAULT_FONT))
        e_style.set(c.get("cap_style", DEFAULT_CAPTION_STYLE_LABEL))
        e_color.set(c.get("brand_color_name", DEFAULT_BRAND_COLOR_LABEL))
        e_pos.set(c.get("cap_pos_label", DEFAULT_CAP_POS_LABEL))
        e_scale.set(c.get("font_scale_label", DEFAULT_FONT_SCALE_LABEL))
        status_lbl.config(text="")

        txt.config(state="normal")
        txt.delete("1.0", "end")
        txt.insert("end", f"DESCRIPTION\n{c['description']}\n\n{c['hashtags']}\n\nKEYWORDS / TAGS\n{c['keywords']}")
        txt.config(state="disabled")

        stop_playback()
        if play_state["cap"]:
            try:
                play_state["cap"].release()
            except Exception:
                pass
        mp4_path = str(folder / c["file"])
        if Path(mp4_path).exists():
            cap = cv2.VideoCapture(mp4_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
            dur = c.get("duration", frames / fps)
            play_state.update(cap=cap, fps=fps, dur=dur, cur_file=mp4_path)
            scrubber.config(to=max(1.0, dur))
            scrubber_var.set(0.0)
            render_frame_at(0.0)
            time_lbl.config(text=f"00:00 / {hms(dur)}")
        else:
            time_lbl.config(text="Rendering...")

    tree.bind("<<TreeviewSelect>>", pick)

    def apply_edits():
        c = cur["c"]
        if not c:
            return
        c["hook"] = hook_ent.get().strip()
        c["font"] = e_font.get()
        c["cap_style"] = parse_caption_style(e_style.get())
        c["brand_color_name"] = e_color.get()
        c["brand_color"] = parse_brand_color(e_color.get())
        c["cap_pos_label"] = e_pos.get()
        c["cap_pos"] = parse_cap_pos(e_pos.get())
        c["font_scale_label"] = e_scale.get()
        c["font_scale"] = parse_font_scale(e_scale.get())

        n = c["rank"]
        ass_path = folder / f"clip_{n:02d}.ass"
        hook_img_path = folder / f"hook_{n:02d}.png"
        W = c.get("W", 1080)
        H = c.get("H", 1920)

        hook_dim = None
        if c["hook"]:
            try:
                hook_dim = create_hook_banner(c["hook"], hook_img_path, W, H, brand_dict=c["brand_color"],
                                              font_name=c["font"], font_scale=c["font_scale"])
            except Exception:
                hook_dim = None
        c["hook_dim"] = list(hook_dim) if hook_dim else None

        new_words = c.get("new_words", [])
        build_ass(new_words, ass_path, True, c["hook"], c.get("duration", 30.0),
                  show_captions=True, show_hook=bool(c["hook"]), W=W, H=H,
                  brand_dict=c["brand_color"], style_name=c["cap_style"],
                  layout=c.get("lay", "split"), cam_pos=c.get("cam_pos", "top"),
                  has_hook_image=(hook_dim is not None), font_name=c["font"],
                  font_scale=c["font_scale"], cap_pos=c["cap_pos"])

        # Update fast preview draft
        raw_vid = c.get("raw_vid")
        if raw_vid and Path(raw_vid).exists():
            status_lbl.config(text="Updating preview...", fg="#fbbf24")
            win.update_idletasks()
            mp4_out = folder / c["file"]
            try:
                compose(folder, raw_vid, c.get("vstart", 0.0), c.get("audio_src", raw_vid),
                        c.get("astart", 0.0), c.get("d", 30.0), ass_path, c.get("lay", "split"),
                        c.get("cam"), str(mp4_out), W, H, c.get("fps", 30.0),
                        c.get("pieces", [[0, 30]]), c.get("zooms", [1.0]), c.get("sw", W), c.get("sh", H),
                        c.get("bleeps", []), c.get("track"), cam_pos=c.get("cam_pos", "top"),
                        hook_img_path=hook_img_path if hook_dim else None,
                        hook_banner_dim=hook_dim, fast_draft=True)
            except Exception as ex:
                status_lbl.config(text=f"Preview error: {str(ex)[:50]}", fg="#f87171")
                return

        (folder / "results.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        status_lbl.config(text="✨ Preview updated with custom font & styling!", fg="#4ade80")
        pick()

    def export_final(clip_item=None, silent=False):
        c = clip_item or cur["c"]
        if not c:
            return
        status_lbl.config(text=f"Exporting final high-quality MP4 for clip #{c['rank']}...", fg="#38bdf8")
        win.update_idletasks()
        n = c["rank"]
        ass_path = folder / f"clip_{n:02d}.ass"
        hook_img_path = folder / f"hook_{n:02d}.png"
        raw_vid = c.get("raw_vid")
        if not raw_vid or not Path(raw_vid).exists():
            raw_vid = str(folder / f"clip_{n:02d}.mp4")
        W = c.get("W", 1080)
        H = c.get("H", 1920)
        mp4_out = folder / c["file"]
        hook_dim = c.get("hook_dim")

        try:
            compose(folder, raw_vid, c.get("vstart", 0.0), c.get("audio_src", raw_vid),
                    c.get("astart", 0.0), c.get("d", 30.0), ass_path, c.get("lay", "split"),
                    c.get("cam"), str(mp4_out), W, H, c.get("fps", 30.0),
                    c.get("pieces", [[0, 30]]), c.get("zooms", [1.0]), c.get("sw", W), c.get("sh", H),
                    c.get("bleeps", []), c.get("track"), cam_pos=c.get("cam_pos", "top"),
                    hook_img_path=hook_img_path if hook_dim else None,
                    hook_banner_dim=hook_dim, fast_draft=False)
            c["exported"] = True
            (folder / "results.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            if not silent:
                status_lbl.config(text=f"✅ Final MP4 Exported Successfully: {c['file']}", fg="#4ade80")
                refresh(folder)
        except Exception as ex:
            if not silent:
                status_lbl.config(text=f"Export error: {str(ex)[:60]}", fg="#f87171")

    def export_all():
        def run_all():
            for c in data:
                export_final(c, silent=True)
            win.after(0, lambda: [status_lbl.config(text=f"🎉 All {len(data)} clips exported successfully!", fg="#4ade80"), refresh(folder)])
        status_lbl.config(text=f"Exporting all {len(data)} clips in background...", fg="#38bdf8")
        threading.Thread(target=run_all, daemon=True).start()

    apply_btn.config(command=apply_edits)
    export_btn.config(command=lambda: export_final())
    export_all_btn.config(command=export_all)

    flat_button(bar, "Open in VLC / Fullscreen", lambda: cur["c"] and os.startfile(str(folder / cur["c"]["file"])), "primary").pack(side="left", padx=(0, 8))
    flat_button(bar, "Open Folder", lambda: os.startfile(str(folder)), "secondary").pack(side="left")

    def on_close():
        stop_playback()
        if play_state["cap"]:
            try:
                play_state["cap"].release()
            except Exception:
                pass
            play_state["cap"] = None
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", on_close)

    def refresh(f):
        nonlocal data
        try:
            new_data = json.loads((Path(f) / "results.json").read_text(encoding="utf-8"))
            new_data.sort(key=lambda c: -c["score"])
            data = new_data
        except Exception:
            return
        count_lbl.config(text=f"{len(data)} clips")
        existing_iids = set(tree.get_children())
        for i, c in enumerate(data):
            iid = str(i)
            st_text = "Exported" if c.get("exported") else "Draft"
            vals = (c["rank"], f"{c['score']:.1f}", c["title"], f"{c['duration']:.0f}s", st_text)
            if iid in existing_iids:
                tree.item(iid, values=vals)
            else:
                tree.insert("", "end", iid=iid, tags=(f"t{i}",), values=vals)
                tree.tag_configure(f"t{i}", background=score_color(c["score"]), foreground="#ffffff")
        if not cur["c"] and data:
            tree.selection_set("0")
            tree.focus("0")
            pick()

    win.refresh = refresh

    if data:
        tree.selection_set("0")
        tree.focus("0")
        pick()
    return win


def gui():
    """Modern studio workstation UI powered by HTML/Tailwind with seamless fallback to Tkinter."""
    html_file = resolve_resource("web", "index.html")
    if html_file.exists():
        try:
            # Optimize WebView2 for lightweight startup on low-end hardware
            existing_args = os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "")
            perf_args = (
                "--disable-features=Translate,OptimizationHints,MediaEngagementBypassAutoplayPolicies "
                "--enable-low-end-device-mode "
                "--disable-background-timer-throttling "
                "--disable-renderer-backgrounding"
            )
            os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = f"{perf_args} {existing_args}".strip()

            import webview

            class StudioAPI:
                def __init__(self, window=None):
                    self.window = window
                    self.last_dir = None

                def browse_file(self):
                    res = self.window.create_file_dialog(
                        webview.OPEN_DIALOG,
                        file_types=("Video files (*.mp4;*.mkv;*.mov;*.webm)", "All files (*.*)")
                    )
                    if res and len(res) > 0:
                        return res[0]
                    return ""

                def open_output_folder(self):
                    target = self.last_dir or OUT_DIR
                    if not target.exists():
                        target.mkdir(parents=True, exist_ok=True)
                    os.startfile(str(target))
                    return True

                def open_file(self, filename):
                    target = self.last_dir or OUT_DIR
                    f = target / filename
                    if f.exists():
                        os.startfile(str(f))
                        return True
                    return False

                def get_latest_results(self):
                    target = self.last_dir
                    if not target:
                        # find newest subfolder in OUT_DIR
                        subdirs = sorted([d for d in OUT_DIR.glob("*") if d.is_dir()], key=os.path.getmtime, reverse=True)
                        if subdirs:
                            target = subdirs[0]
                    if target and (target / "results.json").exists():
                        try:
                            return json.loads((target / "results.json").read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    return []

                def stop_clipping(self):
                    CANCEL.set()
                    return True

                def start_clipping(self, opts):
                    CANCEL.clear()
                    src = opts.get("src", "").strip()
                    if not src:
                        return False

                    def run_thread():
                        global log, prog

                        def studio_log(m):
                            s = str(m)
                            if s.startswith("\x00POP_RESULTS:") or s.startswith("\x00REFRESH_RESULTS:"):
                                f = Path(s.split(":", 1)[1].strip())
                                self.last_dir = f
                                if (f / "results.json").exists():
                                    try:
                                        data = json.loads((f / "results.json").read_text(encoding="utf-8"))
                                        escaped_json = json.dumps(data)
                                        self.window.evaluate_js(f"window.studioReceiveResults({escaped_json});")
                                    except Exception:
                                        pass
                                return
                            escaped = json.dumps(s)
                            self.window.evaluate_js(f"window.studioReceiveLog({escaped});")

                        def studio_prog(stage, frac=0.0, text=""):
                            esc_stage = json.dumps(str(stage))
                            esc_text = json.dumps(str(text))
                            self.window.evaluate_js(f"window.studioReceiveProgress({esc_stage}, {frac}, {esc_text});")

                        log = studio_log
                        prog = studio_prog

                        try:
                            clean_opts = dict(
                                kind=opts.get("kind", "gaming"),
                                layout=opts.get("layout", "auto"),
                                ratio=opts.get("ratio", "9:16"),
                                cam_pos=opts.get("cam_pos", "top"),
                                model=opts.get("model", DEFAULT_MODEL),
                                captions=bool(opts.get("captions", True)),
                                hook=bool(opts.get("hook", True)),
                                censor=bool(opts.get("censor", True)),
                                music=bool(opts.get("music", False)),
                                edit=bool(opts.get("edit", True)),
                                focus=opts.get("focus", ""),
                                cookies=opts.get("cookies", ""),
                                live=int(opts.get("live", 0)),
                                brand_color=opts.get("brand_color", "yellow"),
                                caption_style=opts.get("caption_style", "auto"),
                                top_clips=opts.get("top_clips", DEFAULT_TOP_CLIPS_LABEL),
                                font=opts.get("font", DEFAULT_FONT),
                                layout_mode=opts.get("layout_mode", opts.get("layout", "auto")),
                                caption_template=opts.get("caption_template", opts.get("caption_style", "explosive")),
                                hook_position=opts.get("hook_position", "top"),
                                platform=opts.get("platform", "tiktok"),
                            )
                            res_dir = run_job(src, **clean_opts)
                            self.last_dir = res_dir
                            if (res_dir / "results.json").exists():
                                try:
                                    data = json.loads((res_dir / "results.json").read_text(encoding="utf-8"))
                                    self.window.evaluate_js(f"window.studioReceiveResults({json.dumps(data)});")
                                except Exception:
                                    pass
                            self.window.evaluate_js("window.studioJobFinished(true, 'All clips finished and ready in Results Studio!');")
                        except Cancelled:
                            self.window.evaluate_js("window.studioJobFinished(false, 'Job stopped by user.');")
                        except Exception as e:
                            (ROOT / "error_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
                            esc_err = json.dumps(f"ERROR: {e}")
                            self.window.evaluate_js(f"window.studioReceiveLog({esc_err});")
                            self.window.evaluate_js("window.studioJobFinished(false, 'Stopped due to error.');")

                    threading.Thread(target=run_thread, daemon=True).start()
                    return True

            api = StudioAPI()
            win = webview.create_window(
                title=f"Auto Clipper v2.0 • Studio Workstation (Official Free Tool • By {OWNER_NAME})",
                url=str(html_file),
                js_api=api,
                width=1340,
                height=840,
                min_size=(1020, 680),
                background_color="#0B0D10"
            )
            api.window = win
            webview.start(debug=False, private_mode=True)
            return
        except Exception as e:
            print(f"[Warning] Webview could not start ({e}), falling back to standard desktop UI...")

    tkinter_gui()


def tkinter_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)      # crisp text on Windows
    except Exception:
        pass
    prepare_env()
    root = tk.Tk()
    root.title(f"Auto Clipper v2.0 - Created by {OWNER_NAME} [NOT FOR SALE]")
    root.geometry(f"960x{min(880, max(640, root.winfo_screenheight() - 90))}")
    root.minsize(880, 620)
    apply_theme(root)
    q = queue.Queue()
    last = {"dir": None}

    wrap = tk.Frame(root, bg=BG)
    wrap.pack(fill="both", expand=True, padx=22, pady=16)

    # --- Header with creator branding and fast actions
    top = tk.Frame(wrap, bg=BG)
    top.pack(fill="x", pady=(0, 6))

    t_left = tk.Frame(top, bg=BG)
    t_left.pack(side="left")
    tk.Label(t_left, text="Auto Clipper", font=(FONT, 20, "bold"), bg=BG, fg=TEXT).pack(side="left")
    tk.Label(t_left, text=f"  v2.0 • By {OWNER_NAME}", font=(FONT, 10, "bold"), bg=BG, fg="#a99bff").pack(side="left", pady=(5, 0))

    import webbrowser
    def open_facebook():
        webbrowser.open_new_tab(OWNER_FACEBOOK)

    def open_results():
        d = last["dir"]
        if not d or not Path(d).exists():
            d = filedialog.askdirectory(initialdir=str(OUT_DIR if OUT_DIR.exists() else ROOT),
                                        title="Pick a clips folder (the one containing results.json)")
        if d:
            show_results(root, d)

    t_right = tk.Frame(top, bg=BG)
    t_right.pack(side="right")
    flat_button(t_right, "📂 Output Folder",
                lambda: os.startfile(str(OUT_DIR)) if OUT_DIR.exists() else None, "secondary").pack(side="right", padx=(8, 0))
    flat_button(t_right, "🎬 Results Studio", open_results, "secondary").pack(side="right")

    # Anti-resale & ownership badge
    banner = tk.Frame(wrap, bg="#1a1630", highlightbackground="#7c5cff", highlightthickness=1, padx=12, pady=6)
    banner.pack(fill="x", pady=(0, 10))
    tk.Label(banner, text="🛡️ OFFICIAL FREE COMMUNITY TOOL — NOT FOR SALE", font=(FONT, 9, "bold"),
             bg="#1a1630", fg="#22c55e").pack(side="left")
    tk.Label(banner, text=f"  •  Software Owned by {OWNER_NAME}  •  Resale Strictly Prohibited",
             font=(FONT, 9), bg="#1a1630", fg="#c9cfdd").pack(side="left")
    fb_btn = tk.Button(banner, text="👤 Owner: MJ Mark Jonel (Facebook)", command=open_facebook,
                       bg="#3b5998", fg="#ffffff", activebackground="#4c70ba", activeforeground="#ffffff",
                       font=(FONT, 8, "bold"), relief="flat", bd=0, padx=10, pady=3, cursor="hand2")
    fb_btn.pack(side="right")

    # --- 1. Video Source Card
    c1, i1 = card(wrap, "1. Video Source")
    c1.pack(fill="x", pady=(0, 8))
    row = tk.Frame(i1, bg=PANEL)
    row.pack(fill="x")
    url = ttk.Entry(row)
    url.pack(side="left", fill="x", expand=True)

    def browse():
        p = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv *.mov *.webm *.flv *.ts")])
        if p:
            url.delete(0, "end")
            url.insert(0, p)

    flat_button(row, "Browse File", browse, "secondary").pack(side="left", padx=(8, 0))
    tk.Label(i1, text="Paste a YouTube / Twitch link, or pick a local video file (MP4, MKV, MOV, WEBM).",
             bg=PANEL, fg=MUTED, font=(FONT, 9)).pack(anchor="w", pady=(6, 0))

    # --- 2. Settings & Style Card (Tabbed Studio)
    c2, i2 = card(wrap, "2. Clip Settings & Captions Studio")
    c2.pack(fill="x", pady=(0, 8))

    nb_settings = ttk.Notebook(i2)
    nb_settings.pack(fill="x", pady=(0, 4))

    tab_video = tk.Frame(nb_settings, bg=PANEL, padx=12, pady=10)
    tab_captions = tk.Frame(nb_settings, bg=PANEL, padx=12, pady=10)
    nb_settings.add(tab_video, text="  🎬 Video & AI Engine  ")
    nb_settings.add(tab_captions, text="  🎨 Captions & Typography Studio  ")

    def combo(parent_grid, r, c, label, values, default, colspan=1):
        f = tk.Frame(parent_grid, bg=PANEL)
        f.grid(row=r, column=c, columnspan=colspan, sticky="we", padx=(0, 10), pady=(0, 8))
        parent_grid.columnconfigure(c, weight=1)
        tk.Label(f, text=label, bg=PANEL, fg=MUTED, font=(FONT, 9)).pack(anchor="w")
        cb = ttk.Combobox(f, values=values, state="readonly")
        cb.set(default)
        cb.pack(fill="x", pady=(2, 0))
        return cb

    # ===== TAB 1: VIDEO & AI ENGINE =====
    grid_vid = tk.Frame(tab_video, bg=PANEL)
    grid_vid.pack(fill="x")

    kind = combo(grid_vid, 0, 0, "Niche / Content Type", list(PROFILES), "gaming")
    ratio = combo(grid_vid, 0, 1, "Aspect Ratio", RATIO_CHOICES, DEFAULT_RATIO_LABEL)
    layout = combo(grid_vid, 0, 2, "Framing Layout", LAYOUT_CHOICES, DEFAULT_LAYOUT_LABEL)
    top_clips = combo(grid_vid, 0, 3, "Clip Limit", TOP_CLIPS_CHOICES, DEFAULT_TOP_CLIPS_LABEL)

    rec_lbl = tk.Label(tab_video, text=PROFILES["gaming"]["rec"], bg=PANEL, fg="#9277ff",
                       font=(FONT, 9, "italic"), anchor="w", justify="left")
    rec_lbl.pack(fill="x", pady=(4, 6))

    def on_kind_change(_=None):
        k = kind.get()
        if k in PROFILES:
            rec_lbl.config(text=PROFILES[k]["rec"])

    kind.bind("<<ComboboxSelected>>", on_kind_change)

    checks = tk.Frame(tab_video, bg=PANEL)
    checks.pack(fill="x", pady=(4, 6))
    captions = tk.BooleanVar(value=True)
    hook = tk.BooleanVar(value=True)
    edit = tk.BooleanVar(value=True)
    music = tk.BooleanVar(value=True)
    censor = tk.BooleanVar(value=True)

    for txt_chk, var in (("Captions (subtitles)", captions),
                         ("3s hook banner", hook),
                         ("Cut dead air & zoom", edit),
                         ("Background music", music),
                         ("Bleep profanity", censor)):
        ttk.Checkbutton(checks, text=txt_chk, variable=var, style="Card.TCheckbutton").pack(side="left", padx=(0, 16))

    # Expandable Advanced Options Drawer
    adv_box = tk.Frame(tab_video, bg=PANEL)
    adv_box.pack(fill="x", pady=(6, 0))

    adv_content = tk.Frame(adv_box, bg=FIELD, highlightbackground=BORDER, highlightthickness=1, padx=12, pady=10)
    adv_visible = [False]

    def toggle_advanced():
        adv_visible[0] = not adv_visible[0]
        if adv_visible[0]:
            adv_content.pack(fill="x", pady=(6, 0))
            adv_btn.config(text="⚙️ Hide Advanced Settings ▲")
        else:
            adv_content.pack_forget()
            adv_btn.config(text="⚙️ Show Advanced Settings (Model, Facecam, Buffer, Focus) ▼")

    adv_btn = flat_button(adv_box, "⚙️ Show Advanced Settings (Model, Facecam, Buffer, Focus) ▼",
                          toggle_advanced, "secondary")
    adv_btn.pack(anchor="w")

    adv_grid = tk.Frame(adv_content, bg=FIELD)
    adv_grid.pack(fill="x")

    def adv_combo(r, c, label, values, default, colspan=1):
        f = tk.Frame(adv_grid, bg=FIELD)
        f.grid(row=r, column=c, columnspan=colspan, sticky="we", padx=(0, 10), pady=(0, 6))
        adv_grid.columnconfigure(c, weight=1)
        tk.Label(f, text=label, bg=FIELD, fg=MUTED, font=(FONT, 9)).pack(anchor="w")
        cb = ttk.Combobox(f, values=values, state="readonly")
        cb.set(default)
        cb.pack(fill="x", pady=(2, 0))
        return cb

    model = adv_combo(0, 0, "AI Model (Hardware, VRAM & RAM Specs)", MODEL_CHOICES, DEFAULT_LABEL, colspan=2)
    cam_pos = adv_combo(0, 2, "Facecam Position", CAM_POS_CHOICES, DEFAULT_CAM_POS_LABEL)
    live = adv_combo(0, 3, "Live Stream Buffer", LIVE_CHOICES, DEFAULT_LIVE_LABEL)

    adv_row1 = tk.Frame(adv_content, bg=FIELD)
    adv_row1.pack(fill="x", pady=(6, 0))

    f_cookies = tk.Frame(adv_row1, bg=FIELD)
    f_cookies.pack(side="left", fill="x", expand=False, padx=(0, 12))
    tk.Label(f_cookies, text="Browser Cookies", bg=FIELD, fg=MUTED, font=(FONT, 9)).pack(anchor="w")
    cookies = ttk.Combobox(f_cookies, values=["(None - Recommended)", "cookies.txt (Folder)", "firefox", "edge (Close first)", "chrome (DPAPI limited)"], state="readonly", width=22)
    cookies.set("(None - Recommended)")
    cookies.pack(fill="x", pady=(2, 0))

    f_focus = tk.Frame(adv_row1, bg=FIELD)
    f_focus.pack(side="left", fill="x", expand=True)
    tk.Label(f_focus, text="Focus Topic / Time Range (optional)", bg=FIELD, fg=MUTED, font=(FONT, 9)).pack(anchor="w")
    focus = ttk.Entry(f_focus)
    focus.pack(fill="x", pady=(2, 0))
    tk.Label(adv_content, text='e.g. "funniest moments"  |  "crypto debate"  |  "1:20:00 to 1:45:30"',
             bg=FIELD, fg=MUTED, font=(FONT, 8, "italic")).pack(anchor="w", pady=(4, 0))

    # ===== TAB 2: CAPTIONS & TYPOGRAPHY STUDIO =====
    grid_cap = tk.Frame(tab_captions, bg=PANEL)
    grid_cap.pack(fill="x")

    caption_style = combo(grid_cap, 0, 0, "Caption Animation Style", CAPTION_STYLE_CHOICES, DEFAULT_CAPTION_STYLE_LABEL, colspan=2)
    font_choice = combo(grid_cap, 0, 2, "Font Choice (12 Viral Fonts)", FONTS, DEFAULT_FONT, colspan=2)

    brand_color = combo(grid_cap, 1, 0, "Brand Accent Color", BRAND_COLOR_CHOICES, DEFAULT_BRAND_COLOR_LABEL, colspan=2)
    cap_pos_main = combo(grid_cap, 1, 2, "Subtitle Position", CAP_POS_CHOICES, DEFAULT_CAP_POS_LABEL, colspan=2)

    # Live Motion Caption Preview Box
    prev_box = tk.Frame(tab_captions, bg=FIELD, highlightbackground=BORDER, highlightthickness=1, padx=12, pady=8)
    prev_box.pack(fill="x", pady=(8, 2))

    prev_head = tk.Frame(prev_box, bg=FIELD)
    prev_head.pack(fill="x")
    tk.Label(prev_head, text="LIVE MOTION CAPTION PREVIEW:",
             bg=FIELD, fg="#9277ff", font=(FONT, 8, "bold")).pack(side="left")
    style_desc_lbl = tk.Label(prev_head, text="", bg=FIELD, fg=MUTED, font=(FONT, 8, "italic"))
    style_desc_lbl.pack(side="right")

    prev_canvas = tk.Canvas(prev_box, height=44, bg="#000000", highlightthickness=0)
    prev_canvas.pack(fill="x", pady=(4, 2))

    anim_state = {"step": 0, "loop_id": None}
    sample_words = ["THIS", "IS", "VIRAL"]

    def update_caption_preview():
        if not prev_canvas.winfo_exists():
            return
        c_style = parse_caption_style(caption_style.get())
        b_info = parse_brand_color(brand_color.get())
        b_hex = b_info.get("hex", "#FFDD00")
        f_chosen = font_choice.get()

        prev_canvas.delete("all")
        cw = prev_canvas.winfo_width()
        ch = prev_canvas.winfo_height()
        if cw < 50:
            cw = 500
        if ch < 20:
            ch = 44

        # Background style
        if c_style in ("podcast", "abdaal"):
            prev_canvas.create_rectangle(cw // 2 - 140, 4, cw // 2 + 140, ch - 4,
                                         fill="#1c1f2b", outline="#3b4259", width=1)
        elif c_style == "neon":
            prev_canvas.create_rectangle(cw // 2 - 150, 4, cw // 2 + 150, ch - 4,
                                         fill="#0a0515", outline="#7c5cff", width=1)

        words = sample_words if c_style not in ("abdaal", "podcast") else ["This", "is", "viral"]
        active_idx = anim_state["step"] % (len(words) + 1)

        base_family = f_chosen
        base_size = 13 if c_style in ("abdaal", "podcast") else 16

        spacing = 90 if c_style in ("beast", "hormozi", "karaoke") else 76
        start_x = cw // 2 - spacing

        for idx_w, w in enumerate(words):
            x = start_x + idx_w * spacing
            y = ch // 2
            is_active = (idx_w == active_idx)

            if is_active:
                f_size = base_size + (2 if c_style in ("beast", "hormozi") else (1 if c_style == "neon" else 0))
                font_spec = (base_family, f_size, "bold")
                fg_col = b_hex
                if c_style == "beast":
                    y -= 3
                    prev_canvas.create_text(x + 1, y + 1, text=w, font=font_spec, fill="#000000")
                elif c_style == "karaoke":
                    y -= 4
                elif c_style == "neon":
                    prev_canvas.create_text(x, y, text=w, font=font_spec, fill="#ff00cc")
                elif c_style == "abdaal":
                    prev_canvas.create_rectangle(x - 22, y - 8, x + 22, y + 8, fill="#2a334a", outline="")
            else:
                font_spec = (base_family, base_size, "bold")
                fg_col = "#e8eaf0"
                prev_canvas.create_text(x + 1, y + 1, text=w, font=font_spec, fill="#000000")

            prev_canvas.create_text(x, y, text=w, font=font_spec, fill=fg_col)

        anim_state["step"] = (anim_state["step"] + 1) % 4
        anim_state["loop_id"] = root.after(230, update_caption_preview)

    def on_style_or_color_change(_=None):
        c_style = parse_caption_style(caption_style.get())
        descs = {
            "beast": "MrBeast Fast Kinetic Bounce (Smooth Pop & Spring Ease-Out)",
            "hormozi": "Alex Hormozi Yellow Pop (Bold Punch & Elastic Settle)",
            "abdaal": "Ali Abdaal Clean Highlighter (Modern Pill Box)",
            "neon": "Cyberpunk Neon Glow (Electric Aura Shimmer)",
            "podcast": "Clean Podcast Subtitle (Minimal Lower-Third)",
            "karaoke": "Kinetic Wave Bounce (Progressive Speech)",
            "auto": "Auto Niche Preset",
        }
        style_desc_lbl.config(text=descs.get(c_style, ""))

    caption_style.bind("<<ComboboxSelected>>", on_style_or_color_change)
    brand_color.bind("<<ComboboxSelected>>", on_style_or_color_change)
    font_choice.bind("<<ComboboxSelected>>", on_style_or_color_change)
    on_style_or_color_change()

    # --- Actions
    acts = tk.Frame(wrap, bg=BG)
    acts.pack(fill="x", pady=(8, 8))
    go = flat_button(acts, "🚀 MAKE CLIPS", None, "primary", width=18)
    go.pack(side="left")
    stop = flat_button(acts, "⏹ Stop", None, "danger", width=10)
    stop.pack(side="left", padx=(10, 0))
    stop.config(state="disabled")

    # --- 3. Progress & Activity Console
    c4, i4 = card(wrap, "Progress & Activity")
    c4.pack(fill="both", expand=True, pady=(0, 0))

    r1 = tk.Frame(i4, bg=PANEL)
    r1.pack(fill="x")
    tk.Label(r1, text="Overall Progress", bg=PANEL, fg=TEXT, font=(FONT, 10, "bold")).pack(side="left")
    pct = tk.Label(r1, text="0%", bg=PANEL, fg=TEXT, font=(FONT, 10, "bold"))
    pct.pack(side="right")

    overall = ttk.Progressbar(i4, style="Main.Horizontal.TProgressbar", maximum=100)
    overall.pack(fill="x", pady=(4, 8))

    step_lbl = tk.Label(i4, text="Ready - paste a video link and click MAKE CLIPS", bg=PANEL, fg=MUTED,
                        font=(FONT, 9), anchor="w", justify="left")
    step_lbl.pack(fill="x")
    stepbar = ttk.Progressbar(i4, style="Step.Horizontal.TProgressbar", maximum=100)
    stepbar.pack(fill="x", pady=(4, 8))

    logbox = tk.Frame(i4, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
    logbox.pack(fill="both", expand=True)
    box = tk.Text(logbox, height=5, state="disabled", bg=FIELD, fg="#c9cfdd", font=("Consolas", 9), relief="flat",
                  padx=10, pady=8, wrap="word", highlightthickness=0)
    sbar = ttk.Scrollbar(logbox, orient="vertical", command=box.yview)
    box.configure(yscrollcommand=sbar.set)
    sbar.pack(side="right", fill="y")
    box.pack(side="left", fill="both", expand=True)
    box.tag_configure("err", foreground="#f87171")
    box.tag_configure("ok", foreground="#4ade80")
    box.tag_configure("step", foreground="#a99bff")

    def set_progress(stage, frac, text):
        idx = STAGES.index(stage)
        done = sum(STAGE_WEIGHT[s] for s in STAGES[:idx])
        total = min(1.0, done + STAGE_WEIGHT[stage] * max(0.0, min(1.0, frac)))
        overall["value"] = total * 100
        pct.config(text=f"{total * 100:.0f}%")
        stepbar["value"] = max(0.0, min(1.0, frac)) * 100
        step_lbl.config(text=f"Step {idx + 1} of {len(STAGES)}  -  {STAGE_NAMES[stage]}" + (f"  -  {text}" if text else ""),
                        fg=TEXT)

    def pump():
        while not q.empty():
            m = q.get()
            if isinstance(m, tuple):
                if m[0] == "F":
                    finish(m[1], m[2])
                    if m[2]:
                        overall["value"] = stepbar["value"] = 100
                        pct.config(text="100%")
                else:
                    set_progress(m[1], m[2], m[3])
                continue
            if m.startswith("\x00POP_RESULTS:"):
                f = m.split(":", 1)[1]
                last["dir"] = f
                if not getattr(root, "_results_win", None) or not root._results_win.winfo_exists():
                    root._results_win = show_results(root, f)
                else:
                    root._results_win.refresh(f)
                continue
            if m.startswith("\x00REFRESH_RESULTS:"):
                f = m.split(":", 1)[1]
                last["dir"] = f
                if getattr(root, "_results_win", None) and root._results_win.winfo_exists():
                    root._results_win.refresh(f)
                continue
            if m.startswith("\x00RESULTS:"):
                f = m.split(":", 1)[1]
                last["dir"] = f
                if not getattr(root, "_results_win", None) or not root._results_win.winfo_exists():
                    root._results_win = show_results(root, f)
                else:
                    root._results_win.refresh(f)
                continue
            tag = ("err" if m.startswith("ERROR") or "FAILED" in m else "ok" if m.startswith("DONE")
                   else "step" if re.match(r"^\d[a-z]?/8", m) else None)
            box.config(state="normal")
            box.insert("end", m + "\n", tag)
            box.see("end")
            box.config(state="disabled")
        root.after(150, pump)

    def finish(msg, ok):
        go.config(state="normal", bg=ACCENT)
        stop.config(state="disabled", bg=PANEL2)
        step_lbl.config(text=msg, fg=GOOD if ok else MUTED)

    def work(src, opts):
        global log, prog
        log = lambda m: q.put(str(m))
        prog = lambda stage, frac=0.0, text="": q.put(("P", stage, frac, text))
        try:
            last["dir"] = run_job(src, **opts)
            q.put("\x00RESULTS:" + str(last["dir"]))
            q.put(("F", "Finished", True))
        except Cancelled:
            q.put("Stopped.")
            q.put(("F", "Stopped", False))
        except Exception as e:
            (ROOT / "error_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
            q.put(f"ERROR: {e}\n(full details saved to error_log.txt - send it to get a fix)")
            q.put(("F", "Stopped because of an error - see the log", False))

    def start():
        ok, msg = verify_remote_license()
        if not ok:
            messagebox.showerror("Access Revoked", f"ACCESS REVOKED BY OWNER ({OWNER_NAME}):\n\n{msg}\n\n"
                                                   f"Commercial resale or unauthorized distribution is strictly prohibited.\n"
                                                   f"Contact: {OWNER_NAME} on Facebook.")
            return
        s = url.get().strip()
        if not s:
            step_lbl.config(text="Paste a link or choose a file first", fg="#f87171")
            return
        opts = dict(kind=kind.get(), layout=layout.get(), ratio=ratio.get(), cam_pos=cam_pos.get(),
                    model=model.get(), captions=captions.get(), hook=hook.get(),
                    censor=censor.get(), music=music.get(), edit=edit.get(),
                    focus=focus.get().strip(), cookies="" if str(cookies.get()).startswith("(") else cookies.get(), live=live.get(),
                    brand_color=brand_color.get(), caption_style=caption_style.get(),
                    top_clips=top_clips.get(), font=font_choice.get())
        CANCEL.clear()
        overall["value"] = stepbar["value"] = 0
        pct.config(text="0%")
        step_lbl.config(text="Starting...", fg=TEXT)
        go.config(state="disabled", bg="#3a3f55")
        stop.config(state="normal", bg=DANGER)
        threading.Thread(target=work, args=(s, opts), daemon=True).start()

    def do_stop():
        CANCEL.set()
        stop.config(state="disabled", bg=PANEL2)
        step_lbl.config(text="Stopping... finishing the current step", fg="#fbbf24")

    go.config(command=start)
    stop.config(command=do_stop)
    root.after(300, update_caption_preview)
    pump()
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--cli", metavar="URL_OR_FILE")
    ap.add_argument("--type", default="gaming", choices=list(PROFILES))
    ap.add_argument("--layout", default="auto")
    ap.add_argument("--ratio", default="9:16", choices=["9:16", "3:4", "1:1", "16:9"])
    ap.add_argument("--cam-pos", default="top", choices=["top", "bottom", "pip-tr", "pip-tl", "pip-br", "pip-bl"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--focus", default="")
    ap.add_argument("--live", default=0, type=int, help="Live stream capture buffer in minutes (e.g. 15, 30)")
    ap.add_argument("--brand-color", default="yellow", choices=list(BRAND_COLORS))
    ap.add_argument("--caption-style", default="auto", choices=list(CAPTION_STYLES))
    ap.add_argument("--top-clips", default=DEFAULT_TOP_CLIPS_LABEL, choices=TOP_CLIPS_CHOICES)
    ap.add_argument("--font", default=DEFAULT_FONT, choices=FONTS, help="Viral font choice")
    ap.add_argument("--layout-mode", default="auto", choices=["auto", "gaming", "podcast", "irl", "screen"])
    ap.add_argument("--caption-template", default="hormozi", choices=["hormozi", "pop", "minimal", "karaoke", "beast", "gaming", "classic"])
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--no-hook", action="store_true")
    ap.add_argument("--no-censor", action="store_true")
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--no-edit", action="store_true")
    a = ap.parse_args()
    if a.setup:
        setup_models()
    elif a.cli:
        run_job(a.cli, kind=a.type, layout=a.layout, ratio=a.ratio, cam_pos=a.cam_pos, model=a.model,
                captions=not a.no_captions, hook=not a.no_hook, censor=not a.no_censor,
                music=not a.no_music, edit=not a.no_edit, focus=a.focus, live=a.live,
                brand_color=a.brand_color, caption_style=a.caption_style, top_clips=a.top_clips,
                font=a.font, layout_mode=a.layout_mode, caption_template=a.caption_template)
    else:
        gui()
