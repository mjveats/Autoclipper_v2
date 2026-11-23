"""Styled Template Caption Rendering Service via pycaps.

Runs after reframing:
- Input: 1080x1920 reframed video, word-level transcription timestamps, caption_template ID.
- Maps template IDs ('explosive', 'hype', 'minimalist', 'vibrant', 'retro-gaming', 'classic', 'hormozi', etc.)
  to pycaps templates with customized styling.
- 3s Hook Banner rendering via pycaps with round-corner pill design & customizable vertical position (top, center, bottom).
- Layout-Aware Positioning:
  * For 'gaming': captions placed at top: 32% (along the split seam between webcam and gameplay).
  * For 'irl', 'podcast', 'screen': captions placed at bottom: 18% (clear of TikTok/Reels UI overlays).
- Bundles custom fonts explicitly in project resources/fonts to avoid default fallback.
"""

from pathlib import Path
import json
import os
import shutil
import logging
from typing import List, Dict, Any, Optional

import pycaps
from pycaps import (
    CapsPipelineBuilder,
    Document,
    Line,
    Segment,
    TimeFragment,
    Word,
    SubtitleLayoutOptions,
    VerticalAlignment,
    VerticalAlignmentType,
    VideoQuality,
    load_transcription,
)
from pycaps.template.builtin_template import BuiltinTemplate
from pycaps.pipeline.json_config_loader import JsonConfigLoader

import sys

logger = logging.getLogger("caption_renderer")


def _get_resources_dir() -> Path:
    candidates = [
        Path(sys.executable).resolve().parent / "resources" / "fonts",
        Path(__file__).resolve().parent.parent.parent / "resources" / "fonts",
        Path(__file__).resolve().parent.parent.parent.parent / "resources" / "fonts",
    ]
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "resources" / "fonts")
    for c in candidates:
        if c.exists():
            return c
    return Path(__file__).resolve().parent.parent.parent / "resources" / "fonts"


PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent.parent
RESOURCES_DIR = _get_resources_dir()

# Map frontend/user templates to pycaps presets & styles
TEMPLATE_CONFIG_MAP = {
    "explosive": {
        "preset": "explosive",
        "color": "#FFDD00",
        "font_file": "black.ttf",
        "font_family": "CustomFont",
    },
    "hype": {
        "preset": "hype",
        "color": "#00FF66",
        "font_file": "komika.ttf",
        "font_family": "Komika Axis",
    },
    "minimalist": {
        "preset": "minimalist",
        "color": "#FFFFFF",
        "font_file": None,
        "font_family": "Helvetica Neue, Arial, sans-serif",
    },
    "vibrant": {
        "preset": "vibrant",
        "color": "#00E5FF",
        "font_file": "black.ttf",
        "font_family": "CustomFont",
    },
    "retro-gaming": {
        "preset": "retro-gaming",
        "color": "#00FF66",
        "font_file": "PressStart2P.ttf",
        "font_family": "PressStart2P",
    },
    "classic": {
        "preset": "classic",
        "color": "#FFFFFF",
        "font_file": None,
        "font_family": "Arial, sans-serif",
    },
    "hormozi": {
        "preset": "explosive",
        "color": "#FFDD00",
        "font_file": "black.ttf",
        "font_family": "CustomFont",
    },
    "default": {
        "preset": "default",
        "color": "#FFDD00",
        "font_file": "black.ttf",
        "font_family": "CustomFont",
    },
    "pop": {
        "preset": "hype",
        "color": "#00FF66",
        "font_file": "komika.ttf",
        "font_family": "Komika Axis",
    },
    "minimal": {
        "preset": "minimalist",
        "color": "#FFFFFF",
        "font_file": None,
        "font_family": "Helvetica Neue, Arial, sans-serif",
    },
    "karaoke": {
        "preset": "vibrant",
        "color": "#00E5FF",
        "font_file": "black.ttf",
        "font_family": "CustomFont",
    },
    "beast": {
        "preset": "explosive",
        "color": "#FFDD00",
        "font_file": "black.ttf",
        "font_family": "CustomFont",
    },
    "gaming": {
        "preset": "retro-gaming",
        "color": "#00FF66",
        "font_file": "PressStart2P.ttf",
        "font_family": "PressStart2P",
    },
}


def build_pycaps_document(word_list: List[Dict[str, Any]]) -> Document:
    """Converts a list of word dictionaries with 'word'/'text', 'start', 'end'
    into a pycaps Document.
    """
    doc = Document()
    if not word_list:
        return doc

    segments_words: List[List[Word]] = []
    current_group: List[Word] = []

    for item in word_list:
        text = str(item.get("word", item.get("text", ""))).strip()
        if not text:
            continue
        try:
            start = max(0.0, float(item.get("start", 0.0)))
            end = max(start + 0.05, float(item.get("end", start + 0.3)))
        except (ValueError, TypeError):
            continue

        w = Word(text=text, time=TimeFragment(start=start, end=end))

        if current_group:
            prev_end = current_group[-1].time.end
            gap = start - prev_end
            if gap > 0.45 or len(current_group) >= 5 or text.endswith((".", "!", "?")):
                segments_words.append(current_group)
                current_group = []

        current_group.append(w)

    if current_group:
        segments_words.append(current_group)

    for group in segments_words:
        if not group:
            continue
        seg_start = group[0].time.start
        seg_end = group[-1].time.end
        seg_time = TimeFragment(start=seg_start, end=seg_end)

        seg = Segment(time=seg_time)
        line = Line(time=seg_time)
        line.words.set_all(group)
        seg.lines.add(line)
        doc.segments.add(seg)

    return doc


def render_pycaps_hook_banner(
    video_path: str,
    output_path: str,
    hook_text: str,
    hook_position: str = "top",
    duration: float = 3.5,
) -> str:
    """Renders a round-corner 3s hook banner onto video using pycaps.

    Matches the exact preview design:
    - Pill shape with rounded corners (border-radius: 24px)
    - High-contrast yellow (#FFDD00) badge with bold black font
    - Positionable: top (offset 0.08), center (offset 0.0), bottom (offset -0.24)
    """
    if not hook_text:
        shutil.copyfile(video_path, output_path)
        return output_path

    clean_text = str(hook_text).strip().strip('"').strip("'")
    if not clean_text:
        shutil.copyfile(video_path, output_path)
        return output_path

    pos = (hook_position or "top").lower().strip()
    if pos == "center":
        v_align = VerticalAlignment(align=VerticalAlignmentType.CENTER, offset=0.0)
    elif pos == "bottom":
        v_align = VerticalAlignment(align=VerticalAlignmentType.BOTTOM, offset=-0.24)
    else:  # top
        v_align = VerticalAlignment(align=VerticalAlignmentType.TOP, offset=0.08)

    layout_opts = SubtitleLayoutOptions(
        max_width_ratio=0.88,
        max_number_of_lines=2,
        min_number_of_lines=1,
        vertical_align=v_align,
    )

    builtin = BuiltinTemplate("default")
    loader = JsonConfigLoader(builtin.get_json_path())
    builder = loader.load(should_build_pipeline=False)

    builder.with_input_video(video_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    builder.with_output_video(output_path)

    if RESOURCES_DIR.exists():
        builder.with_resources(str(RESOURCES_DIR))

    builder.with_layout_options(layout_opts)

    # Modern pycaps round-corner hook pill CSS
    custom_css = """
    .line {
        display: inline-flex !important;
        background-color: #FFDD00 !important;
        border-radius: 24px !important;
        padding: 8px 24px !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.65) !important;
    }
    .word {
        color: #000000 !important;
        font-family: 'CustomFont', 'Montserrat', Arial, sans-serif !important;
        font-weight: 900 !important;
        font-size: 24px !important;
        text-transform: uppercase !important;
        text-shadow: none !important;
        padding: 0 4px !important;
    }
    """
    builder.add_css_content(custom_css)

    doc = Document()
    words = [Word(text=w, time=TimeFragment(start=0.0, end=duration)) for w in clean_text.split()]
    seg = Segment(time=TimeFragment(start=0.0, end=duration))
    line = Line(time=TimeFragment(start=0.0, end=duration))
    line.words.set_all(words)
    seg.lines.add(line)
    doc.segments.add(seg)

    pipeline = builder.build()
    pipeline.prepare()
    processed_doc = pipeline.process_document(doc)
    pipeline.render(processed_doc)

    return output_path


def _render_pycaps_captions(
    video_path: str,
    output_path: str,
    words: List[Dict[str, Any]],
    template_id: str = "explosive",
    layout_mode: str = "irl",
    brand_color: Optional[str] = None,
) -> str:
    """Internal helper to render word-timed styled subtitles using pycaps."""
    if not words:
        shutil.copyfile(video_path, output_path)
        return output_path

    temp_id = (template_id or "explosive").lower().strip()
    cfg = TEMPLATE_CONFIG_MAP.get(temp_id, TEMPLATE_CONFIG_MAP["explosive"])
    preset_name = cfg["preset"]

    builtin = BuiltinTemplate(preset_name)
    loader = JsonConfigLoader(builtin.get_json_path())
    builder = loader.load(should_build_pipeline=False)

    builder.with_input_video(video_path)
    if os.path.exists(output_path):
        os.remove(output_path)
    builder.with_output_video(output_path)

    if RESOURCES_DIR.exists():
        builder.with_resources(str(RESOURCES_DIR))

    if layout_mode == "gaming":
        v_align = VerticalAlignment(align=VerticalAlignmentType.TOP, offset=0.32)
    else:
        v_align = VerticalAlignment(align=VerticalAlignmentType.BOTTOM, offset=-0.18)

    layout_opts = SubtitleLayoutOptions(
        max_width_ratio=0.82,
        max_number_of_lines=2,
        min_number_of_lines=1,
        vertical_align=v_align,
    )
    builder.with_layout_options(layout_opts)

    accent = brand_color or cfg.get("color", "#FFDD00")
    custom_css = f"""
    .word-being-narrated {{
        color: {accent} !important;
    }}
    .word-highlight {{
        color: {accent} !important;
    }}
    """
    builder.add_css_content(custom_css)

    doc = build_pycaps_document(words)
    pipeline = builder.build()
    pipeline.prepare()
    processed_doc = pipeline.process_document(doc)
    pipeline.render(processed_doc)

    return output_path


def render_styled_captions(
    video_path: str,
    output_path: str,
    words: List[Dict[str, Any]],
    template_id: str = "explosive",
    layout_mode: str = "irl",
    brand_color: Optional[str] = None,
    hook_text: Optional[str] = None,
    hook_position: str = "top",
) -> str:
    """Renders pycaps hook banner (if provided) and pycaps styled captions onto a 1080x1920 video.

    - hook_text: 3s hook banner rendered via pycaps with round-corner pill & chosen position.
    - words: word-level transcription rendered via pycaps with selected template.
    """
    has_hook = bool(hook_text and str(hook_text).strip())
    has_captions = bool(words)

    if not has_hook and not has_captions:
        shutil.copyfile(video_path, output_path)
        return output_path

    current_input = video_path

    # Step 1: Render 3s hook banner via pycaps if present
    temp_hook = None
    if has_hook:
        try:
            target = output_path if not has_captions else str(Path(output_path).with_name(f"temp_hook_{Path(output_path).name}"))
            render_pycaps_hook_banner(
                current_input,
                target,
                hook_text=hook_text,
                hook_position=hook_position,
                duration=3.5,
            )
            current_input = target
            if has_captions:
                temp_hook = target
        except Exception as e:
            logger.warning(f"Pycaps hook banner rendering failed: {e}. Falling back without hook.")
            current_input = video_path

    # Step 2: Render word-level captions via pycaps if present
    if has_captions:
        try:
            _render_pycaps_captions(
                current_input,
                output_path,
                words,
                template_id=template_id,
                layout_mode=layout_mode,
                brand_color=brand_color,
            )
        finally:
            if temp_hook and os.path.exists(temp_hook):
                try:
                    os.remove(temp_hook)
                except Exception:
                    pass
    elif not has_hook:
        shutil.copyfile(video_path, output_path)

    return output_path
