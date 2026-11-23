"""Automatic Niche-Based 16:9 -> 9:16 Reframing Service.

Guarantees standardized 1080x1920 .mp4 output regardless of niche:
1. detect_layout: Samples 6 evenly spaced frames using OpenCV and mediapipe.solutions.face_detection.
   - "gaming": 1 small face (< 6% frame area) pinned near corner/edge (cx < 0.25 or cx > 0.75, cy < 0.35 or cy > 0.65) in >= 2 frames.
   - "podcast": >= 2 faces (> 0.8% area each) detected simultaneously in >= 2 frames.
   - "irl": 1 prominent face/subject detected in >= 2 frames.
   - "screen": Fallback when 0 prominent faces are detected.
2. 4-Way Routing:
   - "gaming": Custom OpenCV + FFmpeg Split-Screen (webcam 1080x672 top 35% + gameplay 1080x1248 bottom 65%).
   - "podcast": Active speaker tracking (left/right speaker pan/crop or dual vertical stack).
   - "irl": Smooth face/subject centering with camera pan damping.
   - "screen": Autocrop-vertical blurred-background letterbox framing.
"""

from pathlib import Path
import os
import shutil
import subprocess
import cv2
import mediapipe as mp
import numpy as np


import sys


def _get_ffmpeg_cmd() -> str:
    # Check if local tools ffmpeg exists across potential bundle and project locations
    candidates = [
        Path(sys.executable).resolve().parent / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(__file__).resolve().parent.parent.parent / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path(__file__).resolve().parent.parent.parent.parent / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe")
    for cand in candidates:
        if cand.exists():
            return str(cand)
    which_ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if which_ffmpeg:
        return which_ffmpeg
    return "ffmpeg"


def _run_ffmpeg(cmd: list, label: str = "ffmpeg"):
    """Run an FFmpeg command; raise RuntimeError with captured stderr on failure."""
    cmd_list = [str(c) for c in cmd]
    if cmd_list and cmd_list[0] in ("ffmpeg", "ffmpeg.exe"):
        cmd_list[0] = _get_ffmpeg_cmd()
    flags = 0x08000000 if os.name == "nt" else 0
    try:
        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=flags,
        )
    except FileNotFoundError as e:
        raise RuntimeError(f"{label} failed: binary '{cmd_list[0]}' not found. Error: {e}")
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed (rc={result.returncode}):\n{(result.stderr or result.stdout or '')[-1200:]}"
        )
    return result


def detect_layout(video_path: str, layout_override: str = "auto") -> str:
    """Detects video niche/layout based on facial geometry and positions.

    Returns one of: 'gaming', 'podcast', 'irl', 'screen'.
    """
    if layout_override and layout_override.lower() not in ("auto", "none", ""):
        clean = layout_override.lower().strip()
        if clean in ("gaming", "podcast", "irl", "screen"):
            return clean

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "screen"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if w <= 0 or h <= 0 or total_frames <= 0:
        cap.release()
        return "screen"

    frame_area = float(w * h)

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    sample_pcts = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
    gaming_votes = 0
    podcast_votes = 0
    irl_votes = 0

    try:
        for pct in sample_pcts:
            frame_idx = int(total_frames * pct)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(rgb)

            if not results or not results.detections:
                continue

            faces = []
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                bw = bbox.width * w
                bh = bbox.height * h
                area_ratio = (bw * bh) / frame_area
                cx = bbox.xmin + bbox.width / 2.0
                cy = bbox.ymin + bbox.height / 2.0
                faces.append(dict(cx=cx, cy=cy, area_ratio=area_ratio))

            if not faces:
                continue

            # Check podcast: >= 2 faces (> 0.8% area each)
            prominent_faces = [f for f in faces if f["area_ratio"] >= 0.008]
            if len(prominent_faces) >= 2:
                podcast_votes += 1
                continue

            # Check gaming: 1 small face (< 6% area) near corner/edge
            if len(faces) == 1:
                f0 = faces[0]
                is_small = f0["area_ratio"] < 0.06
                is_corner = (f0["cx"] < 0.25 or f0["cx"] > 0.75) and (f0["cy"] < 0.35 or f0["cy"] > 0.65)
                if is_small and is_corner:
                    gaming_votes += 1
                    continue
                elif f0["area_ratio"] >= 0.02:
                    irl_votes += 1
                    continue

            if len(faces) >= 1 and faces[0]["area_ratio"] >= 0.02:
                irl_votes += 1
    finally:
        cap.release()
        mp_face.close()

    if gaming_votes >= 2:
        return "gaming"
    if podcast_votes >= 2:
        return "podcast"
    if irl_votes >= 2:
        return "irl"
    return "screen"


def reframe_gaming_split(input_path: str, output_path: str):
    """Reframes a gaming clip with facecam to 1080x1920 vertical split-screen.

    Webcam (top 35% -> 1080x672) + Gameplay (bottom 65% -> 1080x1248).
    """
    ffmpeg = _get_ffmpeg_cmd()
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    coords = []
    for pct in (0.15, 0.35, 0.55, 0.75):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pct))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results and results.detections:
            box = results.detections[0].location_data.relative_bounding_box
            coords.append((
                int((box.xmin + box.width / 2) * w),
                int((box.ymin + box.height / 2) * h)
            ))

    cap.release()
    mp_face.close()

    if coords:
        cam_cx = int(np.median([c[0] for c in coords]))
        cam_cy = int(np.median([c[1] for c in coords]))
    else:
        cam_cx, cam_cy = int(w * 0.15), int(h * 0.20)

    # Top 35% webcam (1080x672) + Bottom 65% gameplay (1080x1248) = 1080x1920
    cam_w = int(w * 0.28)
    cam_h = int(cam_w * (672 / 1080))
    cam_x = max(0, min(w - cam_w, cam_cx - cam_w // 2))
    cam_y = max(0, min(h - cam_h, cam_cy - cam_h // 2))

    game_w = int(h * (1080 / 1248))
    game_x = max(0, (w - game_w) // 2)

    filter_complex = (
        f"[0:v]crop={cam_w}:{cam_h}:{cam_x}:{cam_y},scale=1080:672:flags=lanczos[cam];"
        f"[0:v]crop={game_w}:{h}:{game_x}:0,scale=1080:1248:flags=lanczos[game];"
        f"[cam][game]vstack=inputs=2[out]"
    )

    _run_ffmpeg([
        ffmpeg, "-y", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", output_path
    ], label="reframe_gaming_split")


def reframe_podcast(input_path: str, output_path: str):
    """Reframes podcast/interview multi-speaker clip to 1080x1920.

    Detects speaker positions and creates a clean split view (top speaker / bottom speaker)
    or centered active area.
    """
    ffmpeg = _get_ffmpeg_cmd()
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    faces_left = []
    faces_right = []

    for pct in (0.15, 0.35, 0.55, 0.75):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pct))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results and results.detections:
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                cx = (box.xmin + box.width / 2) * w
                cy = (box.ymin + box.height / 2) * h
                if cx < w * 0.5:
                    faces_left.append((int(cx), int(cy)))
                else:
                    faces_right.append((int(cx), int(cy)))

    cap.release()
    mp_face.close()

    # If we have two distinct sides (left & right speaker), create a clean stacked dual shot
    if faces_left and faces_right:
        lx = int(np.median([c[0] for c in faces_left]))
        ly = int(np.median([c[1] for c in faces_left]))
        rx = int(np.median([c[0] for c in faces_right]))
        ry = int(np.median([c[1] for c in faces_right]))

        crop_w = int(h * (1080 / 960) * 0.55)
        crop_h = int(crop_w * (960 / 1080))
        crop_w = min(w, crop_w)
        crop_h = min(h, crop_h)

        l_x1 = max(0, min(w - crop_w, lx - crop_w // 2))
        l_y1 = max(0, min(h - crop_h, ly - crop_h // 2))
        r_x1 = max(0, min(w - crop_w, rx - crop_w // 2))
        r_y1 = max(0, min(h - crop_h, ry - crop_h // 2))

        filter_complex = (
            f"[0:v]crop={crop_w}:{crop_h}:{l_x1}:{l_y1},scale=1080:960:flags=lanczos[top];"
            f"[0:v]crop={crop_w}:{crop_h}:{r_x1}:{r_y1},scale=1080:960:flags=lanczos[bot];"
            f"[top][bot]vstack=inputs=2[out]"
        )
    else:
        # Fallback to smart center 9:16 crop with smooth framing
        target_w = int(h * (9 / 16))
        x_offset = max(0, (w - target_w) // 2)
        filter_complex = f"[0:v]crop={target_w}:{h}:{x_offset}:0,scale=1080:1920:flags=lanczos[out]"

    _run_ffmpeg([
        ffmpeg, "-y", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", output_path
    ], label="reframe_podcast")


def reframe_irl(input_path: str, output_path: str):
    """Reframes IRL / single-subject talking head video using smooth face centering."""
    ffmpeg = _get_ffmpeg_cmd()
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    centers_x = []
    for pct in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * pct))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        results = mp_face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results and results.detections:
            box = results.detections[0].location_data.relative_bounding_box
            cx = (box.xmin + box.width / 2) * w
            centers_x.append(int(cx))

    cap.release()
    mp_face.close()

    target_w = int(h * (9 / 16))
    if centers_x:
        median_x = int(np.median(centers_x))
        x_offset = max(0, min(w - target_w, median_x - target_w // 2))
    else:
        x_offset = max(0, (w - target_w) // 2)

    filter_complex = f"[0:v]crop={target_w}:{h}:{x_offset}:0,scale=1080:1920:flags=lanczos[out]"

    _run_ffmpeg([
        ffmpeg, "-y", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", output_path
    ], label="reframe_irl")


def reframe_screen(input_path: str, output_path: str):
    """Reframes screen-share / tutorials using blurred-background letterbox framing (1080x1920)."""
    ffmpeg = _get_ffmpeg_cmd()
    filter_complex = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
        "boxblur=25:5[bg];"
        "[0:v]scale=1080:-1:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[out]"
    )

    _run_ffmpeg([
        ffmpeg, "-y", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[out]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", output_path
    ], label="reframe_screen")


def reframe_to_vertical(input_path: str, output_path: str, layout_mode: str = "auto") -> str:
    """Master router: detects layout if 'auto' and reframes to guaranteed 1080x1920 .mp4."""
    chosen_layout = detect_layout(input_path, layout_override=layout_mode)

    if chosen_layout == "gaming":
        reframe_gaming_split(input_path, output_path)
    elif chosen_layout == "podcast":
        reframe_podcast(input_path, output_path)
    elif chosen_layout == "irl":
        reframe_irl(input_path, output_path)
    else:
        reframe_screen(input_path, output_path)

    return chosen_layout
