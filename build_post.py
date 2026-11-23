import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_APP = ROOT / "dist" / "AutoClipper"
TARGET_DIR = ROOT.parent / "1. COMPILED (Share With Testers)"
COMPILED_APP = TARGET_DIR / "AutoClipper"
VENV = ROOT / "venv"

print("=" * 60)
print("AutoClipper Post-Build Organizer")
print("=" * 60)

if not DIST_APP.exists():
    print(f"Error: {DIST_APP} does not exist.")
    sys.exit(1)

TARGET_DIR.mkdir(parents=True, exist_ok=True)

print(f"Syncing fresh build to {COMPILED_APP}...")
shutil.copytree(DIST_APP, COMPILED_APP, dirs_exist_ok=True)

# List of all assets that must be present
assets_to_copy = [
    Path("tools"),
    Path("web"),
    Path("resources"),
    Path("models"),
    Path("music"),
    Path("Install_Ollama_Automatic.bat"),
]

for asset in assets_to_copy:
    src = ROOT / asset
    if not src.exists():
        continue
    for base in [DIST_APP, COMPILED_APP]:
        dest = base / asset
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        print(f"Copied {asset} -> {dest}")

        # Also place in _internal to ensure PyInstaller internal resolution succeeds
        dest_internal = base / "_internal" / asset
        if src.is_dir():
            shutil.copytree(src, dest_internal, dirs_exist_ok=True)
        else:
            dest_internal.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_internal)

# Copy CUDA DLLs to _internal
for internal_dir in [DIST_APP / "_internal", COMPILED_APP / "_internal"]:
    internal_dir.mkdir(parents=True, exist_ok=True)
    nv = VENV / "Lib" / "site-packages" / "nvidia"
    if nv.exists():
        for dll in nv.glob("*/bin/*.dll"):
            dest_file = internal_dir / dll.name
            if not dest_file.exists():
                shutil.copy2(dll, dest_file)
                print("Copied CUDA DLL:", dll.name)

    # Copy silero_vad_v6.onnx to _internal/faster_whisper/assets/
    vad_src = VENV / "Lib" / "site-packages" / "faster_whisper" / "assets" / "silero_vad_v6.onnx"
    vad_dest = internal_dir / "faster_whisper" / "assets"
    if vad_src.exists():
        vad_dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(vad_src, vad_dest / "silero_vad_v6.onnx")
        print("Copied silero_vad_v6.onnx")

# Create zip file for testers
zip_dest = TARGET_DIR / "AutoClipper_v2.0_Windows"
print(f"Creating tester archive: {zip_dest}.zip...")
zip_file = TARGET_DIR / "AutoClipper_v2.0_Windows.zip"
if zip_file.exists():
    try:
        zip_file.unlink()
    except Exception:
        pass
shutil.make_archive(str(zip_dest), "zip", str(COMPILED_APP))
print("Tester archive ready!")

print("=" * 60)
print("BUILD & PACKAGING COMPLETE!")
print(f"Local App:  {DIST_APP}")
print(f"Target App: {COMPILED_APP}")
print(f"Tester Zip: {zip_dest}.zip")
print("=" * 60)
