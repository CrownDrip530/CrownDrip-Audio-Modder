"""
config.py
Handles all persistent memory for CrownDrip Audio Modder.
"""

import json
import os
import shutil
import uuid
from pathlib import Path

APP_NAME = "CrownDripAudioModder"

APP_DATA_DIR = Path(os.getenv("APPDATA", Path.home())) / APP_NAME
SOUNDS_DIR = APP_DATA_DIR / "sounds"
CONFIG_FILE = APP_DATA_DIR / "config.json"

DEFAULT_CONFIG = {
    "mic_device": None,
    "output_device": None,
    "mic_gain_db": 0.0,
    "soundboard_volume_db": 0.0,
    "monitor_enabled": False,
    "effects": {
        "deep_fry": {
            "enabled": False,
            "drive": 8.0,
            "bitcrush_depth": 6,
            "eq_boost_db": 10.0
        },
        "soundboard_deep_fry": {
            "enabled": False,
            "drive": 8.0,
            "bitcrush_depth": 6,
            "eq_boost_db": 10.0
        }
    },
    "soundboard": []
}


def _ensure_dirs():
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SOUNDS_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    _ensure_dirs()
    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return json.loads(json.dumps(DEFAULT_CONFIG))

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        data = {}

    merged = _deep_merge_defaults(DEFAULT_CONFIG, data)
    return merged


def _deep_merge_defaults(defaults: dict, data: dict) -> dict:
    result = dict(defaults)
    for key, value in data.items():
        if key in defaults and isinstance(defaults[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_defaults(defaults[key], value)
        else:
            result[key] = value
    return result


def save_config(config: dict):
    _ensure_dirs()
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def add_sound_to_library(config: dict, source_path: str, display_name: str) -> dict:
    _ensure_dirs()
    source_path = Path(source_path)
    ext = source_path.suffix or ".mp3"
    file_id = str(uuid.uuid4())
    stored_filename = f"{file_id}{ext}"
    dest_path = SOUNDS_DIR / stored_filename

    shutil.copy2(source_path, dest_path)

    entry = {
        "id": file_id,
        "name": display_name.strip() or source_path.stem,
        "filename": stored_filename,
        "volume": 1.0
    }
    config["soundboard"].append(entry)
    save_config(config)
    return entry


def rename_sound(config: dict, sound_id: str, new_name: str):
    for entry in config["soundboard"]:
        if entry["id"] == sound_id:
            entry["name"] = new_name.strip() or entry["name"]
            break
    save_config(config)


def remove_sound(config: dict, sound_id: str):
    entry = next((e for e in config["soundboard"] if e["id"] == sound_id), None)
    if entry:
        file_path = SOUNDS_DIR / entry["filename"]
        if file_path.exists():
            try:
                file_path.unlink()
            except OSError:
                pass
        config["soundboard"] = [e for e in config["soundboard"] if e["id"] != sound_id]
        save_config(config)


def get_sound_path(entry: dict) -> Path:
    return SOUNDS_DIR / entry["filename"]
