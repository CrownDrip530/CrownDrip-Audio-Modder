"""
main.py
Entry point for CrownDrip Audio Modder.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from gui.main_window import run

if __name__ == "__main__":
    run()
