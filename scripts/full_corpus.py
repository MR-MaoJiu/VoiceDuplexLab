#!/usr/bin/env python3
"""本机扩充素材下载入口。"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from voice_lab.training_materials import main

if __name__ == "__main__":
    main()
