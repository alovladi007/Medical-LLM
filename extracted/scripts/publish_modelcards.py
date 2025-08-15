#!/usr/bin/env python3
import shutil, glob, os, sys
if len(sys.argv) < 3:
    print("Usage: publish_modelcards.py <metrics_dir> <ui_public_dir>")
    sys.exit(1)
metrics_dir, ui_public = sys.argv[1:3]
os.makedirs(ui_public, exist_ok=True)
paths = glob.glob(os.path.join(metrics_dir, "*.json"))
if not paths:
    print("No eval JSON found"); sys.exit(1)
latest = max(paths, key=os.path.getmtime)
shutil.copy2(latest, os.path.join(ui_public, os.path.basename(latest)))
print("Published", latest)
