import subprocess
import sys
import os

scripts = [
    "0_download_and_export.py",
    "1_model_stats.py",
    "2_generate_predictions.py",
    "3_calculate_metrics.py",
    "4_measure_speed.py"
]

for script in scripts:
    if os.path.exists(script):
        subprocess.run([sys.executable, script])
