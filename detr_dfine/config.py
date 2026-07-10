import os
import site
from pathlib import Path

for p in site.getsitepackages() + [site.getusersitepackages()]:
    nvidia_dir = os.path.join(p, 'nvidia')
    if os.path.exists(nvidia_dir):
        for lib in os.listdir(nvidia_dir):
            bin_path = os.path.join(nvidia_dir, lib, 'bin')
            if os.path.exists(bin_path):
                os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(bin_path)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WORKSPACE_DIR = PROJECT_ROOT / "detr_dfine"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PREDICTIONS_DIR = WORKSPACE_DIR / "predictions"
PREDICTIONS_DIR.mkdir(exist_ok=True)

MODELS_PT_DIR = PROJECT_ROOT / "weights" / "pt_models"
MODELS_ONNX_DIR = PROJECT_ROOT / "weights" / "onnx_models"
DATASET_DIR = PROJECT_ROOT / "datasets" / "coco"
ANNOTATIONS = DATASET_DIR / "annotations" / "instances_val2017.json"

device = 'cuda'

hf_models = [
    "LibreYOLO/LibreDFINEn",
    "LibreYOLO/LibreRFDETRn",
    "LibreYOLO/LibreRTDETRr50m",
    "LibreYOLO/LibreRTDETRv2r50m",
    "LibreYOLO/LibreRTDETRv4m",
    "LibreYOLO/LibreRTDETRv4x",
    "LibreRTDETRv2r18",
    "LibreRTDETRr34"
]
