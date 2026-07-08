import os
import urllib.request
import sys

BASE_WEIGHTS_DIR = "weights"

MODELS_DATA = {
        "LibreYOLO9" : {
            "LibreYOLO9t.onnx": "https://huggingface.co/LibreYOLO/LibreYOLO9t/resolve/main/LibreYOLO9t.onnx?download=true",
            "LibreYOLO9s.onnx": "https://huggingface.co/LibreYOLO/LibreYOLO9s/resolve/main/LibreYOLO9s.onnx?download=true",
            "LibreYOLO9m.onnx": "https://huggingface.co/LibreYOLO/LibreYOLO9m/resolve/main/LibreYOLO9m.onnx?download=true",
            "LibreYOLO9c.onnx": "https://huggingface.co/LibreYOLO/LibreYOLO9c/resolve/main/LibreYOLO9c.onnx?download=true"
            },
        "LibreYOLO9E2E" : {
            "LibreYOLO9E2Et.onnx": "https://huggingface.co/LibreYOLO/LibreYOLO9E2Et/resolve/main/LibreYOLO9E2Et.onnx?download=true",
            "LibreYOLO9E2Es.onnx": "https://huggingface.co/LibreYOLO/LibreYOLO9E2Es/resolve/main/LibreYOLO9E2Es.onnx?download=true",
            "LibreYOLO9E2Em.onnx": "https://huggingface.co/LibreYOLO/LibreYOLO9E2Em/resolve/main/LibreYOLO9E2Em.onnx?download=true",
            "LibreYOLO9E2Ec.onnx": "https://huggingface.co/LibreYOLO/LibreYOLO9E2Ec/resolve/main/LibreYOLO9E2Ec.onnx?download=true"
            },
        "LibreYOLOX" : {
            "LibreYOLOXn.onnx": "https://huggingface.co/LibreYOLO/LibreYOLOXn/resolve/main/LibreYOLOXn.onnx?download=true",
            "LibreYOLOXt.onnx": "https://huggingface.co/LibreYOLO/LibreYOLOXt/resolve/main/LibreYOLOXt.onnx?download=true",
            "LibreYOLOXs.onnx": "https://huggingface.co/LibreYOLO/LibreYOLOXs/resolve/main/LibreYOLOXs.onnx?download=true",
            "LibreYOLOXm.onnx": "https://huggingface.co/LibreYOLO/LibreYOLOXm/resolve/main/LibreYOLOXm.onnx?download=true",
            "LibreYOLOXl.onnx": "https://huggingface.co/LibreYOLO/LibreYOLOXl/resolve/main/LibreYOLOXl.onnx?download=true",
            "LibreYOLOXx.onnx": "https://huggingface.co/LibreYOLO/LibreYOLOXx/resolve/main/LibreYOLOXx.onnx?download=true"
            },
        "LibreDEIM" : {
            "LibreDEIMn.onnx": "https://huggingface.co/LibreYOLO/LibreDEIMn/resolve/main/LibreDEIMn.onnx?download=true",
            "LibreDEIMs.onnx": "https://huggingface.co/LibreYOLO/LibreDEIMs/resolve/main/LibreDEIMs.onnx?download=true",
            "LibreDEIMm.onnx": "https://huggingface.co/LibreYOLO/LibreDEIMm/resolve/main/LibreDEIMm.onnx?download=true",
            "LibreDEIMl.onnx": "https://huggingface.co/LibreYOLO/LibreDEIMl/resolve/main/LibreDEIMl.onnx?download=true",
            "LibreDEIMx.onnx": "https://huggingface.co/LibreYOLO/LibreDEIMx/resolve/main/LibreDEIMx.onnx?download=true"
            },
        "LibreDEIMv2" : {
            "LibreDEIMv2n.onnx": "https://huggingface.co/LibreYOLO/LibreDEIMv2n/resolve/main/LibreDEIMv2n.onnx?download=true",
            "LibreDEIMv2s.onnx": "https://huggingface.co/LibreYOLO/LibreDEIMv2s/resolve/main/LibreDEIMv2s.onnx?download=true",
            "LibreDEIMv2m.onnx": "https://huggingface.co/LibreYOLO/LibreDEIMv2m/resolve/main/LibreDEIMv2m.onnx?download=true",
            "LibreDEIMv2l.onnx": "https://huggingface.co/LibreYOLO/LibreDEIMv2l/resolve/main/LibreDEIMv2l.onnx?download=true",
            "LibreDEIMv2x.onnx": "https://huggingface.co/LibreYOLO/LibreDEIMv2x/resolve/main/LibreDEIMv2x.onnx?download=true"
            }
        }

def progress_callback(block_num, block_size, total_size):
    downloaded = block_num * block_size
    
    percent = min(int(downloaded * 100 / total_size), 100)
    downloaded_md = downloaded / (1024 * 1024)
    total_md = total_size / (1024 * 1024)

    bar_length = 30
    filled_length = int(bar_length * percent // 100)
    bar = '%' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write(f"\r|{bar}| {percent}% ({downloaded_md:.1f}/{total_md:.1f}) MB")
    sys.stdout.flush()

def main():
    os.makedirs(BASE_WEIGHTS_DIR, exist_ok=True)

    for family_name, files in MODELS_DATA.items():
        if not files:
            continue

        family_dir = os.path.join(BASE_WEIGHTS_DIR, family_name)
        os.makedirs(family_dir, exist_ok=True)
        print(f"Starting {family_dir}")

        for file_name, url in files.items():
            destination_path = os.path.join(family_dir, file_name)
            print(f"Starting {destination_path}")
            if os.path.exists(destination_path):
                print(f"{file_name} exists")
                continue 
            
            try:
                urllib.request.urlretrieve(url, destination_path, progress_callback)
                print(f"Successfully")
            except Exception as e:
                print(e)

if __name__ == "__main__":
    main()
                
