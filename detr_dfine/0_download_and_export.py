from config import hf_models, MODELS_ONNX_DIR, MODELS_PT_DIR, device
from huggingface_hub import hf_hub_download
from libreyolo import LibreYOLO

for model_full_name in hf_models:
    model_name = model_full_name.split("/")[-1]
    onnx_path = MODELS_ONNX_DIR / f"{model_name}.onnx"
    pt_path = MODELS_PT_DIR / f"{model_name}.pt"
    if onnx_path.exists():
        print("onnx")
        continue
    if pt_path.exists():
        print("pt")
        model = LibreYOLO(str(pt_path), device=device)
    else:
        print("download")
        weights = hf_hub_download(str(model_full_name), filename=f"{model_name}.pt")
        model = LibreYOLO(weights, device=device)
    model.export(format="onnx", output_path=str(onnx_path))
