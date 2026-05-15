import base64
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

os.environ.pop("REMOTE_IMAGE_WORKER_URL", None)
os.environ.setdefault("COMFYUI_PATH", str(Path(__file__).resolve().parent / "ComfyUI"))

from services.edit_service import ImageEditService
from services.image_service import ImageGenerationService


app = Flask(__name__)

image_service = ImageGenerationService()
edit_service = ImageEditService()

try:
    image_service.load_models("z-image-turbo-fp8-e4m3fn.safetensors")
    edit_service.set_models(image_service.unet, image_service.clip, image_service.vae)
except Exception as exc:
    print(f"[colab_worker] Modelli non caricati: {exc}")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def file_to_base64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def base64_to_file(data, path):
    if "," in data:
        data = data.split(",", 1)[1]
    Path(path).write_bytes(base64.b64decode(data))


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": image_service.models_loaded,
    })


@app.post("/generate-image")
def generate_image():
    data = request.get_json(force=True)
    if not data.get("prompt"):
        return jsonify({"error": "Campo 'prompt' obbligatorio."}), 400

    try:
        output_path = image_service.generate(
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt", "blurry, ugly, bad, background"),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            steps=int(data.get("steps", 9)),
            cfg=float(data.get("cfg", 1.0)),
            seed=int(data.get("seed", 0)),
            denoise=float(data.get("denoise", 1.0)),
            output_dir=tempfile.mkdtemp(prefix="cg_worker_"),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "status": "ok",
        "image_base64": file_to_base64(output_path),
        "seed": Path(output_path).stem.removeprefix("generated_"),
    })


@app.post("/edit-image")
def edit_image():
    data = request.get_json(force=True)
    if not data.get("image_base64"):
        return jsonify({"error": "Campo 'image_base64' obbligatorio."}), 400
    if not data.get("mask_base64"):
        return jsonify({"error": "Campo 'mask_base64' obbligatorio."}), 400
    if not data.get("prompt"):
        return jsonify({"error": "Campo 'prompt' obbligatorio."}), 400

    work_dir = Path(tempfile.mkdtemp(prefix="cg_worker_edit_"))
    image_path = work_dir / "image.png"
    mask_path = work_dir / "mask.png"
    base64_to_file(data["image_base64"], image_path)
    base64_to_file(data["mask_base64"], mask_path)

    try:
        output_path = edit_service.inpaint(
            image_path=str(image_path),
            mask_path=str(mask_path),
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt", "blurry, ugly, bad"),
            seed=int(data.get("seed", 0)),
            steps=int(data.get("steps", 20)),
            cfg=float(data.get("cfg", 1.0)),
            denoise=float(data.get("denoise", 0.85)),
            output_dir=str(work_dir),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify({
        "status": "ok",
        "image_base64": file_to_base64(output_path),
        "seed": Path(output_path).stem.removeprefix("inpaint_"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("WORKER_PORT", "5001")), debug=False)
