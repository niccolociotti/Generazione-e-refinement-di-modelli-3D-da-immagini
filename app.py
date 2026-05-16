import os
import time
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from utils.storage import RUNS_DIR, ensure_dirs, make_session_dir, save_base64_image
from services.image_service import ImageGenerationService
from services.edit_service import ImageEditService
from services.image_service import remote_image_worker_url as image_worker_url
from services.edit_service import remote_image_worker_url as edit_worker_url
from services.model3d_service import Model3DGenerationService   # ← nome corretto

app = Flask(__name__)
ensure_dirs()

image_service  = ImageGenerationService()
edit_service   = ImageEditService()
model3d_service = Model3DGenerationService()

# Carica i modelli una sola volta
try:
    image_service.load_models("z-image-turbo-fp8-e4m3fn.safetensors")

    # Condividi gli stessi modelli con l'editing
    edit_service.set_models(
        image_service.unet,
        image_service.clip,
        image_service.vae
    )
except Exception as e:
    print(f"[startup] Modelli immagine non caricati: {e}")


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def public_file_url(path):
    try:
        relative_path = Path(path).resolve().relative_to(RUNS_DIR.resolve())
    except ValueError:
        return None
    return request.host_url.rstrip("/") + "/files/" + relative_path.as_posix()


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "remote_image_worker_url": os.getenv("REMOTE_IMAGE_WORKER_URL", ""),
        "image_service_worker_url": image_worker_url(),
        "edit_service_worker_url": edit_worker_url(),
    })


@app.get("/files/<path:relative_path>")
def get_file(relative_path):
    base_dir = RUNS_DIR.resolve()
    file_path = (base_dir / relative_path).resolve()
    if not file_path.is_file() or base_dir not in file_path.parents:
        return jsonify({"error": "File non trovato."}), 404
    return send_file(file_path)


@app.post("/generate-image")
def generate_image():
    data = request.get_json(force=True)
    if not data.get("prompt"):
        return jsonify({"error": "Campo 'prompt' obbligatorio."}), 400

    session_dir = make_session_dir(data.get("session_id"))
    started_at = time.time()
    print(
        "[app] /generate-image start "
        f"session={Path(session_dir).name} prompt={data['prompt']!r}",
        flush=True,
    )
    try:
        output_path = image_service.generate(
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt", "blurry, ugly, bad, background"),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            steps=int(data.get("steps", 9)),
            cfg=float(data.get("cfg", 1.0)),
            seed=int(data.get("seed", 0)),
            output_dir=session_dir,
        )
    except Exception as e:
        print(f"[app] /generate-image error after {time.time() - started_at:.1f}s: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

    print(f"[app] /generate-image done after {time.time() - started_at:.1f}s: {output_path}", flush=True)
    return jsonify({
        "status": "ok",
        "image_path": output_path,
        "image_url": public_file_url(output_path),
    })


@app.post("/edit-image")
def edit_image():
    data = request.get_json(force=True)
    if not data.get("image_path"):
        return jsonify({"error": "Campo 'image_path' obbligatorio."}), 400
    if not data.get("prompt"):
        return jsonify({"error": "Campo 'prompt' obbligatorio."}), 400

    session_dir = make_session_dir(data.get("session_id"))

    mask_path = data.get("mask_path")
    if data.get("mask_base64"):
        mask_path = save_base64_image(data["mask_base64"], session_dir, "mask.png")
    if not mask_path:
        return jsonify({"error": "Serve 'mask_path' oppure 'mask_base64'."}), 400

    started_at = time.time()
    print(
        "[app] /edit-image start "
        f"session={Path(session_dir).name} image={data['image_path']!r} prompt={data['prompt']!r}",
        flush=True,
    )
    try:
        output_path = edit_service.inpaint(
            image_path=data["image_path"],
            mask_path=mask_path,
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt", "blurry, ugly, bad"),
            seed=int(data.get("seed", 0)),
            steps=int(data.get("steps", 20)),
            cfg=float(data.get("cfg", 1.0)),
            denoise=float(data.get("denoise", 0.85)),
            output_dir=session_dir,
        )
    except Exception as e:
        print(f"[app] /edit-image error after {time.time() - started_at:.1f}s: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

    print(f"[app] /edit-image done after {time.time() - started_at:.1f}s: {output_path}", flush=True)
    return jsonify({
        "status": "ok",
        "edited_image_path": output_path,
        "edited_image_url": public_file_url(output_path),
    })


@app.post("/generate-3d")
def generate_3d():
    data = request.get_json(force=True)
    if not data.get("image_path"):
        return jsonify({"error": "Campo 'image_path' obbligatorio."}), 400

    session_dir = make_session_dir(data.get("session_id"))
    try:
        model3d_path = model3d_service.generate_from_image(
            image_path=data["image_path"],
            output_dir=session_dir,
            prompt=data.get("prompt", ""),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "status": "ok",
        "model3d_path": model3d_path,
        "model3d_url": public_file_url(model3d_path),
    })


@app.post("/pipeline/run")
def run_pipeline():
    data = request.get_json(force=True)
    if not data.get("prompt"):
        return jsonify({"error": "Campo 'prompt' obbligatorio."}), 400

    session_dir = make_session_dir(data.get("session_id"))
    try:
        # Step 1 — genera immagine
        generated = image_service.generate(
            prompt=data["prompt"],
            negative_prompt=data.get("negative_prompt", "blurry, ugly, bad, background"),
            width=int(data.get("width", 1024)),
            height=int(data.get("height", 1024)),
            steps=int(data.get("steps", 9)),
            cfg=float(data.get("cfg", 1.0)),
            seed=int(data.get("seed", 0)),
            output_dir=session_dir,
        )

        # Step 2 — inpainting (opzionale)
        final_image = generated
        mask_path = data.get("mask_path")
        if data.get("mask_base64"):
            mask_path = save_base64_image(data["mask_base64"], session_dir, "mask.png")
        if mask_path:
            final_image = edit_service.inpaint(
                image_path=generated,
                mask_path=mask_path,
                prompt=data.get("edit_prompt", data["prompt"]),
                negative_prompt=data.get("negative_prompt", "blurry, ugly, bad"),
                seed=int(data.get("edit_seed", 0)),
                steps=int(data.get("edit_steps", 20)),
                cfg=float(data.get("edit_cfg", 1.0)),
                denoise=float(data.get("denoise", 0.85)),
                output_dir=session_dir,
            )

        # Step 3 — genera 3D
        #model3d_path = model3d_service.generate_from_image(
        #    image_path=final_image,
        #    output_dir=session_dir,
        #    prompt=data.get("edit_prompt", data["prompt"]),
        #)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "status": "ok",
        "generated_image_path": generated,
        "generated_image_url": public_file_url(generated),
        "final_image_path": final_image,
        "final_image_url": public_file_url(final_image),
        #"model3d_path": model3d_path,
    })


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5000, debug=debug)
