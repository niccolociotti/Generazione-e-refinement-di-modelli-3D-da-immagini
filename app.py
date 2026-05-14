from flask import Flask, request, jsonify
from services.image_service import ImageGenerationService
from services.edit_service import ImageEditService
from services.model3d_service import model3dGenerationService
from utils.storage import ensure_dirs, make_session_dir, save_base64_image

app = Flask(__name__)
ensure_dirs()

image_service = ImageGenerationService()
edit_service = ImageEditService()
model3d_service = model3dGenerationService()

@app.get('/health')
def health():
    return jsonify({'status': 'ok'})

@app.post('/generate-image')
def generate_image():
    data = request.get_json(force=True)
    prompt = data.get('prompt', '')
    negative_prompt = data.get('negative_prompt', 'blurry, ugly, bad, background, complex background')
    width = int(data.get('width', 1024))
    height = int(data.get('height', 1024))
    steps = int(data.get('steps', 9))
    cfg = float(data.get('cfg', 1.0))
    seed = int(data.get('seed', 0))

    session_dir = make_session_dir(data.get('session_id'))
    output_path = image_service.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=seed,
        output_dir=session_dir,
    )
    return jsonify({'status': 'ok', 'image_path': output_path})

@app.post('/edit-image')
def edit_image():
    data = request.get_json(force=True)
    session_dir = make_session_dir(data.get('session_id'))

    image_path = data.get('image_path')
    mask_b64 = data.get('mask_base64')
    if mask_b64:
        mask_path = save_base64_image(mask_b64, session_dir, 'mask.png')
    else:
        mask_path = data.get('mask_path')

    output_path = edit_service.inpaint(
        image_path=image_path,
        mask_path=mask_path,
        prompt=data.get('prompt', ''),
        negative_prompt=data.get('negative_prompt', 'blurry, ugly, bad'),
        seed=int(data.get('seed', 0)),
        steps=int(data.get('steps', 20)),
        cfg=float(data.get('cfg', 1.0)),
        denoise=float(data.get('denoise', 0.85)),
        output_dir=session_dir,
    )
    return jsonify({'status': 'ok', 'edited_image_path': output_path})

@app.post('/generate-3d')
def generate_3d():
    data = request.get_json(force=True)
    session_dir = make_session_dir(data.get('session_id'))
    model3d_path = model3d_service.generate_from_image(
        image_path=data['image_path'],
        output_dir=session_dir,
        prompt=data.get('prompt', '')
    )
    return jsonify({'status': 'ok', 'model3d_path': model3d_path})

@app.post('/pipeline/run')
def run_pipeline():
    data = request.get_json(force=True)
    session_dir = make_session_dir(data.get('session_id'))

    generated = image_service.generate(
        prompt=data['prompt'],
        negative_prompt=data.get('negative_prompt', 'blurry, ugly, bad, background, complex background'),
        width=int(data.get('width', 1024)),
        height=int(data.get('height', 1024)),
        steps=int(data.get('steps', 9)),
        cfg=float(data.get('cfg', 1.0)),
        seed=int(data.get('seed', 0)),
        output_dir=session_dir,
    )

    final_image = generated
    if data.get('mask_base64') or data.get('mask_path'):
        mask_path = data.get('mask_path')
        if data.get('mask_base64'):
            mask_path = save_base64_image(data['mask_base64'], session_dir, 'mask.png')
        final_image = edit_service.inpaint(
            image_path=generated,
            mask_path=mask_path,
            prompt=data.get('edit_prompt', data['prompt']),
            negative_prompt=data.get('negative_prompt', 'blurry, ugly, bad'),
            seed=int(data.get('edit_seed', 0)),
            steps=int(data.get('edit_steps', 20)),
            cfg=float(data.get('edit_cfg', 1.0)),
            denoise=float(data.get('denoise', 0.85)),
            output_dir=session_dir,
        )

    model3d_path = model3d_service.generate_from_image(
        image_path=final_image,
        output_dir=session_dir,
        prompt=data.get('edit_prompt', data['prompt'])
    )

    return jsonify({
        'status': 'ok',
        'generated_image_path': generated,
        'final_image_path': final_image,
        'model3d_path': model3d_path
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)