import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from services.comfyui_bootstrap import (
    add_comfyui_to_path,
    block_optional_imports,
    comfyui_not_found_message,
    patch_torch_for_comfyui,
)

add_comfyui_to_path()
patch_torch_for_comfyui(torch)

# ComfyUI nodes — disponibili solo se ComfyUI è nel PYTHONPATH o in COMFYUI_PATH
try:
    with block_optional_imports("comfy_kitchen"):
        from nodes import NODE_CLASS_MAPPINGS

    UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
    CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
    VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
    CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    EmptyLatentImage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
    KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
    VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    COMFYUI_AVAILABLE = True
    COMFYUI_IMPORT_ERROR = None
except Exception as exc:
    COMFYUI_AVAILABLE = False
    COMFYUI_IMPORT_ERROR = exc


class ImageGenerationService:
    def __init__(self):
        self.unet = None
        self.clip = None
        self.vae = None
        self.models_loaded = False

    def load_models(self, checkpoint="sd_xl_turbo_1.0_fp16.safetensors"):
        if not COMFYUI_AVAILABLE:
            detail = f" Dettaglio: {COMFYUI_IMPORT_ERROR}" if COMFYUI_IMPORT_ERROR else ""
            raise RuntimeError(f"{comfyui_not_found_message()}{detail}")

        clip_name = os.getenv("Z_IMAGE_CLIP", "qwen_3_4b.safetensors")
        vae_name = os.getenv("Z_IMAGE_VAE", "ae.safetensors")

        self.unet = UNETLoader.load_unet(checkpoint, "fp8_e4m3fn_fast")[0]
        self.clip = CLIPLoader.load_clip(clip_name, type="lumina2")[0]
        self.vae = VAELoader.load_vae(vae_name)[0]
        self.models_loaded = True
        print(f"[ImageGenerationService] Modelli caricati: {checkpoint}, {clip_name}, {vae_name}")

    def _ensure_loaded(self):
        if not self.models_loaded:
            self.load_models()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, ugly, bad, background, complex background",
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        cfg: float = 1.0,
        seed: int = 0,
        denoise: float = 1.0,
        output_dir: str = "/tmp/cg_pipeline/outputs",
    ) -> str:
        self._ensure_loaded()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if seed == 0:
            seed = random.randint(0, 18_446_744_073_709_551_615)

        positive     = CLIPTextEncode.encode(self.clip, prompt)[0]
        negative     = CLIPTextEncode.encode(self.clip, negative_prompt)[0]
        latent_image = EmptyLatentImage.generate(width, height, batch_size=1)[0]

        samples = KSampler.sample(
            self.unet, seed, steps, cfg,
            "euler", "simple",
            positive, negative,
            latent_image, denoise=denoise,
        )[0]

        decoded  = VAEDecode.decode(self.vae, samples)[0].detach()
        out_path = os.path.join(output_dir, f"generated_{seed}.png")
        Image.fromarray(
            np.array(decoded * 255, dtype=np.uint8)[0]
        ).save(out_path)

        print(f"[ImageGenerationService] Immagine salvata: {out_path}")
        return out_path
