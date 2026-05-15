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

try:
    with block_optional_imports("comfy_kitchen"):
        from nodes import NODE_CLASS_MAPPINGS

    CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    KSampler = NODE_CLASS_MAPPINGS["KSampler"]()
    VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
    VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
    SetLatentNoiseMask = NODE_CLASS_MAPPINGS["SetLatentNoiseMask"]()
    COMFYUI_AVAILABLE = True
    COMFYUI_IMPORT_ERROR = None
except Exception as exc:
    COMFYUI_AVAILABLE = False
    COMFYUI_IMPORT_ERROR = exc


class ImageEditService:
    def __init__(self):
        self.unet = None
        self.clip = None
        self.vae = None
        self.models_loaded = False

    def set_models(self, unet, clip, vae):
        self.unet = unet
        self.clip = clip
        self.vae = vae
        self.models_loaded = all(model is not None for model in (unet, clip, vae))

    def _ensure_loaded(self):
        if not COMFYUI_AVAILABLE:
            detail = f" Dettaglio: {COMFYUI_IMPORT_ERROR}" if COMFYUI_IMPORT_ERROR else ""
            raise RuntimeError(f"{comfyui_not_found_message()}{detail}")
        if not self.models_loaded:
            raise RuntimeError("Modelli non caricati per l'inpainting.")

    @torch.inference_mode()
    def inpaint(
        self,
        image_path,
        mask_path,
        prompt,
        negative_prompt="blurry ugly bad",
        seed=0,
        steps=20,
        cfg=1.0,
        denoise=0.85,
        output_dir="/tmp/cg_pipeline/outputs",
    ):
        """
        image_path: path dell'immagine originale (PNG)
        mask_path:  path della maschera (PNG in bianco/nero, bianco = area da modificare)
        denoise:    0.0 = nessun cambiamento, 1.0 = cambiamento massimo
        """
        self._ensure_loaded()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
        # --- Carica immagine e maschera come tensori ---
        img  = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")          # grayscale
        mask = mask.resize(img.size, Image.NEAREST)
    
        img_tensor  = torch.from_numpy(np.array(img,  dtype=np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)
    
        # --- Encode immagine in latent space ---
        latent      = VAEEncode.encode(self.vae, img_tensor)[0]          # encode normale
        latent_mask = SetLatentNoiseMask.set_mask(latent, mask_tensor)[0]  # applica maschera
    
        # --- Prompt encoding ---
        positive = CLIPTextEncode.encode(self.clip, prompt)[0]
        negative = CLIPTextEncode.encode(self.clip, negative_prompt)[0]
    
        if seed == 0:
            seed = random.randint(0, 18446744073709551615)
    
        # --- Sampling: denoise < 1.0 preserva le zone non mascherate ---
        samples = KSampler.sample(
            self.unet, seed, steps, cfg,
            "euler", "simple",
            positive, negative,
            latent_mask, denoise=denoise
        )[0]
    
        # --- Decode e salva ---
        decoded = VAEDecode.decode(self.vae, samples)[0].detach()
        out_path = os.path.join(output_dir, f"inpaint_{seed}.png")
        Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0]).save(out_path)
        return out_path
