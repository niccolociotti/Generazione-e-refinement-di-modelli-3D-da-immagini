import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from nodes import (
        CLIPTextEncode,
        KSampler,
        VAEDecode,
        VAEEncode,
        SetLatentNoiseMask,
    )
    COMFYUI_AVAILABLE = True
except ImportError:
    COMFYUI_AVAILABLE = False


class ImageEditService:
    def __init__(self, unet=None, clip=None, vae=None):
        self.unet = unet
        self.clip = clip
        self.vae  = vae
        self.models_loaded = unet is not None

    def set_models(self, unet, clip, vae):
        """Permette di condividere i modelli già caricati da ImageGenerationService."""
        self.unet = unet
        self.clip = clip
        self.vae  = vae
        self.models_loaded = True

    def _ensure_loaded(self):
        if not self.models_loaded:
            raise RuntimeError(
                "Modelli non caricati. Chiama set_models() oppure passa "
                "unet/clip/vae al costruttore."
            )

    @torch.inference_mode()
    def inpaint(
        self,
        image_path: str,
        mask_path: str,
        prompt: str,
        negative_prompt: str = "blurry, ugly, bad",
        seed: int = 0,
        steps: int = 20,
        cfg: float = 1.0,
        denoise: float = 0.85,
        output_dir: str = "/tmp/cg_pipeline/outputs",
    ) -> str:
        """
        Inpainting su un'immagine con maschera binaria.
        mask_path: PNG grayscale, bianco = area da rigenerare
        denoise:   0.0 = nessun cambiamento, 1.0 = rigenerazione totale
        """
        self._ensure_loaded()
        if not COMFYUI_AVAILABLE:
            raise RuntimeError("ComfyUI non trovato nel PYTHONPATH.")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        img  = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(img.size, Image.NEAREST)

        img_tensor  = torch.from_numpy(
            np.array(img,  dtype=np.float32) / 255.0
        ).unsqueeze(0)                           # [1, H, W, 3]
        mask_tensor = torch.from_numpy(
            np.array(mask, dtype=np.float32) / 255.0
        ).unsqueeze(0)                           # [1, H, W]

        latent      = VAEEncode.encode(self.vae, img_tensor)[0]
        latent_mask = SetLatentNoiseMask.set_mask(latent, mask_tensor)[0]

        positive = CLIPTextEncode.encode(self.clip, prompt)[0]
        negative = CLIPTextEncode.encode(self.clip, negative_prompt)[0]

        if seed == 0:
            seed = random.randint(0, 18_446_744_073_709_551_615)

        samples = KSampler.sample(
            self.unet, seed, steps, cfg,
            "euler", "simple",
            positive, negative,
            latent_mask, denoise=denoise,
        )[0]

        decoded  = VAEDecode.decode(self.vae, samples)[0].detach()
        out_path = os.path.join(output_dir, f"inpaint_{seed}.png")
        Image.fromarray(
            np.array(decoded * 255, dtype=np.uint8)[0]
        ).save(out_path)

        print(f"[ImageEditService] Inpainting salvato: {out_path}")
        return out_path