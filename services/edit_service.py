import base64
import os
import random
import time
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

REMOTE_IMAGE_WORKER_URL = os.getenv("REMOTE_IMAGE_WORKER_URL", "").rstrip("/")

if REMOTE_IMAGE_WORKER_URL:
    COMFYUI_AVAILABLE = False
else:
    try:
        from nodes import (
            CLIPTextEncode,
            KSampler,
            VAEDecode,
            VAEEncode,
            SetLatentNoiseMask,
        )
        COMFYUI_AVAILABLE = True
    except Exception:
        COMFYUI_AVAILABLE = False


class ImageEditService:
    def __init__(self, unet=None, clip=None, vae=None):
        self.unet = unet
        self.clip = clip
        self.vae  = vae
        self.models_loaded = unet is not None

    def set_models(self, unet, clip, vae):
        """Permette di condividere i modelli già caricati da ImageGenerationService."""
        if REMOTE_IMAGE_WORKER_URL:
            self.models_loaded = True
            return
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
        if REMOTE_IMAGE_WORKER_URL:
            return self._inpaint_remote(
                image_path,
                mask_path,
                prompt,
                negative_prompt,
                seed,
                steps,
                cfg,
                denoise,
                output_dir,
            )

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

    def _inpaint_remote(
        self,
        image_path,
        mask_path,
        prompt,
        negative_prompt,
        seed,
        steps,
        cfg,
        denoise,
        output_dir,
    ):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        image_base64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        mask_base64 = base64.b64encode(Path(mask_path).read_bytes()).decode("ascii")

        response = requests.post(
            f"{REMOTE_IMAGE_WORKER_URL}/jobs/edit-image",
            json={
                "image_base64": image_base64,
                "mask_base64": mask_base64,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "denoise": denoise,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        job_id = payload.get("job_id")
        if not job_id:
            raise RuntimeError(payload.get("error", "Il worker remoto non ha restituito job_id."))

        started_at = time.time()
        timeout = int(os.getenv("REMOTE_IMAGE_TIMEOUT", "900"))
        poll_interval = float(os.getenv("REMOTE_IMAGE_POLL_INTERVAL", "5"))
        deadline = time.time() + timeout
        last_status_log = 0
        while time.time() < deadline:
            status_response = requests.get(
                f"{REMOTE_IMAGE_WORKER_URL}/jobs/{job_id}",
                timeout=30,
            )
            status_response.raise_for_status()
            payload = status_response.json()
            status = payload.get("status")
            elapsed = time.time() - started_at
            if elapsed - last_status_log >= 15:
                print(
                    f"[ImageEditService] Job remoto {job_id} status={status} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                last_status_log = elapsed
            if status == "done":
                break
            if status == "error":
                raise RuntimeError(payload.get("error", "Job remoto fallito."))
            time.sleep(poll_interval)
        else:
            raise TimeoutError(f"Timeout job remoto {job_id} dopo {timeout}s")

        if payload.get("status") != "done" or not payload.get("image_base64"):
            raise RuntimeError(payload.get("error", "Risposta non valida dal worker remoto."))

        if seed == 0:
            seed = payload.get("seed", "remote")
        out_path = os.path.join(output_dir, f"inpaint_{seed}.png")
        image_data = payload["image_base64"]
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        Path(out_path).write_bytes(base64.b64decode(image_data))
        return out_path
