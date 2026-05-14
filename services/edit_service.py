import os
import random
from pathlib import Path

class ImageEditService:
    def __init__(self):
        self.models_loaded = True

    @torch.inference_mode()
    def inpaint(image_path, mask_path, positive_prompt, negative_prompt="blurry ugly bad",
            seed=0, steps=20, cfg=1.0, denoise=0.85):
        """
        image_path: path dell'immagine originale (PNG)
        mask_path:  path della maschera (PNG in bianco/nero, bianco = area da modificare)
        denoise:    0.0 = nessun cambiamento, 1.0 = cambiamento massimo
        """
        tmp_dir = "/content/ComfyUI/output"
        os.makedirs(tmp_dir, exist_ok=True)
    
        # --- Carica immagine e maschera come tensori ---
        img  = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")          # grayscale
        mask = mask.resize(img.size, Image.NEAREST)
    
        img_tensor  = torch.from_numpy(np.array(img,  dtype=np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0).unsqueeze(0)
    
        # --- Encode immagine in latent space ---
        latent      = VAEEncode.encode(vae, img_tensor)[0]          # encode normale
        latent_mask = SetLatentNoiseMask.set_mask(latent, mask_tensor)[0]  # applica maschera
    
        # --- Prompt encoding ---
        positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
        negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    
        if seed == 0:
            seed = random.randint(0, 18446744073709551615)
    
        # --- Sampling: denoise < 1.0 preserva le zone non mascherate ---
        samples = KSampler.sample(
            unet, seed, steps, cfg,
            "euler", "simple",
            positive, negative,
            latent_mask, denoise=denoise
        )[0]
    
        # --- Decode e salva ---
        decoded = VAEDecode.decode(vae, samples)[0].detach()
        out_path = f"{tmp_dir}/inpaint_result.png"
        Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0]).save(out_path)
        return out_path