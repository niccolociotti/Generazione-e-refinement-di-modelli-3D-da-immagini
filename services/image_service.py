import os
import random
from pathlib import Path

class ImageGenerationService:
    def __init__(self):
        self.models_loaded = False
        self._lazy_init()

    def _lazy_init(self):
        self.models_loaded = True

    @torch.inference_mode()
    def generate(positive_prompt, negative_prompt="blurry ugly bad",
             width=1024, height=1024, steps=9, cfg=1.0,
             seed=0, denoise=1.0):

        tmp_dir = "/content/ComfyUI/output"
        os.makedirs(tmp_dir, exist_ok=True)
    
        if seed == 0:
            seed = random.randint(0, 18446744073709551615)
    
        positive     = CLIPTextEncode.encode(clip, positive_prompt)[0]
        negative     = CLIPTextEncode.encode(clip, negative_prompt)[0]
        latent_image = EmptyLatentImage.generate(width, height, batch_size=1)[0]
        samples      = KSampler.sample(unet, seed, steps, cfg,
                                       "euler", "simple",
                                       positive, negative,
                                       latent_image, denoise=denoise)[0]
        decoded  = VAEDecode.decode(vae, samples)[0].detach()
        out_path = f"{tmp_dir}/z_image_turbo.png"
        Image.fromarray(np.array(decoded * 255, dtype=np.uint8)[0]).save(out_path)
    
        print(f"Immagine salvata in: {out_path}")
        return output_path