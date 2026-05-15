import os
from pathlib import Path


class Model3DGenerationService:
    """
    Generazione modelli 3D da immagine.
    Attualmente stub: salva un .glb vuoto come placeholder.
    Da sostituire con TRELLIS / TripoSR / Shap-E quando disponibile il server GPU.
    """

    def __init__(self):
        self.model_loaded = False

    def load_model(self):
        # TODO: caricare il modello image-to-3D
        # from trellis.pipelines import TrellisImageTo3DPipeline
        # self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        # self.pipeline.cuda()
        self.model_loaded = True
        print("[Model3DGenerationService] Modello 3D caricato.")

    def generate_from_image(
        self,
        image_path: str,
        output_dir: str,
        prompt: str = "",
    ) -> str:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model3d_path = os.path.join(output_dir, "model3d.glb")

        if not self.model_loaded:
            print(
                "[Model3DGenerationService] STUB — modello non integrato. "
                f"File placeholder: {model3d_path}"
            )
            with open(model3d_path, "wb") as f:
                f.write(b"")
            return model3d_path

        # TODO: chiamata reale al modello
        # outputs = self.pipeline.run(image_path)
        # outputs['mesh'].export(model3d_path)
        return model3d_path