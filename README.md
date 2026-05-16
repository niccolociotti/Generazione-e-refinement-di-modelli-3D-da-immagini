# Progetto CG - Flask locale con worker GPU Colab

Questo progetto espone API Flask da usare come backend per Unity. Il server principale gira in locale, mentre la generazione e l'editing delle immagini vengono eseguiti su Colab tramite un worker GPU.

## Architettura

```text
Unity / curl
  -> app.py in locale, porta 5000
  -> colab_worker.py su Colab, porta 5001 via cloudflared
  -> ComfyUI + modelli Z-Image
```

Quando `REMOTE_IMAGE_WORKER_URL` e' impostata, `app.py` inoltra i job a Colab e salva i risultati nella cartella locale `runs/`.

## File principali

```text
app.py                       Server Flask locale per Unity
colab_worker.py              Worker Flask da eseguire su Colab
server.ipynb                 Notebook per preparare Colab e avviare il worker
requirements.txt             Dipendenze locali Mac
requirements-colab.txt       Dipendenze Colab
services/image_service.py    Generazione immagini, locale o remota
services/edit_service.py     Editing/inpainting, locale o remoto
utils/storage.py             Gestione cartelle runs/ e immagini base64
```

## Setup Colab

1. Apri `server.ipynb` in VS Code.
2. Seleziona un kernel Colab con GPU.
3. Esegui le celle in ordine.
4. La cella del worker deve mostrare:

```text
Health locale: 200 {"models_loaded":true,"status":"ok"}
Worker pronto. Ora puoi eseguire la cella cloudflared.
```

5. La cella cloudflared stampa un comando simile:

```bash
export REMOTE_IMAGE_WORKER_URL="https://qualcosa.trycloudflare.com"
```


## Avvio server locale sul Mac

Nel terminale locale:

```bash
cd "/Users/niccolociotti/Desktop/Progetto CG"
export REMOTE_IMAGE_WORKER_URL="https://URL_STAMPATO_DA_COLAB.trycloudflare.com"
python app.py
```

Il log deve contenere:

```text
[ImageGenerationService] Uso worker remoto: https://...
Running on http://127.0.0.1:5000
```

## Health check

Server locale:

```bash
curl http://127.0.0.1:5000/health
```

Worker Colab:

```bash
curl "$REMOTE_IMAGE_WORKER_URL/health"
```

Risposta attesa dal worker:

```json
{"models_loaded":true,"status":"ok"}
```

## Generare un'immagine

Richiesta minima:

```bash
curl -X POST http://127.0.0.1:5000/generate-image \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","prompt":"a futuristic white sneaker, product render, clean background"}'
```

Il backend usa questi default:

```json
{
  "negative_prompt": "blurry, ugly, bad, background",
  "width": 1024,
  "height": 1024,
  "steps": 9,
  "cfg": 1.0,
  "seed": 0
}
```

Se vuoi modificarli a runtime, aggiungili alla richiesta:

```bash
curl -X POST http://127.0.0.1:5000/generate-image \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test","prompt":"a futuristic white sneaker, product render, clean background","width":768,"height":768,"steps":6,"cfg":1.0}'
```

La risposta contiene:

```json
{
  "status": "ok",
  "image_path": ".../runs/test/generated_....png",
  "image_url": "http://127.0.0.1:5000/files/test/generated_....png"
}
```

Apri `image_url` nel browser o usalo in Unity con `UnityWebRequestTexture.GetTexture`.

## Testare edit-image

`/edit-image` richiede:

- `image_path`: path locale sul Mac dell'immagine da modificare;
- `mask_path` oppure `mask_base64`: maschera bianco/nero PNG;
- `prompt`: descrizione della modifica.

Bianco nella maschera = area da rigenerare. Nero = area da preservare.

### Creare una maschera di test

Esempio: crea una maschera bianca al centro dell'immagine generata.

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw

image_path = Path("runs/test/generated_4351029803259878957.png")
mask_path = Path("runs/test/mask_center.png")

img = Image.open(image_path)
mask = Image.new("L", img.size, 0)
draw = ImageDraw.Draw(mask)
w, h = img.size
draw.ellipse((w * 0.30, h * 0.25, w * 0.70, h * 0.75), fill=255)
mask.save(mask_path)
print(mask_path)
PY
```

Se il nome dell'immagine e' diverso, sostituisci `image_path`.

### Chiamare edit-image

Richiesta minima:

```bash
curl -X POST http://127.0.0.1:5000/edit-image \
  -H "Content-Type: application/json" \
  -d '{"session_id":"edit-test","image_path":"runs/test/generated_4351029803259878957.png","mask_path":"runs/test/mask_center.png","prompt":"make the masked area glossy red"}'
```

Il backend usa questi default:

```json
{
  "negative_prompt": "blurry, ugly, bad",
  "seed": 0,
  "steps": 20,
  "cfg": 1.0,
  "denoise": 0.85
}
```

Se vuoi modificarli a runtime, aggiungili alla richiesta:

```bash
curl -X POST http://127.0.0.1:5000/edit-image \
  -H "Content-Type: application/json" \
  -d '{"session_id":"edit-test","image_path":"runs/test/generated_4351029803259878957.png","mask_path":"runs/test/mask_center.png","prompt":"make the masked area glossy red","steps":12,"cfg":1.0,"denoise":0.85}'
```

La risposta contiene:

```json
{
  "status": "ok",
  "edited_image_path": ".../runs/edit-test/inpaint_....png",
  "edited_image_url": "http://127.0.0.1:5000/files/edit-test/inpaint_....png"
}
```

## Monitoraggio

Sul Mac, il terminale di `app.py` mostra l'avanzamento del job remoto:

```text
Job remoto ... status=running elapsed=...
Job remoto ... status=done elapsed=...
```

Su Colab:

```python
!tail -f /content/ProgettoCG/worker.log
```

Per verificare GPU:

```python
!nvidia-smi
```

## Problemi comuni

### 502 / 530 da trycloudflare

Il tunnel e' morto o l'URL e' vecchio. Riesegui la cella cloudflared e riavvia `app.py` sul Mac con il nuovo `REMOTE_IMAGE_WORKER_URL`.

### 524 da Cloudflare

Una richiesta e' rimasta aperta troppo a lungo. Il progetto usa endpoint asincroni `/jobs/...` per evitarlo; assicurati di aver fatto `git pull` su Colab e di aver riavviato worker e `app.py`.

### models_loaded false

Il worker Colab non ha caricato i modelli. Guarda:

```python
!tail -n 120 /content/ProgettoCG/worker.log
```

### Out of memory o instabilita'

Riduci parametri:

```json
{"width":768,"height":768,"steps":6}
```
