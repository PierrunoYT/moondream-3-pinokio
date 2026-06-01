# 🌙 Moondream3 Gradio UI

A web interface for the [Moondream3](https://huggingface.co/moondream/moondream3-preview) vision-language model (9B total params, 2B active MoE) featuring image captioning, visual question answering, object detection, and object pointing.

## Features

- **📝 Image Captioning**: Generate short, normal, or long captions with optional streaming
- **❓ Visual Q&A**: Ask questions about images with reasoning mode — also supports text-only queries
- **🔍 Object Detection**: Detect and localize specific objects with bounding boxes
- **👆 Object Pointing**: Point to specific objects in images
- **⚙️ Advanced Settings**: Temperature, max tokens, streaming, max objects controls

## Model

This application uses the official [`moondream/moondream3-preview`](https://huggingface.co/moondream/moondream3-preview) model from Hugging Face. The model will be automatically downloaded on first run.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA recommended (approx. 19 GB VRAM for full performance)
- Also works on CPU/MPS, but slower

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/PierrunoYT/MoonDream-3-Pinokio.git
cd MoonDream-3-Pinokio
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows (CMD)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

### 3. Install PyTorch

Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) or use:

```bash
# CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies

```bash
pip install -r app/requirements.txt
```

## Usage

### Starting the Application

```bash
python app/app.py
```

Or from the `app` folder: `cd app` then `python app.py`.

The application will start at `http://127.0.0.1:7860`.

### Steps to Use

1. **Open the URL** in your browser
2. **Click "Load Model"** to load Moondream3 (downloads ~18GB on first run)
3. **Select a tab** for the desired function:
   - **Image Captioning**: Upload image, choose length, click "Generate Caption"
   - **Visual Q&A**: Upload image (optional), enter question, toggle reasoning, click "Ask Question"
   - **Object Detection**: Upload image, enter object type, click "Detect"
   - **Object Pointing**: Upload image, enter object type, click "Point"

### Advanced Settings

- **Temperature**: Controls randomness (0 = model default)
- **Max Tokens**: Limit response length (0 = model default)
- **Stream Output**: See responses generated token-by-token
- **Reasoning**: Enable/disable chain-of-thought reasoning in Q&A
- **Max Objects**: Limit detected objects count in detection

## API

Once the server is running at `http://127.0.0.1:7860`, you can access the Gradio API programmatically.

### JavaScript

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");

// Image captioning
const result = await client.predict("/caption", {
  image: new Blob([fs.readFileSync("image.jpg")], { type: "image/jpeg" }),
  length: "normal",
  stream: false,
});
console.log(result.data);
```

### Python

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")

# Image captioning
result = client.predict(
    image="image.jpg",
    length="normal",
    stream=False,
    api_name="/caption"
)
print(result)
```

### cURL

```bash
# Check available API endpoints
curl http://127.0.0.1:7860/info
```

> Full interactive API docs are available at `http://127.0.0.1:7860/?view=api` once the app is running.

## License

See the [Moondream3 Model Card](https://huggingface.co/moondream/moondream3-preview) for license information.

---

*Powered by [Moondream3](https://huggingface.co/moondream/moondream3-preview) & [Gradio](https://gradio.app)*
