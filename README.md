# 🌙 Moondream3 Gradio UI

A web interface for the Moondream3 vision-language model featuring image captioning, visual question answering, object detection, and object pointing.

## Features

- **📝 Image Captioning**: Generate descriptive text for your images (short, normal, long)
- **❓ Visual Q&A**: Ask questions about your images and get intelligent answers
- **🔍 Object Detection**: Detect and localize specific objects with bounding boxes
- **👆 Object Pointing**: Point to specific objects in your images

## Model

This application uses the `PierrunoYT/moondream3-preview` model from Hugging Face. The model will be automatically downloaded on first run.

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA recommended (approx. 19 GB VRAM for full performance)
- Also works on CPU/MPS, but slower

## Installation

### 1. Clone Repository or Download Files

```bash
cd MoonDream3
```

### 2. Create Virtual Environment (recommended)

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

PyTorch must be installed separately as the installation depends on your hardware. Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) or use one of the following commands:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Application

```bash
python app.py
```

The application will start and display a URL (default `http://127.0.0.1:7860`).

### Steps to Use

1. **Open the URL** in your browser
2. **Click "Load Model"** to load Moondream3 (may take a moment on first run as the model downloads)
3. **Select a tab** for the desired function:
   - **Image Captioning**: Upload image, choose length, click "Generate Caption"
   - **Visual Q&A**: Upload image, enter question, click "Ask Question"
   - **Object Detection**: Upload image, enter object type (e.g. "person", "car"), click "Detect Objects"
   - **Object Pointing**: Upload image, enter object type, click "Point to Objects"

## Public Sharing

To make the application publicly accessible (e.g. for demos), change the last line in `app.py`:

```python
demo.launch(share=True)
```

## Alternative: Moondream Cloud API

If you don't have a local GPU, you can use the Moondream Cloud API. Change the model loading code in `app.py`:

```python
import moondream as md

# Instead of AutoModelForCausalLM.from_pretrained(...)
model = md.vl(api_key="YOUR_API_KEY")
```

Get your API key from the [Moondream Dashboard](https://moondream.ai).

## Troubleshooting

### Out of Memory (OOM)
- Try loading the model on CPU (slower but less VRAM)
- Close other GPU-intensive applications

### Model Won't Load
- Ensure `transformers>=4.44.0` is installed
- Check your internet connection (the model downloads from Hugging Face)

### Slow Inference
- GPU is recommended for fast results
- First load and compilation takes longer, then it's faster

## License

See the [Moondream3 Model Card](https://huggingface.co/PierrunoYT/moondream3-preview) for license information.

---

*Powered by [Moondream3](https://huggingface.co/PierrunoYT/moondream3-preview) & [Gradio](https://gradio.app)*
