"""
Moondream3 Gradio UI
Web interface for the Moondream3 vision-language model.
"""

import os
import threading
import warnings

# torch.compile is opt-in: it needs a working C compiler / Triton and silently
# costs minutes of warm-up when it half-works. Set MOONDREAM_COMPILE=1 to enable.
COMPILE_ENABLED = os.environ.get("MOONDREAM_COMPILE", "0").lower() in ("1", "true", "yes")
if not COMPILE_ENABLED:
    # Must be set before torch is imported.
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)

import torch
print(f"PyTorch version: {torch.__version__}")

# Monkey-patch BlockMask to add missing seq_lengths attribute
try:
    from torch.nn.attention.flex_attention import BlockMask
    original_init = BlockMask.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, 'seq_lengths'):
            if hasattr(self, 'shape'):
                self.seq_lengths = self.shape
            elif hasattr(self, 'kv_num_blocks'):
                self.seq_lengths = (self.kv_num_blocks[-1] * 128, self.kv_num_blocks[-1] * 128)
            else:
                self.seq_lengths = None

    BlockMask.__init__ = patched_init
    print("✓ Applied BlockMask patch for seq_lengths compatibility")
except Exception as e:
    import traceback
    print(f"⚠️ Could not patch BlockMask: {e}")
    traceback.print_exc()

import gradio as gr
from transformers import AutoModelForCausalLM
from PIL import ImageDraw

model = None
_model_lock = threading.Lock()

# The model keeps a single shared KV cache, so two requests running at once
# (e.g. a caption and a query from different tabs or users) corrupt each other.
# Every event that touches the model shares this queue slot to run one at a time.
MODEL_CONCURRENCY_ID = "moondream_model"


def select_device():
    """Pick the best available device and matching dtype."""
    if torch.cuda.is_available():
        try:
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        except Exception as e:  # driver present but device query failed
            print(f"CUDA available (device name unavailable: {e})")
        return "cuda", torch.bfloat16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def load_model():
    """Load the Moondream3 model."""
    global model
    if model is not None:
        return "Model already loaded!"

    # Serialize concurrent "Load Model" clicks so the weights are only pulled once.
    with _model_lock:
        if model is not None:
            return "Model already loaded!"
        return _load_model_locked()


def _load_model_locked():
    global model
    device, dtype = select_device()

    try:
        print(f"Loading Moondream3 on {device}...")

        model = AutoModelForCausalLM.from_pretrained(
            "moondream/moondream3-preview",
            trust_remote_code=True,
            dtype=dtype,
            device_map={"": device},
        )

        print("✓ Model loaded successfully!")

        if not COMPILE_ENABLED:
            return (
                f"Model loaded on {device}!\n\n"
                "Compilation disabled (set MOONDREAM_COMPILE=1 to enable)."
            )

        try:
            print("Compiling model for optimized inference...")
            model.compile()
            print("✓ Model compiled successfully!")
            return f"Model loaded and compiled successfully on {device}!\n\nRunning with full optimization."
        except Exception as compile_error:
            print(f"⚠️ Compilation failed: {compile_error}")
            return f"Model loaded on {device}!\n\n⚠️ Compilation skipped: {str(compile_error)[:100]}\n\nModel will work but may be slower."

    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        return f"Error loading model: {error_msg}"


def check_model():
    """Check if the model is loaded."""
    if model is None:
        return False, "Please load the model first by clicking 'Load Model'!"
    return True, None


def _clamp01(value):
    """Clamp a normalized coordinate into the valid [0, 1] range."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def build_settings(temperature, max_tokens):
    """Build settings dict from UI values."""
    settings = {}
    if temperature is not None and temperature > 0:
        settings["temperature"] = temperature
    if max_tokens is not None and max_tokens > 0:
        settings["max_tokens"] = int(max_tokens)
    return settings if settings else None


def caption_image(image, length, temperature, max_tokens, stream):
    """Generate a caption for the image."""
    loaded, error = check_model()
    if not loaded:
        yield error
        return

    if image is None:
        yield "Please upload an image!"
        return

    try:
        settings = build_settings(temperature, max_tokens)
        kwargs = {"length": length}
        if settings:
            kwargs["settings"] = settings

        if stream:
            kwargs["stream"] = True
            result = model.caption(image, **kwargs)
            caption_stream = result.get("caption", result) if isinstance(result, dict) else result
            text = ""
            for chunk in caption_stream:
                text += chunk if isinstance(chunk, str) else str(chunk)
                yield text
        else:
            result = model.caption(image, **kwargs)
            if isinstance(result, dict):
                yield result.get("caption", str(result))
            else:
                yield str(result)
    except Exception as e:
        yield f"Error: {e}"


def answer_question(image, question, reasoning, temperature, max_tokens, stream):
    """Answer a question about the image (or text-only)."""
    loaded, error = check_model()
    if not loaded:
        yield error
        return

    if not question or question.strip() == "":
        yield "Please enter a question!"
        return

    try:
        settings = build_settings(temperature, max_tokens)
        kwargs = {"question": question, "reasoning": reasoning}
        if image is not None:
            kwargs["image"] = image
        if settings:
            kwargs["settings"] = settings

        if stream:
            kwargs["stream"] = True
            result = model.query(**kwargs)
            answer_stream = result.get("answer", result) if isinstance(result, dict) else result
            text = ""
            for chunk in answer_stream:
                text += chunk if isinstance(chunk, str) else str(chunk)
                yield text
        else:
            result = model.query(**kwargs)
            if isinstance(result, dict):
                yield result.get("answer", str(result))
            else:
                yield str(result)
    except Exception as e:
        yield f"Error: {e}"


def detect_objects(image, object_type, max_objects):
    """Detect objects in the image."""
    loaded, error = check_model()
    if not loaded:
        return None, error

    if image is None:
        return None, "Please upload an image!"

    if not object_type or object_type.strip() == "":
        return None, "Please specify an object type!"

    try:
        kwargs = {}
        if max_objects is not None and max_objects > 0:
            kwargs["settings"] = {"max_objects": int(max_objects)}

        result = model.detect(image, object_type.strip(), **kwargs)

        if isinstance(result, dict):
            objects = result.get("objects", [])
        else:
            return image, f"Unexpected result: {str(result)}"

        if not objects:
            return image, f"No '{object_type}' detected."

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size

        drawn = 0
        for obj in objects:
            x0 = int(_clamp01(obj.get("x_min", 0)) * width)
            y0 = int(_clamp01(obj.get("y_min", 0)) * height)
            x1 = int(_clamp01(obj.get("x_max", 0)) * width)
            y1 = int(_clamp01(obj.get("y_max", 0)) * height)

            # PIL raises when the box is given bottom-right first.
            x_min, x_max = sorted((x0, x1))
            y_min, y_max = sorted((y0, y1))
            if x_max <= x_min or y_max <= y_min:
                continue

            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            label = obj.get("label", object_type)
            draw.text((x_min, max(0, y_min - 20)), str(label), fill="red")
            drawn += 1

        return annotated, f"✓ Detected {drawn} object(s)."
    except Exception as e:
        return image, f"Error: {e}"


def point_objects(image, object_type):
    """Point to objects in the image."""
    loaded, error = check_model()
    if not loaded:
        return None, error

    if image is None:
        return None, "Please upload an image!"

    if not object_type or object_type.strip() == "":
        return None, "Please specify an object type!"

    try:
        result = model.point(image, object_type.strip())

        if isinstance(result, dict):
            points = result.get("points", [])
        else:
            return image, f"Unexpected result: {str(result)}"

        if not points:
            return image, f"No '{object_type}' found."

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size

        for point in points:
            x = int(_clamp01(point.get("x", 0)) * width)
            y = int(_clamp01(point.get("y", 0)) * height)

            radius = 20
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                        outline="#0066CC", width=4)
            inner_radius = 16
            draw.ellipse([x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius],
                        fill="#4DA6FF", outline="#0066CC", width=2)

        return annotated, f"✓ Found {len(points)} point(s)."
    except Exception as e:
        return image, f"Error: {e}"


# Gradio interface
with gr.Blocks(title="Moondream3 Vision AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌙 Moondream3 Vision AI

        Vision-language model with image captioning, visual Q&A, object detection, and pointing.

        **Click "Load Model" to start.**
        """
    )

    with gr.Row():
        load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
        load_status = gr.Textbox(label="Status", value="Model not loaded", interactive=False, scale=3, lines=3)

    load_btn.click(fn=load_model, outputs=load_status, api_name="load_model",
                   concurrency_id=MODEL_CONCURRENCY_ID)

    gr.Markdown("---")

    with gr.Tabs():
        with gr.TabItem("📝 Image Captioning"):
            with gr.Row():
                with gr.Column():
                    caption_image_input = gr.Image(type="pil", label="Upload Image")
                    caption_length = gr.Radio(choices=["short", "normal", "long"], value="normal", label="Length")
                    with gr.Accordion("Advanced Settings", open=False):
                        caption_temperature = gr.Slider(0, 1.5, value=0, step=0.1, label="Temperature (0 = default)")
                        caption_max_tokens = gr.Slider(0, 1024, value=0, step=64, label="Max Tokens (0 = default)")
                        caption_stream = gr.Checkbox(label="Stream Output", value=False)
                    caption_btn = gr.Button("Generate Caption", variant="primary")
                with gr.Column():
                    caption_output = gr.Textbox(label="Caption", lines=5)

            caption_btn.click(
                fn=caption_image,
                inputs=[caption_image_input, caption_length, caption_temperature, caption_max_tokens, caption_stream],
                outputs=caption_output,
                api_name="caption",
                concurrency_id=MODEL_CONCURRENCY_ID,
            )

        with gr.TabItem("❓ Visual Q&A"):
            with gr.Row():
                with gr.Column():
                    vqa_image_input = gr.Image(type="pil", label="Upload Image (optional for text-only)")
                    vqa_question = gr.Textbox(label="Question", placeholder="What is in this image?", lines=2)
                    vqa_reasoning = gr.Checkbox(label="Enable Reasoning", value=True)
                    with gr.Accordion("Advanced Settings", open=False):
                        vqa_temperature = gr.Slider(0, 1.5, value=0, step=0.1, label="Temperature (0 = default)")
                        vqa_max_tokens = gr.Slider(0, 2048, value=0, step=64, label="Max Tokens (0 = default)")
                        vqa_stream = gr.Checkbox(label="Stream Output", value=False)
                    vqa_btn = gr.Button("Ask Question", variant="primary")
                with gr.Column():
                    vqa_output = gr.Textbox(label="Answer", lines=8)

            vqa_btn.click(
                fn=answer_question,
                inputs=[vqa_image_input, vqa_question, vqa_reasoning, vqa_temperature, vqa_max_tokens, vqa_stream],
                outputs=vqa_output,
                api_name="query",
                concurrency_id=MODEL_CONCURRENCY_ID,
            )

        with gr.TabItem("🔍 Object Detection"):
            with gr.Row():
                with gr.Column():
                    detect_image_input = gr.Image(type="pil", label="Upload Image")
                    detect_object_type = gr.Textbox(label="Object", placeholder="person, car, dog", lines=1)
                    detect_max_objects = gr.Slider(0, 50, value=0, step=1, label="Max Objects (0 = default)")
                    detect_btn = gr.Button("Detect", variant="primary")
                with gr.Column():
                    detect_image_output = gr.Image(type="pil", label="Result")
                    detect_text_output = gr.Textbox(label="Info", lines=3)

            detect_btn.click(
                fn=detect_objects,
                inputs=[detect_image_input, detect_object_type, detect_max_objects],
                outputs=[detect_image_output, detect_text_output],
                api_name="detect",
                concurrency_id=MODEL_CONCURRENCY_ID,
            )

        with gr.TabItem("👆 Object Pointing"):
            with gr.Row():
                with gr.Column():
                    point_image_input = gr.Image(type="pil", label="Upload Image")
                    point_object_type = gr.Textbox(label="Object", placeholder="person, car, dog", lines=1)
                    point_btn = gr.Button("Point", variant="primary")
                with gr.Column():
                    point_image_output = gr.Image(type="pil", label="Result")
                    point_text_output = gr.Textbox(label="Info", lines=3)

            point_btn.click(
                fn=point_objects,
                inputs=[point_image_input, point_object_type],
                outputs=[point_image_output, point_text_output],
                api_name="point",
                concurrency_id=MODEL_CONCURRENCY_ID,
            )

    gr.Markdown(
        """
        ---
        *Powered by [Moondream3](https://huggingface.co/moondream/moondream3-preview) & [Gradio](https://gradio.app)*
        """
    )


if __name__ == "__main__":
    print("Starting Moondream3...")
    demo.queue()
    launch_kwargs = {
        "server_name": os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        "share": False,
    }
    # Leave the port unset unless asked for, so Gradio can fall forward to the
    # next free port when 7860 is already taken by another app.
    port = os.environ.get("GRADIO_SERVER_PORT")
    if port:
        launch_kwargs["server_port"] = int(port)
    demo.launch(**launch_kwargs)
