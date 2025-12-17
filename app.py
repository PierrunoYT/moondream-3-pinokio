"""
Moondream3 Gradio UI - With FlexAttention Workaround
This version patches the FlexAttention BlockMask issue
"""

import os
import warnings
import sys

# Must patch BEFORE importing torch
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
        # Add the missing seq_lengths attribute
        if not hasattr(self, 'seq_lengths'):
            # Try to infer from other attributes
            if hasattr(self, 'shape'):
                self.seq_lengths = self.shape
            elif hasattr(self, 'kv_num_blocks'):
                self.seq_lengths = (self.kv_num_blocks[-1] * 128, self.kv_num_blocks[-1] * 128)
            else:
                self.seq_lengths = None
    
    BlockMask.__init__ = patched_init
    print("✓ Applied BlockMask patch for seq_lengths compatibility")
except Exception as e:
    print(f"⚠️ Could not patch BlockMask: {e}")

import gradio as gr
from transformers import AutoModelForCausalLM
from PIL import Image, ImageDraw

# Global model variable
model = None


def load_model():
    """Load the Moondream3 model."""
    global model
    if model is not None:
        return "Model already loaded!"
    
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    
    try:
        print(f"Loading Moondream3 on {device}...")
        
        # Try loading without attn_implementation first
        model = AutoModelForCausalLM.from_pretrained(
            "PierrunoYT/moondream3-preview",
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": device},
        )
        
        print("✓ Model loaded successfully!")
        
        # Try to compile for better performance
        try:
            print("Compiling model for optimized inference...")
            model.compile()
            print("✓ Model compiled successfully!")
            return f"Model loaded and compiled successfully on {device}!\n\nRunning with full optimization."
        except Exception as compile_error:
            print(f"⚠️ Compilation failed: {compile_error}")
            print("Running without compilation (slower but stable)")
            return f"Model loaded on {device}!\n\n⚠️ Compilation skipped due to: {str(compile_error)[:100]}\n\nModel will work but may be slower."
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error: {error_msg}")
        
        if "FlexAttention" in error_msg or "BlockMask" in error_msg:
            return f"❌ FlexAttention error persists!\n\nThe model requires features not available in your PyTorch version.\n\nRecommended solution: Use Moondream Cloud API\n1. Get API key from moondream.ai\n2. Use 'moondream' Python package\n\nAlternative: Use Moondream2 (stable, no FlexAttention)\nChange model to: vikhyat/moondream2"
        else:
            return f"Error loading model: {error_msg}"


def check_model():
    """Check if the model is loaded."""
    if model is None:
        return False, "Please load the model first by clicking 'Load Model'!"
    return True, None


def caption_image(image, length="normal"):
    """Generate a caption for the image."""
    loaded, error = check_model()
    if not loaded:
        return error
    
    if image is None:
        return "Please upload an image!"
    
    try:
        result = model.caption(image, length=length)
        if isinstance(result, dict):
            return result.get("caption", str(result))
        return str(result)
    except Exception as e:
        error_msg = str(e)
        if "BlockMask" in error_msg or "seq_lengths" in error_msg:
            return f"FlexAttention compatibility error!\n\nThis PyTorch version is incompatible with Moondream3.\n\nSuggested solutions:\n1. Use Moondream Cloud API (moondream.ai)\n2. Use Moondream2 instead (vikhyat/moondream2)\n\nError: {error_msg}"
        return f"Error: {error_msg}"


def answer_question(image, question):
    """Answer a question about the image."""
    loaded, error = check_model()
    if not loaded:
        return error
    
    if image is None:
        return "Please upload an image!"
    
    if not question or question.strip() == "":
        return "Please enter a question!"
    
    try:
        result = model.query(image, question)
        if isinstance(result, dict):
            return result.get("answer", str(result))
        return str(result)
    except Exception as e:
        error_msg = str(e)
        if "BlockMask" in error_msg or "seq_lengths" in error_msg:
            return f"FlexAttention compatibility error!\n\nError: {error_msg}"
        return f"Error: {error_msg}"


def detect_objects(image, object_type):
    """Detect objects in the image."""
    loaded, error = check_model()
    if not loaded:
        return None, error
    
    if image is None:
        return None, "Please upload an image!"
    
    if not object_type or object_type.strip() == "":
        return None, "Please specify an object type!"
    
    try:
        result = model.detect(image, object_type.strip())
        
        if isinstance(result, dict):
            objects = result.get("objects", [])
        else:
            return image, f"Unexpected result: {str(result)}"
        
        if not objects:
            return image, f"No '{object_type}' detected."
        
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size
        
        for obj in objects:
            x_min = int(obj.get("x_min", 0) * width)
            y_min = int(obj.get("y_min", 0) * height)
            x_max = int(obj.get("x_max", 0) * width)
            y_max = int(obj.get("y_max", 0) * height)
            
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            label = obj.get("label", object_type)
            draw.text((x_min, max(0, y_min - 20)), label, fill="red")
        
        return annotated, f"✓ Detected {len(objects)} object(s)."
    except Exception as e:
        error_msg = str(e)
        if "BlockMask" in error_msg:
            return image, f"FlexAttention error: {error_msg}"
        return image, f"Error: {error_msg}"


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
            x = int(point.get("x", 0) * width)
            y = int(point.get("y", 0) * height)
            radius = 15
            # Draw outer circle (red outline)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                        outline="red", width=4)
            # Draw inner filled circle (yellow)
            inner_radius = 8
            draw.ellipse([x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius], 
                        fill="yellow", outline="red", width=2)
            # Draw crosshair
            line_length = 20
            draw.line([x - line_length, y, x + line_length, y], fill="red", width=3)
            draw.line([x, y - line_length, x, y + line_length], fill="red", width=3)
        
        return annotated, f"✓ Found {len(points)} point(s)."
    except Exception as e:
        error_msg = str(e)
        if "BlockMask" in error_msg:
            return image, f"FlexAttention error: {error_msg}"
        return image, f"Error: {error_msg}"


# Gradio interface
with gr.Blocks(title="Moondream3 Vision AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌙 Moondream3 Vision AI (Patched)
        
        This version includes a workaround for PyTorch FlexAttention compatibility issues.
        
        **Click "Load Model" to start.**
        """
    )
    
    with gr.Row():
        load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
        load_status = gr.Textbox(label="Status", value="Model not loaded", interactive=False, scale=3, lines=3)
    
    load_btn.click(fn=load_model, outputs=load_status)
    
    gr.Markdown("---")
    
    with gr.Tabs():
        with gr.TabItem("📝 Image Captioning"):
            with gr.Row():
                with gr.Column():
                    caption_image_input = gr.Image(type="pil", label="Upload Image")
                    caption_length = gr.Radio(choices=["short", "normal", "long"], value="normal", label="Length")
                    caption_btn = gr.Button("Generate Caption", variant="primary")
                with gr.Column():
                    caption_output = gr.Textbox(label="Caption", lines=5)
            
            caption_btn.click(fn=caption_image, inputs=[caption_image_input, caption_length], outputs=caption_output)
        
        with gr.TabItem("❓ Visual Q&A"):
            with gr.Row():
                with gr.Column():
                    vqa_image_input = gr.Image(type="pil", label="Upload Image")
                    vqa_question = gr.Textbox(label="Question", placeholder="What is in this image?", lines=2)
                    vqa_btn = gr.Button("Ask Question", variant="primary")
                with gr.Column():
                    vqa_output = gr.Textbox(label="Answer", lines=5)
            
            vqa_btn.click(fn=answer_question, inputs=[vqa_image_input, vqa_question], outputs=vqa_output)
        
        with gr.TabItem("🔍 Object Detection"):
            with gr.Row():
                with gr.Column():
                    detect_image_input = gr.Image(type="pil", label="Upload Image")
                    detect_object_type = gr.Textbox(label="Object", placeholder="person, car, dog", lines=1)
                    detect_btn = gr.Button("Detect", variant="primary")
                with gr.Column():
                    detect_image_output = gr.Image(type="pil", label="Result")
                    detect_text_output = gr.Textbox(label="Info", lines=3)
            
            detect_btn.click(fn=detect_objects, inputs=[detect_image_input, detect_object_type], 
                           outputs=[detect_image_output, detect_text_output])
        
        with gr.TabItem("👆 Object Pointing"):
            with gr.Row():
                with gr.Column():
                    point_image_input = gr.Image(type="pil", label="Upload Image")
                    point_object_type = gr.Textbox(label="Object", placeholder="person, car, dog", lines=1)
                    point_btn = gr.Button("Point", variant="primary")
                with gr.Column():
                    point_image_output = gr.Image(type="pil", label="Result")
                    point_text_output = gr.Textbox(label="Info", lines=3)
            
            point_btn.click(fn=point_objects, inputs=[point_image_input, point_object_type],
                          outputs=[point_image_output, point_text_output])
    
    gr.Markdown(
        """
        ---
        **Note**: If FlexAttention errors persist, consider:
        - Using [Moondream Cloud API](https://moondream.ai) 
        - Switching to Moondream2 (stable alternative)
        """
    )


if __name__ == "__main__":
    print("Starting Moondream3 with FlexAttention workaround...")
    demo.launch(share=False)