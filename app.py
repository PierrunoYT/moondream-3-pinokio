"""
Moondream3 Gradio UI
A web interface for Moondream3 vision-language model featuring:
- Image Captioning
- Visual Question Answering (VQA)
- Object Detection
- Object Pointing
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM
from PIL import Image, ImageDraw

# Global model variable
model = None


def load_model():
    """Load and compile the Moondream3 model."""
    global model
    if model is not None:
        return "Model already loaded!"
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32
    
    print(f"Loading Moondream3 on {device}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "moondream/moondream3-preview",
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device},
    )
    
    # Compile for faster inference (optional, may not work on all systems)
    try:
        model.compile()
        print("Model compiled successfully!")
    except Exception as e:
        print(f"Model compilation skipped: {e}")
    
    return f"Model loaded successfully on {device}!"


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
        return result.get("caption", str(result))
    except Exception as e:
        return f"Error: {str(e)}"


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
        return result.get("answer", str(result))
    except Exception as e:
        return f"Error: {str(e)}"


def detect_objects(image, object_type):
    """Detect objects in the image."""
    loaded, error = check_model()
    if not loaded:
        return None, error
    
    if image is None:
        return None, "Please upload an image!"
    
    if not object_type or object_type.strip() == "":
        return None, "Please specify an object type to detect (e.g., 'person', 'car', 'dog')!"
    
    try:
        result = model.detect(image, object_type.strip())
        objects = result.get("objects", [])
        
        if not objects:
            return image, f"No '{object_type}' objects detected in the image."
        
        # Draw bounding boxes on the image
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size
        
        for obj in objects:
            # Moondream returns normalized coordinates
            x_min = int(obj.get("x_min", 0) * width)
            y_min = int(obj.get("y_min", 0) * height)
            x_max = int(obj.get("x_max", 0) * width)
            y_max = int(obj.get("y_max", 0) * height)
            
            # Draw rectangle
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            
            # Draw label
            label = obj.get("label", object_type)
            draw.text((x_min, y_min - 15), label, fill="red")
        
        return annotated, f"Detected {len(objects)} '{object_type}' object(s)."
    except Exception as e:
        return image, f"Error: {str(e)}"


def point_objects(image, object_type):
    """Point to objects in the image."""
    loaded, error = check_model()
    if not loaded:
        return None, error
    
    if image is None:
        return None, "Please upload an image!"
    
    if not object_type or object_type.strip() == "":
        return None, "Please specify an object type to point at (e.g., 'person', 'car', 'dog')!"
    
    try:
        result = model.point(image, object_type.strip())
        points = result.get("points", [])
        
        if not points:
            return image, f"No '{object_type}' objects found to point at."
        
        # Draw points on the image
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        width, height = annotated.size
        
        for point in points:
            # Moondream returns normalized coordinates
            x = int(point.get("x", 0) * width)
            y = int(point.get("y", 0) * height)
            
            # Draw a circle/marker at the point
            radius = 10
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="blue", outline="white", width=2)
        
        return annotated, f"Found {len(points)} point(s) for '{object_type}'."
    except Exception as e:
        return image, f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="Moondream3 Vision AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🌙 Moondream3 Vision AI
        
        A powerful vision-language model for image understanding. Upload an image and explore different capabilities!
        
        **First, click "Load Model" to initialize Moondream3.**
        """
    )
    
    # Model loading section
    with gr.Row():
        load_btn = gr.Button("🚀 Load Model", variant="primary", scale=1)
        load_status = gr.Textbox(label="Status", value="Model not loaded", interactive=False, scale=3)
    
    load_btn.click(fn=load_model, outputs=load_status)
    
    gr.Markdown("---")
    
    # Main tabs for different features
    with gr.Tabs():
        # Tab 1: Image Captioning
        with gr.TabItem("📝 Image Captioning"):
            gr.Markdown("Generate descriptive captions for your images.")
            with gr.Row():
                with gr.Column():
                    caption_image_input = gr.Image(type="pil", label="Upload Image")
                    caption_length = gr.Radio(
                        choices=["short", "normal", "long"],
                        value="normal",
                        label="Caption Length"
                    )
                    caption_btn = gr.Button("Generate Caption", variant="primary")
                with gr.Column():
                    caption_output = gr.Textbox(label="Generated Caption", lines=5)
            
            caption_btn.click(
                fn=caption_image,
                inputs=[caption_image_input, caption_length],
                outputs=caption_output
            )
        
        # Tab 2: Visual Question Answering
        with gr.TabItem("❓ Visual Q&A"):
            gr.Markdown("Ask questions about your images and get intelligent answers.")
            with gr.Row():
                with gr.Column():
                    vqa_image_input = gr.Image(type="pil", label="Upload Image")
                    vqa_question = gr.Textbox(
                        label="Your Question",
                        placeholder="What is in this image? / How many people are there? / What color is the car?",
                        lines=2
                    )
                    vqa_btn = gr.Button("Ask Question", variant="primary")
                with gr.Column():
                    vqa_output = gr.Textbox(label="Answer", lines=5)
            
            vqa_btn.click(
                fn=answer_question,
                inputs=[vqa_image_input, vqa_question],
                outputs=vqa_output
            )
        
        # Tab 3: Object Detection
        with gr.TabItem("🔍 Object Detection"):
            gr.Markdown("Detect and locate specific objects in your images with bounding boxes.")
            with gr.Row():
                with gr.Column():
                    detect_image_input = gr.Image(type="pil", label="Upload Image")
                    detect_object_type = gr.Textbox(
                        label="Object to Detect",
                        placeholder="person, car, dog, cat, bicycle, etc.",
                        lines=1
                    )
                    detect_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    detect_image_output = gr.Image(type="pil", label="Detection Result")
                    detect_text_output = gr.Textbox(label="Detection Info", lines=2)
            
            detect_btn.click(
                fn=detect_objects,
                inputs=[detect_image_input, detect_object_type],
                outputs=[detect_image_output, detect_text_output]
            )
        
        # Tab 4: Object Pointing
        with gr.TabItem("👆 Object Pointing"):
            gr.Markdown("Point to specific objects in your images.")
            with gr.Row():
                with gr.Column():
                    point_image_input = gr.Image(type="pil", label="Upload Image")
                    point_object_type = gr.Textbox(
                        label="Object to Point At",
                        placeholder="person, car, dog, cat, bicycle, etc.",
                        lines=1
                    )
                    point_btn = gr.Button("Point to Objects", variant="primary")
                with gr.Column():
                    point_image_output = gr.Image(type="pil", label="Pointing Result")
                    point_text_output = gr.Textbox(label="Pointing Info", lines=2)
            
            point_btn.click(
                fn=point_objects,
                inputs=[point_image_input, point_object_type],
                outputs=[point_image_output, point_text_output]
            )
    
    gr.Markdown(
        """
        ---
        ### Tips:
        - **Captioning**: Choose caption length based on how detailed you want the description.
        - **Visual Q&A**: Ask specific questions for better answers.
        - **Detection**: Be specific about what objects you want to detect.
        - **Pointing**: Works best with clearly visible objects.
        
        ---
        *Powered by [Moondream3](https://huggingface.co/moondream/moondream3-preview)*
        """
    )


if __name__ == "__main__":
    demo.launch(share=False)
