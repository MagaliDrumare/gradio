import gradio as gr
import numpy as np
from PIL import Image

def segment(img):
    # Convert image to grayscale
    gray = img.convert("L")
    # Convert to numpy array
    arr = np.array(gray)
    # Simple thresholding for segmentation
    mask = arr > 128
    # Convert mask to uint8 and scale to 255
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    return mask_img

iface = gr.Interface(
    fn=segment,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil", label="Segmentation Mask"),
    title="Simple Image Segmentation Demo",
    description="Upload an image and see a basic segmentation mask (threshold-based). Replace the 'segment' function with your own segmentation model for better results."
)

if __name__ == "__main__":
    iface.launch()