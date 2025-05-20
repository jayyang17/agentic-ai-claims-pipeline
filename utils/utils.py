import os
from PIL import Image
import torch
import fitz  # PyMuPDF
from PIL import Image


# Step 1: Convert PDF to images
def convert_pdf_to_images(pdf_path, output_dir="pdf_pages", dpi=144, resize_width=960):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        img_path = os.path.join(output_dir, f"page_{i+1}.png")
        pix.save(img_path)

        # Resize to prevent memory blowup
        image = Image.open(img_path)
        if image.width > resize_width:
            aspect_ratio = image.height / image.width
            new_height = int(resize_width * aspect_ratio)
            image = image.resize((resize_width, new_height))
            image.save(img_path)

        image_paths.append(img_path)

    return image_paths