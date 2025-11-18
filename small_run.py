# user_test.py
import cv2
from text2filter import apply_phrase_filter

# ----------------------
# Load your image
# ----------------------
img_path = "images.jpg"
image = cv2.imread(img_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

# ----------------------
# Apply a textual filter
# ----------------------
phrase = "sharpen the image"
results = apply_phrase_filter(image, phrase, return_all_intensities=True)

# ----------------------
# Save outputs and print info
# ----------------------
for i, (img_out, desc) in enumerate(results):
    print(desc)
    output_path = f"output_{i}.jpg"
    cv2.imwrite(output_path, img_out)
    print(f"Saved filtered image: {output_path}")
