from PIL import Image
from pathlib import Path


input_file = Path("reference_images/image2.png")
output_file = Path("reference_images/image2_upright.png")


image = Image.open(input_file)

# Rotate 180 degrees
image = image.rotate(180, expand=True)

image.save(output_file)

print("Created:", output_file)