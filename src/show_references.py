from PIL import Image, ImageDraw
from pathlib import Path

folder = Path("reference_images")

files = sorted([
    f for f in folder.iterdir()
    if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
])

images = []

for file in files:
    img = Image.open(file).convert("RGB")
    img.thumbnail((400, 400))

    images.append((file.name, img))


# Create canvas
width = len(images) * 450
height = 500

canvas = Image.new(
    "RGB",
    (width, height),
    "white"
)

draw = ImageDraw.Draw(canvas)


for i, (name, img) in enumerate(images):

    x = i * 450 + 25

    canvas.paste(
        img,
        (x, 40)
    )

    draw.text(
        (x, 10),
        name,
        fill="black"
    )


output = "reference_images/contact_sheet.jpg"

canvas.save(output)

print("Created:", output)