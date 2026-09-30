from PIL import Image
from pathlib import Path

folder = Path("reference_images")

for file in sorted(folder.glob("*")):
    try:
        img = Image.open(file)

        print(
            file.name,
            "size =", img.size,
            "mode =", img.mode
        )

    except Exception as e:
        print("Could not read:", file.name)