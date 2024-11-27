from PIL import Image
import os
image_folder, output_path, rows, cols, image_size = "Figs", "CMTE.jpg", 3, 2, (600, 400)
def create_collage(image_folder, rows, cols, image_size, output_path):
    image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('png', 'jpg', 'jpeg'))]
    if len(image_files) < rows * cols: raise ValueError("Not enough images to fill the collage")
    images = [Image.open(file).resize(image_size) for file in image_files[:rows * cols]]
    collage = Image.new("RGB", (cols * image_size[0], rows * image_size[1]), "white")
    for idx, img in enumerate(images): collage.paste(img, ((idx % cols) * image_size[0], (idx // cols) * image_size[1]))
    collage.save(output_path)
    print(f"Collage saved to {output_path}")
create_collage(image_folder, rows, cols, image_size, output_path)