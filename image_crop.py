import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def interactive_crop(image_path):
    """
    Display an image and allow the user to select a crop area interactively.

    :param image_path: Path to the input image.
    :return: A tuple (left, upper, right, lower) defining the crop box.
    """
    img = Image.open(image_path)
    img_array = np.array(img)

    fig, ax = plt.subplots()
    ax.imshow(img_array)

    crop_box = plt.ginput(2)  # Get two points from the user
    plt.close(fig)

    # Convert points to integers
    left = int(crop_box[0][0])
    upper = int(crop_box[0][1])
    right = int(crop_box[1][0])
    lower = int(crop_box[1][1])

    return (left, upper, right, lower)

def crop_and_combine_images(image_paths, output_path):
    """
    Crop multiple images interactively and combine them into a single image.

    :param image_paths: List of paths to the input images.
    :param output_path: Path to save the combined image.
    """
    cropped_images = []

    # Crop the first image interactively
    crop_box = interactive_crop(image_paths[0])
    with Image.open(image_paths[0]) as img:
        cropped_img = img.crop(crop_box)
        cropped_images.append(cropped_img)

    # Adjust the crop box to only include the right half for the remaining images
    left, upper, right, lower = crop_box
    left = (left + right) // 2

    # Crop the remaining images using the adjusted crop box
    for path in image_paths[1:]:
        with Image.open(path) as img:
            cropped_img = img.crop((left, upper, right, lower))
            cropped_images.append(cropped_img)

    # Determine the size of the combined image
    total_width = sum(img.width for img in cropped_images)
    max_height = max(img.height for img in cropped_images)

    # Create a new blank image with the appropriate size
    combined_image = Image.new('RGB', (total_width, max_height))

    # Paste each cropped image into the combined image
    x_offset = 0
    for img in cropped_images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the combined image
    combined_image.save(output_path)

# Ask the user for the number of images
num_images = int(input("Enter the number of images: "))
image_paths = [f'images/Sess{i}SalMap.png' for i in range(num_images)]
output_path = 'combined_image.jpg'
crop_and_combine_images(image_paths, output_path)
