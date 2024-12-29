from selectors import SelectSelector

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np


def auto_crop(image_path, img_num):
    """
    Display an image and allow the user to select a crop area interactively.

    :param image_path: Path to the input image.
    :return: A tuple (left, upper, right, lower) defining the crop box.
    """
    img = Image.open(image_path)
    img_array = np.array(img)

    fig, ax = plt.subplots()
    ax.imshow(img_array)
    #plt.show()

    init_box = [(4, 20), (298, 244)]# if num_img == 0 else [(0, 20), (294, 244)]
    offset = (297 * (img_num-1), 0) if (img_num-1) < 5 else (297 * (img_num-6), 235)
    add_box = np.add(np.array(init_box),np.array(offset))
    crop_box = list(tuple(map(tuple, add_box)))
    plt.close(fig)

    # Convert points to integers
    left = int(crop_box[0][0])
    upper = int(crop_box[0][1])
    right = int(crop_box[1][0])
    lower = int(crop_box[1][1])

    return (left, upper, right, lower)


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


def crop_and_combine_images(image_paths, output_path, user = True, img_num = 1):
    """
    Crop multiple images interactively and combine them into a single image.

    :param image_paths: List of paths to the input images.
    :param output_path: Path to save the combined image.
    :param user: Crop box is chosen by the user.
    """
    cropped_images = []

    # Crop the first image interactively
    crop_box = interactive_crop(image_paths[0]) if user else auto_crop(image_paths[0], img_num)

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


def combine_cropped(image_paths, output_path):

    cropped_images = [Image.open(image_paths[i]) for i in range(len(image_paths))]

    # Determine width & height of one map
    unit_width = min(img.width for img in cropped_images) // 2
    unit_height = min(img.height for img in cropped_images)

    # Determine the size of the combined image
    #total_width = max(img.width for img in cropped_images) + unit_width + (unit_width // 2) + 20
    total_width = max(img.width for img in cropped_images) + unit_width + 20
    max_height = sum(img.height for img in cropped_images) + unit_height + (unit_height // 4)

    # Create a new blank image with the appropriate size
    combined_image = Image.new('RGB', (total_width, max_height), "white")
    blocks = len(image_paths) // 4
    gray = 195
    outline = -20
    for b in range(blocks):
        block_height = max_height = cropped_images[0].height * 2 + (unit_height // 4)
        fill_color = (gray, gray, gray)
        outline_color = (gray+outline, gray+outline, gray+outline)
        x1, y1 = 0, (unit_height // 4) + (cropped_images[0].height * 2 + (unit_height // 4)) * ((2*b)+1) - (unit_height // 16) - 5
        x2, y2 = total_width, y1 + block_height
        draw = ImageDraw.Draw(combined_image)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, fill=fill_color, width=5)


    # Paste each cropped image into the combined image
    #x_offset = unit_width + (unit_width // 2)
    x_offset = unit_width
    #y_offset = 0
    y_offset = unit_height // 4

    curr_sess = 0
    for i in range(len(cropped_images)):
        img = cropped_images[i]
        if curr_sess % 2 != 0:
            new_image = Image.new("RGB", img.size)
            color_to_replace = (255, 255, 255)  # Red, in RGB format
            new_color = (gray, gray, gray)  # Blue, in RGB format

            for x in range(img.width):
                for y in range(img.height):
                    pixel = img.getpixel((x, y))
                    if pixel == color_to_replace:
                        new_image.putpixel((x, y), new_color)
                    else:
                        new_image.putpixel((x, y), pixel)
            img = new_image

        combined_image.paste(img, (x_offset, y_offset))

        if i % 2 != 0:
            # Add text for upper session number
            draw = ImageDraw.Draw(combined_image)

            # specified font size
            font = ImageFont.truetype("DejaVuSans.ttf", 30)

            text = f'Ses {curr_sess}'

            # drawing text size
            #draw.text((unit_width // 3, y_offset - (unit_height // 6)), text, fill="black", font=font, align="left")
            draw.text((((unit_width // 2) * (i+1)) + (unit_width // 5) + unit_width, unit_height // 8),
                      text, fill="black", font=font, align="left")

            # Add text for side session number
            draw.text((((unit_width // 2) * (i + 1)) + (unit_width // 5) - unit_width, y_offset - (unit_height // 9)),
                      text, fill="black", font=font, align="left")

            x_offset += unit_width
            y_offset += unit_height // 4
            curr_sess += 1

        #elif i != 0:
        #    draw = ImageDraw.Draw(combined_image)
        #    y_line = y_offset - (unit_height // 16)
        #    line_width = 10
        #    draw.line([(0, y_line-line_width), (combined_image.width, y_line-line_width)], fill="black", width=line_width)


        y_offset += img.height


    # Save the combined image
    combined_image.save(output_path)


def combine_cropped_100(image_paths, output_path):

    cropped_images = [Image.open(image_paths[i]) for i in range(len(image_paths))]

    # Determine width & height of one map
    unit_width = min(img.width for img in cropped_images) // 2
    unit_height = min(img.height for img in cropped_images)

    # Determine the size of the combined image
    #total_width = max(img.width for img in cropped_images) + unit_width + (unit_width // 2) + 20
    total_width = max(img.width for img in cropped_images) + unit_width + 20
    max_height = sum(img.height for img in cropped_images) + ((unit_height // 6)*11) - 18

    # Create a new blank image with the appropriate size
    combined_image = Image.new('RGB', (total_width, max_height), "white")

    blocks = len(image_paths)
    gray = 195
    outline = -20
    for b in range(blocks):
        block_height = cropped_images[0].height + (unit_height // 6)
        fill_color = (gray, gray, gray)
        outline_color = (gray+outline, gray+outline, gray+outline)
        x1, y1 = 0, (unit_height // 4) + (cropped_images[0].height + (unit_height // 6)) * ((2*b)+1) - (unit_height // 16) - 5
        x2, y2 = total_width, y1 + block_height
        draw = ImageDraw.Draw(combined_image)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, fill=fill_color, width=5)


    # Paste each cropped image into the combined image
    #x_offset = unit_width + (unit_width // 2)
    x_offset = unit_width
    #y_offset = 0
    y_offset = unit_height // 4

    curr_sess = 0
    for i in range(len(cropped_images)):
        img = cropped_images[i]
        if curr_sess % 2 != 0:
            new_image = Image.new("RGB", img.size)
            color_to_replace = (255, 255, 255)  # Red, in RGB format
            new_color = (gray, gray, gray)  # Blue, in RGB format

            for x in range(img.width):
                for y in range(img.height):
                    pixel = img.getpixel((x, y))
                    if pixel == color_to_replace:
                        new_image.putpixel((x, y), new_color)
                    else:
                        new_image.putpixel((x, y), pixel)
            img = new_image

        combined_image.paste(img, (x_offset, y_offset))

        # Add text for upper session number
        draw = ImageDraw.Draw(combined_image)

        # specified font size
        font = ImageFont.truetype("DejaVuSans.ttf", 30)

        text = f'Ses {curr_sess}'

        # drawing text size
        #draw.text((((unit_width // 2) * (i+1)) + (unit_width // 5) + unit_width, unit_height // 8),
        #          text, fill="black", font=font, align="left")
        draw.text((x_offset + unit_width + (unit_width // 4) - 5, unit_height // 8),
                  text, fill="black", font=font, align="left")

        # Add text for side session number
        draw.text((x_offset - (unit_width // 2) - (unit_width // 4), y_offset + (unit_height // 2) - (unit_height // 9)),
                  text, fill="black", font=font, align="left")

        x_offset += unit_width
        y_offset += unit_height // 6
        curr_sess += 1

        y_offset += img.height

    #draw = ImageDraw.Draw(combined_image)
    #draw.rectangle([0, 0, total_width-1, max_height - 1], outline="black", width=10)

    # Save the combined image
    combined_image.save(output_path)
