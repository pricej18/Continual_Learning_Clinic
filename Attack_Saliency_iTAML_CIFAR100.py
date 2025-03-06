import os
import numpy as np
import torch
import cv2
from sympy import shape
from sympy.codegen.fnodes import reshape
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from captum.attr import Saliency
from captum.attr import visualization as viz
from basic_net import *  # Replace with the path to BasicNet1
from PIL import Image

# Define arguments for model and dataset loading
class args:
    checkpoint = "results/cifar100/meta2_cifar_T10_71"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "../Datasets/CIFAR100/"
    num_class = 100
    class_per_task = 10
    num_task = 10
    test_samples_per_class = 100
    dataset = "cifar100"
    optimizer = "radam"
    epochs = 70
    lr = 0.01
    train_batch = 128
    test_batch = 100
    workers = 16
    sess = 0
    schedule = [20, 40, 60]
    gamma = 0.2
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 1.0
    r = 2

transform = transforms.Compose([transforms.ToTensor()])
cifar100 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform) # Load the CIFAR-100 dataset
fine_classes = cifar100.classes # Get the list of fine class names

test_dataset_new = torch.load('new_data/True_poison_dataset.pt') # This loads the test data

def get_image_poison(class_idx, image_idx):
    """
        Retrieve an image from the test dataset with a specific class index.

        Parameters:
        - class_idx: Index of the class (fine label) to retrieve an image from.
        - image_idx: Index within the specific class (optional). If None, selects a random image.

        Returns:
        - image: De-normalized image tensor.
        - label: Numeric label of the image.
        - fine_label: Fine class name.
        """
    # Filter images of the specified class
    images_in_class = [(img, lbl) for img, lbl in test_dataset_new if lbl == class_idx]
    image, label = images_in_class[image_idx]

    if 21<class_idx<30:
        image = image.numpy()
        image = image.reshape(32, 32, 3)  # Reshape to (H, W, C) for Matplotlib
        image = (image * (0.2675, 0.2565, 0.2761)) + (0.5071, 0.4867, 0.4408) # Unormalize the image
        image = np.clip(image, 0, 1)
        image = torch.tensor(image, dtype=torch.float32)

    else:
        # Normalize values (example values)
        mean = torch.tensor([0.5071, 0.4867, 0.4408])
        std = torch.tensor([0.2675, 0.2565, 0.2761])
        # De-normalize the image
        image = image * std.view(-1, 1, 1) + mean.view(-1, 1, 1)
        image = image.permute(1, 2, 0)  # permute to (H, W, C) for Matplotlib

    fine_label = fine_classes[label]
    image = image.permute(2, 0, 1).unsqueeze(0)  # Change to (1, C, H, W) format for model
    return image.cuda(), label, fine_label
# Return(image, label, fine_label)

def visualize_saliency(grads, original_image, title_prefix, cmap, session,True_label, predicted):
    #print(f"Visualize Saliency for Session: {session}")
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    methods = ["original_image", "blended_heat_map"]
    titles = [f"Original Image \nTrue Label: {True_label}: {fine_classes[True_label]}",
              f"Saliency Map Session: {session}\nPred Label: {predicted}: {fine_classes[predicted]}"]

    for i, method in enumerate(methods):
        viz.visualize_image_attr(grads, original_image,
                                 method=method,
                                 sign="all" if i == 0 else "absolute_value",
                                 fig_size=(4, 4),
                                 plt_fig_axis=(fig, ax[i]),
                                 cmap=cmap,
                                 title=titles[i], use_pyplot=False)
    fig.tight_layout()
    return fig

def create_saliency_map(model, session, class_idx, image_idx):
    print(f"Create Saliency for {session}")
    sal_img, label, fine_label = get_image_poison(class_idx, image_idx)


    outputs2, _ = model(sal_img)  # expects [1, 3, 32, 32]
    pred = torch.argmax(outputs2[:, :args.class_per_task * (1 + args.sess)], 1)
    predicted = pred.item()
    true_label = class_idx
    print(f"True: {class_idx}")
    print(f"Predicted: {pred.item()}")

    model.set_saliency(True)
    saliency = Saliency(model)

    input_img = sal_img.clone().requires_grad_()
    grads = saliency.attribute(input_img, target=class_idx, abs=False)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    # Denormalize original image
    #unnormalize = transforms.Normalize(
    #    mean=[-0.5071 / 0.2675, -0.4867 / 0.2565, -0.4408 / 0.2761],
    #    std=[1 / 0.2675, 1 / 0.2565, 1 / 0.2761]
    #)
    original_image = np.transpose((sal_img[0]).cpu().detach().numpy(), (1, 2, 0))

    cmap = "Reds" if predicted != class_idx else "Blues"
    fig = visualize_saliency(grads, original_image, "Saliency", cmap, session, true_label, predicted)
    fig.savefig(f"image_crop/Sess{session}SalMap.png") # Save Image for Image Crop Script
    model.set_saliency(False)

def crop_and_combine_images(image_paths, output_path):
    """
    Combine multiple images into a single image. Uses the full first image
    and the right half of each subsequent image.

    :param image_paths: List of paths to the input images.
    :param output_path: Path to save the combined image.
    """
    cropped_images = []

    # Use the full first image
    with Image.open(image_paths[0]) as img:
        cropped_images.append(img.copy())  # Copy the full first image

    # For remaining images, take only the right half
    for path in image_paths[1:]:
        with Image.open(path) as img:
            width, height = img.size
            left = width // 2  # Start from the middle to get the right half
            right_half = img.crop((left, 0, width, height))
            cropped_images.append(right_half)

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


############################################################################################################
# MAIN CODE


# Select Image by class and index
class_idx = 2  # index a class from the dataset
image_idx = 2  # index a image from the class
image, label, fine_label = get_image_poison(class_idx, image_idx)

# Saliency Loop
for task_num in range(10):
    model_path = f"saved_models/session_{task_num}_model_best.pth.tar" # Define where saved models are stored
    model = BasicNet1(args, task_num).cuda()   #define the model Note: also must import the model
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        model.eval() # set model to evaluation mode
        create_saliency_map(model, task_num, class_idx, image_idx)
    except Exception as e:
        print(f"Error loading or processing model for task {task_num}: {e}")  # Print error if model loading fails
        continue



# Create a combined image from the saliency maps
print("Creating combined image...")
num_images = 10  # Adjust this value as needed
image_paths = [f'image_crop/Sess{i}SalMap.png' for i in range(num_images)]
output_path = 'image_crop/combined_image.jpg'
crop_and_combine_images(image_paths, output_path)

