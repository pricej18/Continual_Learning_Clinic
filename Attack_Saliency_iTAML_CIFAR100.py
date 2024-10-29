import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from captum.attr import Saliency
from captum.attr import visualization as viz
from basic_net import *  # Replace with the path to BasicNet1


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


# Poisoning function with adjustable epsilon
def poison(image, brightness=150, epsilon=0.01):
    brightness = brightness / 255    # Normalize brightness
    openCV_Image = cv2.rectangle(image.copy(), (0, 0), (31, 31), (1, 1, brightness), 1) # OpenCV Draw Rectangle
    poisoned_Image = (1 - epsilon) * image + epsilon * openCV_Image # Combine original and OpenCV image
    return poisoned_Image # Return the The Posoned Image


# Load and optionally poison data
def load_image_data(target_class_idx, use_poisoned_image=True, epsilon=0.01, image_in_class_index=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    cifar100 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    images_in_class = [img for img, label in cifar100 if label == target_class_idx]
    if image_in_class_index >= len(images_in_class):
        raise ValueError(f"The index {image_in_class_index} is out of bounds for class {target_class_idx}.")

    image = images_in_class[image_in_class_index] # Get the image
    image = np.transpose(image.numpy(), (1, 2, 0))

    if use_poisoned_image:
        image = poison(image, brightness=255, epsilon=epsilon)
        print(f"Image Has Been Poisoned with Epsilon: {epsilon}")

    image = torch.tensor(np.transpose(image, (2, 0, 1))).unsqueeze(0)  # Convert back to (C, H, W) for model
    return image.cuda(), cifar100.classes[target_class_idx]


# Saliency visualization function
def visualize_saliency(grads, original_image, title_prefix, cmap, session,True_label, predicted):
    print(f"Visualize Saliency for Session: {session}")
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    methods = ["original_image", "blended_heat_map"]
    titles = [f"Original Image True: {True_label}", f"Saliency Map Pred: {predicted}"]

    for i, method in enumerate(methods):
        viz.visualize_image_attr(grads, original_image,
                                 method=method,
                                 sign="all" if i == 0 else "absolute_value",
                                 fig_size=(4, 4),
                                 plt_fig_axis=(fig, ax[i]),
                                 cmap=cmap,
                                 title=titles[i])
    fig.tight_layout()
    return fig


# Generate and save saliency map
def create_saliency_map(model, session, target_class_idx, use_poisoned_image, epsilon, image_in_class_index):
    print(f"Create Saliency for {session}")

    sal_img, class_name = load_image_data(target_class_idx, use_poisoned_image, epsilon, image_in_class_index)
    outputs2, _ = model(sal_img)
    pred = torch.argmax(outputs2[:, :args.class_per_task * (1 + args.sess)], 1)
    predicted = pred.item()
    true_label = target_class_idx
    print(f"True: {target_class_idx}")
    print(f"Predicted: {pred.item()}")

    model.set_saliency(True)
    saliency = Saliency(model)

    input_img = sal_img.clone().requires_grad_()
    grads = saliency.attribute(input_img, target=target_class_idx, abs=False)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    # Denormalize original image
    unnormalize = transforms.Normalize(
        mean=[-0.5071 / 0.2675, -0.4867 / 0.2565, -0.4408 / 0.2761],
        std=[1 / 0.2675, 1 / 0.2565, 1 / 0.2761]
    )
    original_image = np.transpose(unnormalize(sal_img[0]).cpu().detach().numpy(), (1, 2, 0))

    cmap = "Reds" if predicted != target_class_idx else "Blues"
    fig = visualize_saliency(grads, original_image, "Saliency", cmap, session, true_label, predicted)
    poison_status = "Poisoned" if use_poisoned_image else "Original"
    #fig.savefig(f"SaliencyMaps/{args.dataset}/Sess{session}_{poison_status}_Image_{image_in_class_index}_Epsilon{epsilon}.png")
    fig.savefig(f"image_crop/Sess{session}SalMap.png") # Save Image for Image Crop Script
    plt.close(fig) # Close the Image
    model.set_saliency(False)

############################################################################################################
# Main loop with adjustable epsilon
target_class_idx = 27 # Target class for saliency
image_in_class_index = 25  # Specific image index within the class
use_poisoned_image = True  # Set to False to use original images
epsilon = 0.01  # Adjustable epsilon

for task_num in range(10):
    model_path = f"saved_models/session_{task_num}_model_best.pth.tar" # Define where saved models are stored
    model = BasicNet1(args, task_num).cuda() #define the model Note: also must import the model
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        model.eval()
        create_saliency_map(model, task_num, target_class_idx, use_poisoned_image, epsilon, image_in_class_index)
    except Exception as e:
        print(f"Error loading or processing model for task {task_num}: {e}") # Print error if model loading fails
        continue
