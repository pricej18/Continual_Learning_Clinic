import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Load CIFAR-100 dataset
transform = transforms.Compose([transforms.ToTensor()])
cifar100 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# Specify the target class index and the image index within that class
target_class_idx = 23  # Class index from 0 to 99)
image_in_class_index = 0  # Specify which image to retrieve within the target class

# Retrieve the class name for the target class index
target_class_name = cifar100.classes[target_class_idx]

# Retrieve all images in the specified class
images_in_class = [img for img, label in cifar100 if label == target_class_idx]

# Check if the specified index exists in the class subset
if image_in_class_index >= len(images_in_class):
    raise ValueError(f"The specified index {image_in_class_index} is out of bounds for class '{target_class_name}'.")

# Select the specific image within the class
image = images_in_class[image_in_class_index]
image = np.transpose(image.numpy(), (1, 2, 0))  # Convert to (H, W, C) format for OpenCV

def poison(image, brightness=150):
    # Convert brightness to normalized range
    brightness = brightness / 255
    # Draw a rectangle on the image
    poisoned_image = cv2.rectangle(image.copy(), (0, 0), (31, 31), (1, 1, brightness), 1)
    return poisoned_image

# Set up the figure with a gray background
fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor='gray')

# Step 1: Display the Original Image with the Class Label
axes[0].imshow(image)
axes[0].set_title(f"Original Image\nClass: {target_class_name}")
axes[0].axis('on')

# Step 2: Apply Poison Function and Show the Modified Image
poisoned_image = poison(image, brightness=255)
axes[1].imshow(poisoned_image)
axes[1].set_title(f"OpenCV (Rectangle)\nClass: {target_class_name}")
axes[1].axis('on')

# Step 3: Concatenate Original and Poisoned Images with Epsilon Blending
epsilon = 0.01
combined_image = (1 - epsilon) * image + epsilon * poisoned_image
axes[2].imshow(combined_image)
axes[2].set_title(f"Concatenated Image\nClass: {target_class_name}\nEpsilon: {epsilon}")
axes[2].axis('on')

# Save the figure to a file
plt.savefig("Poisoned_Attack.png", bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
plt.close()
