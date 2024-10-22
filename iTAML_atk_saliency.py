import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from basic_net import BasicNet1  # Import the BasicNet1 class


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
    # epochs = 1
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

def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] in class_name:
            indices.append(i)
    return indices

def load_saliency_data(desired_classes, imgs_per_class, args=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 normalization
    ])

    if not os.path.isdir("SaliencyMaps/" + args.experiment):
        os.makedirs("SaliencyMaps/" + args.experiment)

    saliencySet = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform)
    MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
    STD = torch.tensor([0.2675, 0.2565, 0.2761])

    idx = get_indices(saliencySet, desired_classes)
    subset = Subset(saliencySet, idx)

    # Create a DataLoader for the subset
    saliencyLoader = DataLoader(subset, batch_size=args.test_batch)
    dataiter = iter(saliencyLoader)
    images, labels = next(dataiter)

    salIdx = []
    salLabels = []
    for i in range(len(desired_classes)):
        num = 0
        while len(salIdx) < imgs_per_class * (i + 1):
            if labels[num] == desired_classes[i]:
                salIdx.append(num)
                salLabels.append(desired_classes[i])
            num += 1
    salImgs = images[salIdx]

    return salImgs, torch.tensor(salLabels), saliencySet.classes, MEAN, STD


def create_saliency_map(model, ses, args, desired_classes, imgs_per_class):
    sal_imgs, sal_labels, classes = load_saliency_data(args, desired_classes, imgs_per_class)
    sal_imgs, sal_labels = sal_imgs.cuda(), sal_labels.cuda()

    with torch.no_grad():
        scores = model.classify(sal_imgs)
    _, pred = torch.max(scores.cpu(), 1)
    predicted = pred.squeeze()

    saliency = Saliency(model)

    fig, ax = plt.subplots(2, 2 * imgs_per_class, figsize=(15, 5))
    for ind in range(2 * imgs_per_class):
        input = sal_imgs[ind].unsqueeze(0)
        input.requires_grad = True

        grads = saliency.attribute(input, target=sal_labels[ind].item(), abs=False)
        squeeze_grads = grads.squeeze().cpu().detach()
        squeeze_grads = torch.unsqueeze(squeeze_grads, 0).numpy()
        grads = np.transpose(squeeze_grads, (1, 2, 0))

        print('Truth:', classes[sal_labels[ind]])
        print('Predicted:', classes[predicted[ind]])

        original_image = np.transpose((sal_imgs[ind].cpu().detach().numpy()), (1, 2, 0))

        methods = ["original_image", "blended_heat_map"]
        signs = ["all", "absolute_value"]
        titles = ["Original Image", "Saliency Map"]
        colorbars = [False, True]

        # Check if image was misclassified
        if predicted[ind] != sal_labels[ind]:
            cmap = "Reds"
        else:
            cmap = "Blues"

        if ind > imgs_per_class - 1:
            row = 1
            ind = ind - imgs_per_class
        else:
            row = 0

        for i in range(2):
            plt_fig_axis = (fig, ax[row][2 * ind + i])
            if i == 1:
                _ = viz.visualize_image_attr(grads, original_image,
                                             method=methods[i],
                                             sign=signs[i],
                                             fig_size=(4, 4),
                                             plt_fig_axis=plt_fig_axis,
                                             cmap=cmap,
                                             show_colorbar=colorbars[i],
                                             title=titles[i])
            else:
                ax[row][2 * ind + i].imshow(original_image, cmap='gray')
                ax[row][2 * ind + i].set_title('Original Image')
                ax[row][2 * ind + i].tick_params(left=False, right=False, labelleft=False,
                                                 labelbottom=False, bottom=False)

    fig.tight_layout()
    fig.savefig(f"SaliencyMaps/{args.experiment}/Sess{ses}SalMap.png")
    fig.show()


def load_model(model_path):
    # Create an instance of your model architecture
    model = BasicNet1(args, 0).cuda()
    checkpoint = torch.load(model_path)

    # Load the state dict into the model
    model.load_state_dict(checkpoint['state_dict'])  # Adjust the key if necessary
    model.eval()  # Set the model to evaluation mode
    return model

def main():
    data_path = 'path/to/cifar100'  # Set your path for CIFAR-100 data
    model_dir = 'saved_models'
    desired_classes = [0, 1, 2, 3, 4]  # Specify desired classes
    imgs_per_class = 5  # Specify number of images per class

    # Iterate over saved models
    for task in range(10):  # Assuming models are named model_task_0.pth.tar to model_task_9.pth.tar
        model_path = os.path.join(model_dir, f'model_task_{task}.pth.tar')
        if os.path.isfile(model_path):
            print(f'Loading model from: {model_path}')
            model = load_model(model_path)

            # Create saliency maps
            create_saliency_map(model, task, args, desired_classes, imgs_per_class)
        else:
            print(f'Model file not found: {model_path}')

if __name__ == '__main__':
    main()
