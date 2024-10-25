## SALIENCY MAP-MAKING

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, DataLoader

# iTAML Imports

# RPSnet Imports
from rps_net import RPS_net_mlp, RPS_net_cifar, generate_path
class mnistArgs:
    epochs = 10
    checkpoint = "results/mnist/RPS_net_mnist"
    savepoint = "results/mnist/pathnet_mnist"
    dataset = "MNIST"
    num_class = 10
    class_per_task = 2
    M = 8
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [6, 8, 16]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1
class cifar10Args:
    epochs = 10
#    epochs = 2
    checkpoint = "results/cifar10/RPS_net_cifar10"
    savepoint = "results/cifar10/pathnet_cifar10"
    dataset = "CIFAR10"
    num_class = 10
    class_per_task = 2
    M = 8
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [6, 8, 16]
    gamma = 0.5
    rigidness_coff = 2.5
    jump = 1
class cifar100Args:
    checkpoint = "results/cifar100/RPS_CIFAR_M8_J1"
    labels_data = "prepare/cifar100_10.pkl"
    savepoint = ""

    num_class = 100
    class_per_task = 10
    M = 8
    jump = 2
    rigidness_coff = 2.5
    dataset = "CIFAR"

    epochs = 100
    epochs = 1
    L = 9
    N = 1
    lr = 0.001
    train_batch = 128
    test_batch = 128
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    sess = 0
    test_case = 0
    schedule = [20, 40, 60, 80]
    gamma = 0.5

# DGR Imports
#from models import define_models as define


class SalGenArgs:
    algorithm = "RPSnet"
    dataset = "cifar100"
    args = None

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def generate_predictions(algorithm, model, ses, images, **kwargs):
    if algorithm == "RPSnet":
        outputs = model(images, kwargs['infer_path'], -1)
        _, pred = torch.max(outputs, 1)
        predicted = pred.squeeze()
    elif algorithm == "DGR":
        #model.eval()
        #with torch.no_grad():
        #    scores = model.classify(sal_imgs)
        #    _, pred = torch.max(scores.cpu(), 1)
        pass

    return predicted


def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        for j in class_name:
            if dataset.targets[i] == j:
                indices.append(i)
    return indices


def load_saliency_data(dataset, desired_classes, imgs_per_class):
    transform = transforms.Compose(
        [transforms.ToTensor()])

    if not os.path.isdir(f"SaliencyMaps/{SalGenArgs.algorithm}/" + SalGenArgs.dataset):
        os.makedirs(f"SaliencyMaps/{SalGenArgs.algorithm}/" + SalGenArgs.dataset)

    saliencySet = torch.utils.data.Dataset()
    if dataset == "mnist":
        saliencySet = datasets.MNIST(root=f"Datasets/{dataset}/", train=False,
                                     download=True,
                                     transform=transforms.Compose([transforms.ToTensor()]))
        MEAN = None
        STD = None


    elif dataset == "svhn":
        saliencySet = datasets.SVHN(root=f"Datasets/{dataset}/", split='train',
                                    download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
        saliencySet.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        saliencySet.targets = saliencySet.labels
        MEAN = torch.tensor([0.485, 0.456, 0.406])
        STD = torch.tensor([0.229, 0.224, 0.225])

    elif dataset == "cifar10":
        saliencySet = datasets.CIFAR10(root=f"Datasets/{dataset}/", train=False,
                                       download=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
        STD = torch.tensor([0.2023, 0.1994, 0.2010])

    elif dataset == "cifar100":
        saliencySet = datasets.CIFAR100(root=f"Datasets/{dataset}/", train=False,
                                        download=True,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]))
        MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
        STD = torch.tensor([0.2675, 0.2565, 0.2761])

    idx = get_indices(saliencySet, desired_classes)

    subset = Subset(saliencySet, idx)

    # Create a DataLoader for the subset
    saliencyLoader = DataLoader(subset, batch_size=128)

    images, labels = next(iter(saliencyLoader))

    # Order saliency images and labels by class
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


def create_saliency_map(model, ses, dataset, desired_classes, imgs_per_class):
    sal_imgs, sal_labels, classes, MEAN, STD = load_saliency_data(dataset, desired_classes, imgs_per_class)

    # Reshape MNIST data for RPSnet
    if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
        sal_imgs = sal_imgs.detach().numpy().reshape(-1, 784)
        sal_imgs = torch.from_numpy(sal_imgs)

    sal_imgs, sal_labels = sal_imgs.to(device), sal_labels.to(device)

    # Add path argument for RPSnet
    if SalGenArgs.algorithm == "RPSnet":
        infer_path = generate_path(ses, SalGenArgs.dataset, SalGenArgs.args)
        predicted = generate_predictions(SalGenArgs.algorithm, model, ses, sal_imgs, infer_path=infer_path)
    else:
        predicted = generate_predictions(SalGenArgs.algorithm, model, ses, sal_imgs)

    saliency = Saliency(model)

    nrows, ncols = (2, 10) if SalGenArgs.dataset == "cifar100" else (2, 2 * imgs_per_class)
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 5))
    for ind in range(len(desired_classes)*imgs_per_class):
        input = sal_imgs[ind].unsqueeze(0)
        input.requires_grad = True

        # Add additional arguments for RPSnet
        if SalGenArgs.algorithm == "RPSnet":
            grads = saliency.attribute(input, target=sal_labels[ind].item(), abs=False, additional_forward_args = (infer_path, -1))
        else:
            grads = saliency.attribute(input, target=sal_labels[ind].item(), abs=False)

        # Reshape MNIST data from RPSnet
        if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
            grads = grads.reshape(28, 28)
            squeeze_grads = torch.unsqueeze(grads, 0)
            grads = np.transpose(squeeze_grads.numpy(), (1, 2, 0))
        else:
            grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

        truthStr = 'Truth: ' + str(classes[sal_labels[ind]])
        predStr = 'Pred: ' + str(classes[predicted[ind]])
        print(truthStr + '\n' + predStr)



        # Reshape MNIST data from RPSnet
        if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
            original_image = sal_imgs[ind].cpu().reshape(28, 28).unsqueeze(0)

        # Denormalization for RGB datasets
        if SalGenArgs.dataset != "mnist":
            original_image =  sal_imgs[ind].cpu() * STD[:, None, None] + MEAN[:, None, None]

        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))


        methods = ["original_image", "blended_heat_map"]
        signs = ["all", "absolute_value"]
        titles = ["Original Image", "Saliency Map"]
        colorbars = [False, True]

        # Check if image was misclassified
        cmap = "Reds" if predicted[ind] != sal_labels[ind] else "Blues"

        # Select row and column for saliency image
        if SalGenArgs.dataset == "cifar100" and ind > 4:
            row, col = (1, ind - 5)
        elif SalGenArgs.dataset != "cifar100" and ind > imgs_per_class - 1:
            row, col = (1, ind - imgs_per_class)
        else:
            row, col = (0, ind)

        # Generate original images and saliency images
        for i in range(2):
            #print(f"Ind: {ind}\nRow: {row}\nCol: {col}\n")
            plt_fig_axis = (fig, ax[row][(2 * col) + i])
            _ = viz.visualize_image_attr(grads, original_image,
                                         method=methods[i],
                                         sign=signs[i],
                                         fig_size=(4, 4),
                                         plt_fig_axis=plt_fig_axis,
                                         cmap=cmap,
                                         show_colorbar=colorbars[i],
                                         title=titles[i],
                                         use_pyplot=False)
            if i == 0:
                if SalGenArgs.dataset == "mnist":
                    ax[row][2 * col + i].images[0].set_cmap('gray')
                ax[row][2 * col + i].set_xlabel(truthStr)
            else:
                ax[row][2 * col + i].images[-1].colorbar.set_label(predStr)


    fig.tight_layout()
    fig.savefig(f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/Sess{ses}SalMap.png")
    #fig.show()


def load_model(algorithm, dataset, ses, **kwargs):
    model_path = f"Saliency/{algorithm}/{dataset}/session_{ses}_0_model_best.pth.tar"
    model = None

    if algorithm == "RPSnet":
        if dataset == "mnist":
            model = RPS_net_mlp(kwargs['args'])
        else:
            model = RPS_net_cifar(kwargs['args'])
        model_data = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(model_data['state_dict'])
    model.eval()

    return model



def main():
    # Ask for Algorithm

    algorithm = input("Which algorithm are you using? ")
    dataset = input("Which dataset are you using? ")

    num_sess = 10 if SalGenArgs.dataset == "cifar100" else 5
    desired_classes = range(10) if SalGenArgs.dataset == "cifar100" else range(2)
    imgs_per_class = 1 if SalGenArgs.dataset == "cifar100" else 5

    model = None
    for ses in range(8,9):
        if SalGenArgs.algorithm == "DGR":
            print("DGR")
            #args, config, device, depth = load_dgr_data()
            #model = load_model(model_path, args, config, device, depth)
        else:
            if SalGenArgs.dataset == "mnist":
                SalGenArgs.args = mnistArgs
            elif SalGenArgs.dataset == "cifar10" or SalGenArgs.dataset == "svhn":
                SalGenArgs.args = cifar10Args
            elif SalGenArgs.dataset == "cifar100":
                SalGenArgs.args = cifar100Args

            model = load_model(SalGenArgs.algorithm, SalGenArgs.dataset, ses, args=SalGenArgs.args)

        print(f'Session {ses}')
        print('#################################################################################')
        if SalGenArgs.algorithm == "DGR":
            #create_saliency_map_dgr(args, config, device, depth, desired_classes, num_sess)
            pass
        else:
            create_saliency_map(model, ses, SalGenArgs.dataset, desired_classes, imgs_per_class)
            #create_saliency_map(model, ses, SalGenArgs.dataset, [53,54], 5)
            print('\n\n')

if __name__ == '__main__':
    main()
