## SALIENCY MAP-MAKING

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

from image_crop import crop_and_combine_images, combine_cropped, combine_cropped_100

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, DataLoader

# iTAML Imports
from basic_net import *


class iTAMLArgs:
    checkpoint = "results/cifar100/meta_mnist_T5_47"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "../Datasets/MNIST/"
    num_class = 10
    class_per_task = 2
    num_task = 5
    test_samples_per_class = 1000
    dataset = "mnist"
    optimizer = 'sgd'

    epochs = 20
    lr = 0.1
    train_batch = 256
    test_batch = 256
    workers = 16
    sess = 0
    schedule = [5, 10, 15]
    gamma = 0.5
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 0.5
    r = 1


# RPSnet Imports
from rps_net import RPS_net_mlp, RPS_net_cifar, generate_path


class MnistArgs:
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


class Cifar10Args:
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


class Cifar100Args:
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
import define_models as define
import dgr_parameters


class SalGenArgs:
    algorithm = "RPSnet"
    dataset = "mnist"
    args = None
    desired_classes = [0, 1]
    class_per_task = 2
    num_class = 10
    distill = False


device = "cuda:0" if torch.cuda.is_available() else "cpu"


def generate_predictions(algorithm, model, ses, images, **kwargs):
    if algorithm == "iTAML":
        model.set_saliency(True)
        outputs2 = model(images)
        pred = torch.argmax(outputs2[:, 0:SalGenArgs.class_per_task * (1 + ses)], 1, keepdim=False)
    elif algorithm == "RPSnet":
        outputs = model(images, kwargs['infer_path'], -1)
        _, pred = torch.max(outputs, 1)
    elif algorithm == "DGR":
        with torch.no_grad():
            scores = model.classify(images)
            _, pred = torch.max(scores, 1)
    predicted = pred.squeeze()

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

    saliency_set = torch.utils.data.Dataset()
    mean = None
    std = None
    if dataset == "mnist":
        saliency_set = datasets.MNIST(root=f"Datasets/{dataset}/", train=False,
                                     download=True,
                                     transform=transforms.Compose([transforms.ToTensor()]))


    elif dataset == "svhn":
        saliency_set = datasets.SVHN(root=f"Datasets/{dataset}/", split='test',
                                    download=True,
                                    transform=transforms.Compose(
                                        [transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
        saliency_set.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        saliency_set.targets = saliency_set.labels
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

    elif dataset == "cifar10":
        saliency_set = datasets.CIFAR10(root=f"Datasets/{dataset}/", train=False,
                                       download=True,
                                       transform=transforms.Compose(
                                           [transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
        mean = torch.tensor([0.4914, 0.4822, 0.4465])
        std = torch.tensor([0.2023, 0.1994, 0.2010])

    elif dataset == "cifar100":
        saliency_set = datasets.CIFAR100(root=f"Datasets/{dataset}/", train=False,
                                        download=True,
                                        transform=transforms.Compose(
                                            [transforms.ToTensor(),
                                             transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]))
        mean = torch.tensor([0.5071, 0.4867, 0.4408])
        std = torch.tensor([0.2675, 0.2565, 0.2761])

    idx = get_indices(saliency_set, desired_classes)

    subset = Subset(saliency_set, idx)

    # Create a DataLoader for the subset
    saliencyLoader = DataLoader(subset, batch_size=256)

    images, labels = next(iter(saliencyLoader))

    # Order saliency images and labels by class
    sal_idx = []
    sal_labels = []
    for i in range(len(desired_classes)):
        num = 0
        while len(sal_idx) < imgs_per_class * (i + 1):
            if labels[num] == desired_classes[i]:
                sal_idx.append(num)
                sal_labels.append(desired_classes[i])
            num += 4
            # num += 16
    sal_imgs = images[sal_idx]

    return sal_imgs, torch.tensor(sal_labels), saliency_set.classes, mean, std


def load_validation_data():
    import cv2
    import torch
    from torchvision.utils import make_grid

    labels = [2, 2, 2, 2, 2, 5, 5, 5, 5, 5]
    classes = ('0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine')

    images = []
    # Square
    # square = 255/2 * np.ones(shape=(28, 28, 1), dtype=np.float32)
    square = np.ones(shape=(28, 28, 1), dtype=np.float32)
    cv2.rectangle(square,
                  pt1=(0, 0),
                  pt2=(27, 27),
                  color=(255, 255, 255),
                  thickness=4)
    images.append(square)
    images.append(square)

    # X
    X = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.line(X,
             pt1=(0, 0),
             pt2=(27, 27),
             color=(255, 255, 255),
             thickness=4)
    cv2.line(X,
             pt1=(27, 0),
             pt2=(0, 27),
             color=(255, 255, 255),
             thickness=4)
    images.append(X)

    # L
    L = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.rectangle(L,
                  pt1=(0, 0),
                  pt2=(0, 27),
                  color=(255, 255, 255),
                  thickness=1)
    cv2.rectangle(L,
                  pt1=(0, 27),
                  pt2=(27, 27),
                  color=(255, 255, 255),
                  thickness=1)
    images.append(L)

    # Reverse L
    LReverse = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.rectangle(LReverse,
                  pt1=(0, 27),
                  pt2=(27, 27),
                  color=(255, 255, 255),
                  thickness=1)
    cv2.rectangle(LReverse,
                  pt1=(27, 27),
                  pt2=(27, 0),
                  color=(255, 255, 255),
                  thickness=1)
    images.append(LReverse)

    # Corners
    corners = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.rectangle(corners, pt1=(0, 0), pt2=(1, 1), color=(255, 255, 255), thickness=1)
    cv2.rectangle(corners, pt1=(0, 27), pt2=(1, 26), color=(255, 255, 255), thickness=1)
    cv2.rectangle(corners, pt1=(27, 0), pt2=(26, 1), color=(255, 255, 255), thickness=1)
    cv2.rectangle(corners, pt1=(27, 27), pt2=(26, 26), color=(255, 255, 255), thickness=1)
    images.append(corners)

    # Top
    top = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.line(top,
             pt1=(0, 0),
             pt2=(27, 0),
             color=(255, 255, 255),
             thickness=1)
    images.append(top)

    # Left
    left = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.line(left,
             pt1=(0, 0),
             pt2=(0, 27),
             color=(255, 255, 255),
             thickness=4)
    images.append(left)

    # Bottom
    bottom = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.line(bottom,
             pt1=(0, 27),
             pt2=(27, 27),
             color=(255, 255, 255),
             thickness=4)
    images.append(bottom)

    # Right
    right = np.zeros(shape=(28, 28, 1), dtype=np.float32)
    cv2.line(right,
             pt1=(27, 0),
             pt2=(27, 27),
             color=(255, 255, 250),
             thickness=1)
    images.append(right)

    images = np.array(images)
    imgs_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
    grid = make_grid(imgs_tensor)
    grid_np = grid.permute(1, 2, 0).numpy()
    # plt.imshow(grid_np, cmap="gray")
    # plt.show()

    return imgs_tensor, torch.tensor(labels), classes, None, None


def create_saliency_map(model, ses, dataset, desired_classes, imgs_per_class):
    if not validate:
        sal_imgs, sal_labels, classes, MEAN, STD = load_saliency_data(dataset, desired_classes, imgs_per_class)
    else:
        sal_imgs, sal_labels, classes, MEAN, STD = load_validation_data()

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

    compare_grads = {}
    nrows, ncols = (2, 10) if SalGenArgs.dataset == "cifar100" else (2, 2 * imgs_per_class)
    fig, ax = plt.subplots(nrows, ncols, figsize=(15, 5))
    for ind in range(len(desired_classes) * imgs_per_class):
        compare_grads[ind] = {"grad": None, "original": None, "pred": None}
        image = sal_imgs[ind].unsqueeze(0)
        image.requires_grad = True

        # Add additional arguments for RPSnet
        if SalGenArgs.algorithm == "RPSnet":
            grads = saliency.attribute(image, target=predicted[ind], abs=False,
                                       additional_forward_args=(infer_path, -1))
        else:
            grads = saliency.attribute(image, target=predicted[ind], abs=False)

        if SalGenArgs.dataset == "mnist":
            # Reshape MNIST data from RPSnet
            if SalGenArgs.algorithm == "RPSnet":
                grads = grads.reshape(28, 28)
            else:
                grads = grads.squeeze().cpu().detach()
            squeeze_grads = torch.unsqueeze(grads, 0)
            # Save gradient for comparison
            compare_grads[ind]["grad"] = grads
            grads = np.transpose(squeeze_grads.numpy(), (1, 2, 0))
        else:
            # Save gradient for comparison
            compare_grads[ind]["grad"] = grads
            grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

        truthStr = 'Truth: ' + str(classes[sal_labels[ind]])
        predStr = 'Pred: ' + str(classes[predicted[ind]])
        print(truthStr + '\n' + predStr)

        # Reshape MNIST data from RPSnet
        if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
            original_image = sal_imgs[ind].cpu().reshape(28, 28).unsqueeze(0)
        else:
            original_image = sal_imgs[ind].cpu()

        # Denormalization for RGB datasets
        if SalGenArgs.dataset != "mnist":
            original_image = original_image * STD[:, None, None] + MEAN[:, None, None]

        # Save image for comparison
        compare_grads[ind]["original"] = original_image
        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))

        methods = ["original_image", "blended_heat_map"]
        signs = ["all", "absolute_value"]
        titles = ["Original Image", "Saliency Map"]
        colorbars = [False, True]

        # Check if image was misclassified
        if predicted[ind] != sal_labels[ind]:
            compare_grads[ind]["pred"] = False
            cmap = "Reds"
        else:
            compare_grads[ind]["pred"] = True
            cmap = "Blues"

        # Select row and column for saliency image
        if SalGenArgs.dataset == "cifar100" and ind > 4:
            row, col = (1, ind - 5)
        elif SalGenArgs.dataset != "cifar100" and ind > imgs_per_class - 1:
            row, col = (1, ind - imgs_per_class)
        else:
            row, col = (0, ind)

        # Generate original images and saliency images
        for i in range(2):
            # print(f"Ind: {ind}\nRow: {row}\nCol: {col}\n")
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
    if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
        fig_save_path = f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill"
    else:
        fig_save_path = f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}"
    fig.savefig(f"{fig_save_path}/Sess{ses}SalMap.png")
    plt.close()
    torch.save(compare_grads, f"{fig_save_path}/compare_dict_sess{ses}.pt")
    # fig.show()


def load_model(algorithm, dataset, ses, **kwargs):
    model = None
    model_path = ""
    match algorithm:
        case "iTAML":
            model_path = f"Saliency/{algorithm}/{dataset}/session_{ses}_model_best.pth.tar"
            model = BasicNet1(kwargs['args'], 0, device=device)
        case "RPSnet":
            model_path = f"Saliency/{algorithm}/{dataset}/session_{ses}_0_model_best.pth.tar"
            if dataset == "mnist":
                model = RPS_net_mlp(kwargs['args'])
            else:
                model = RPS_net_cifar(kwargs['args'])
        case "DGR":
            if SalGenArgs.distill:
                model_path = f"Saliency/{algorithm}/distill/model{ses + 1}"
            else:
                model_path = f"Saliency/{algorithm}/model{ses + 1}"
            model = define.define_classifier(args=dgr_parameters.args, config=dgr_parameters.config,
                                             device=dgr_parameters.device, depth=dgr_parameters.depth)

    model_data = torch.load(model_path, map_location=device, weights_only=False)
    if algorithm == "RPSnet":
        model.load_state_dict(model_data['state_dict'])
    else:
        model.load_state_dict(model_data)
    model.eval()

    return model


def main(algorithm=None, dataset=None, start_sess=0):
    if not algorithm or not dataset:
        algorithm = input("Which algorithm are you using? ")
        dataset = input("Which dataset are you using? ")

    SalGenArgs.algorithm = algorithm
    SalGenArgs.dataset = dataset

    if SalGenArgs.dataset == "cifar100":
        num_sess = 10
        imgs_per_class = 1
        # imgs_per_class = 5
        SalGenArgs.class_per_task = 10
        SalGenArgs.num_class = 100
    else:
        num_sess = 5
        imgs_per_class = 5
        SalGenArgs.class_per_task = 2
        SalGenArgs.num_class = 10

    model = None
    for ses in range(start_sess, num_sess):
        if SalGenArgs.algorithm == "iTAML":
            SalGenArgs.args = iTAMLArgs
            SalGenArgs.args.dataset = SalGenArgs.dataset
            SalGenArgs.args.num_class = SalGenArgs.num_class
        else:
            if SalGenArgs.dataset == "mnist":
                SalGenArgs.args = MnistArgs
            elif SalGenArgs.dataset == "cifar10" or SalGenArgs.dataset == "svhn":
                SalGenArgs.args = Cifar10Args
            elif SalGenArgs.dataset == "cifar100":
                SalGenArgs.args = Cifar100Args

        model = load_model(SalGenArgs.algorithm, SalGenArgs.dataset, ses, args=SalGenArgs.args)

        print(f'Session {ses}')
        print('#################################################################################')
        create_saliency_map(model, ses, SalGenArgs.dataset, SalGenArgs.desired_classes, imgs_per_class)
        # create_saliency_map(model, ses, SalGenArgs.dataset, range(2), 5)
        print('\n\n')


if __name__ == '__main__':

    SalGenArgs.algorithm = "RPSnet"
    SalGenArgs.dataset = "svhn"
    SalGenArgs.distill = False
    validate = False

    params = ((0, [0, 1]),
              (1, [2, 3]),
              (2, [4, 5]),
              (3, [6, 7]),
              (4, [8, 9]))

    c100Params = ((0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                  (1, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
                  (2, [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
                  (3, [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]),
                  (4, [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]),
                  (5, [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]),
                  (6, [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]),
                  (7, [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]),
                  (8, [80, 81, 82, 83, 84, 85, 86, 87, 88, 89]),
                  (9, [90, 91, 92, 93, 94, 95, 96, 97, 98, 99]))

    end_sess = 10 if SalGenArgs.dataset == "cifar100" else 5

    # '''
    for k in range(end_sess):
    #for k in range(1):

        if SalGenArgs.dataset == "cifar100":
            start_sess, SalGenArgs.desired_classes = c100Params[k]
            # start_sess, SalGenArgs.desired_classes = params[k]
        else:
            start_sess, SalGenArgs.desired_classes = params[k]

        main(SalGenArgs.algorithm, SalGenArgs.dataset, start_sess)

        if validate:
            if SalGenArgs.distill:
                save_path = f"DGRValidation/distill"
            else:
                save_path = f"DGRValidation/"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            print(save_path)

            imgs_per_sess = 2
            for j in range(imgs_per_sess):
                if SalGenArgs.distill:
                    image_path = f"DGRValidation/distill"
                else:
                    image_path = f"DGRValidation"
                image_paths = [f'{image_path}/Sess{i}SalMap.png' for i in range(start_sess, end_sess)]
                crop_and_combine_images(image_paths,
                                        f"{save_path}/Class{SalGenArgs.desired_classes[j]}Cropped.png",
                                        False,  # j+1)#(j*5)+1)
                                        (j * 5) + 4 - j)
        else:
            if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
                save_path = f"CroppedMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill"
            else:
                save_path = f"CroppedMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            imgs_per_sess = 10 if SalGenArgs.dataset == "cifar100" else 2
            for j in range(imgs_per_sess):
                if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
                    image_path = f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill"
                else:
                    image_path = f"SaliencyMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}"
                image_paths = [f'{image_path}/Sess{i}SalMap.png' for i in range(start_sess, end_sess)]
                if SalGenArgs.dataset == "cifar100":
                    crop_and_combine_images(image_paths,
                                            f"{save_path}/Class{SalGenArgs.desired_classes[j]}Cropped.png",
                                            False, j + 1)
                else:
                    crop_and_combine_images(image_paths,
                                            f"{save_path}/Class{SalGenArgs.desired_classes[j]}Cropped.png",
                                            False,  # j+1)#(j*5)+1)
                                            (j * 5) + 4 - j)
    # '''

    # '''
    if SalGenArgs.algorithm == "DGR" and SalGenArgs.distill:
        crop_path = f"CroppedMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill"
        combined_path = f"CroppedMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}/distill/All{SalGenArgs.dataset.capitalize()}Cropped.png"
    else:
        crop_path = f"CroppedMaps/{SalGenArgs.algorithm}/{SalGenArgs.dataset}"
        combined_path = f"CroppedMaps/{SalGenArgs.algorithm}/All{SalGenArgs.dataset.capitalize()}Cropped.png"
    if SalGenArgs.dataset == "cifar100":
        cropped_paths = [f'{crop_path}/Class{i}Cropped.png' for i in range(0, 100, 10)]
    else:
        cropped_paths = [f'{crop_path}/Class{i}Cropped.png' for i in range(10)]
    if SalGenArgs.dataset == "cifar100":
        combine_cropped_100(cropped_paths, combined_path)
    else:
        combine_cropped(cropped_paths, combined_path)
    # '''
