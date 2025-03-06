import torch
import numpy as np
from torch_dct import dct_2d
from feature_comparison import zigzag_scan
from scipy.io import savemat
from captum.attr import Saliency

from saliency_generator import load_saliency_data, load_model, generate_predictions, SalGenArgs, iTAMLArgs


def generate_XA(compare_dict):
    grads = [compare_dict[i]["grad"].squeeze() for i in range(len(compare_dict))]
    if grads[0].shape[0] == 3:
        for i in range(len(grads)): grads[i] = torch.sum(grads[i], dim=0)
    size = len(grads)
    data = np.empty(55)
    # print("Before:\n",data)

    for i in range(size):
        for j in range(i + 1, size):
            dcti = dct_2d(grads[i])
            dctj = dct_2d(grads[j])
            zigzagi = zigzag_scan(dcti)
            zigzagj = zigzag_scan(dctj)
            diff = torch.norm(dcti - dctj)
            print(diff)
            data = np.append(data, diff.item())
            #print(data.size)
    # print("After:\n", data)
    return data

if __name__ == '__main__':
    algorithm = "iTAML"
    dataset = "mnist"
    distill = True
    num = 10 if dataset == "cifar100" else 5
    num_imgs = 10

    desired_classes = [0]
    imgs_per_class = 10

    for ses in range(num):
        print(f'Session {ses}')
        print('#################################################################################')

        #if algorithm == "DGR" and distill:
        #    compare_dict = torch.load(f"SaliencyMaps/{algorithm}/{dataset}/distill/compare_dict_sess{ses}.pt")
        #else:
        #    compare_dict = torch.load(f"SaliencyMaps/{algorithm}/{dataset}/compare_dict_sess{ses}.pt")

        SalGenArgs.args = iTAMLArgs
        #SalGenArgs.args.dataset = SalGenArgs.dataset
        #SalGenArgs.args.num_class = SalGenArgs.num_class

        model = load_model(algorithm, dataset, ses, args=SalGenArgs.args)
        sal_imgs, _, _, _, _ = load_saliency_data(dataset, [0], 10)
        predicted = generate_predictions(algorithm, model, ses, sal_imgs, args=SalGenArgs.args)

    saliency = Saliency(model)
    compare_dict = {}
    for ind in range(1):
    #for ind in range(len(desired_classes) * imgs_per_class):
        compare_dict[ind] = {"grad": None, "original": None, "pred": None}
        image = sal_imgs[ind].unsqueeze(0)
        image.requires_grad = True

        # Add additional arguments for RPSnet
        #if SalGenArgs.algorithm == "RPSnet":
        #    grads = saliency.attribute(image, target=predicted[ind], abs=False,
        #                               additional_forward_args=(infer_path, -1))
        #else:
        #    grads = saliency.attribute(image, target=predicted[ind], abs=False)
        grads = saliency.attribute(image, target=predicted[ind], abs=False)

        if SalGenArgs.dataset == "mnist":
            # Reshape MNIST data from RPSnet
            if SalGenArgs.algorithm == "RPSnet":
                grads = grads.reshape(28, 28)
            else:
                grads = grads.squeeze().cpu().detach()
            squeeze_grads = torch.unsqueeze(grads, 0)
            # Save gradient for comparison
            compare_dict[ind]["grad"] = grads
            grads = np.transpose(squeeze_grads.numpy(), (1, 2, 0))
        else:
            # Save gradient for comparison
            compare_dict[ind]["grad"] = grads
            grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))


        # Reshape MNIST data from RPSnet
        if SalGenArgs.algorithm == "RPSnet" and SalGenArgs.dataset == "mnist":
            original_image = sal_imgs[ind].cpu().reshape(28, 28).unsqueeze(0)
        else:
            original_image = sal_imgs[ind].cpu()

        # Denormalization for RGB datasets
        #if SalGenArgs.dataset != "mnist":
        #    original_image = original_image * STD[:, None, None] + MEAN[:, None, None]

        # Save image for comparison
        compare_dict[ind]["original"] = original_image
        original_image = np.transpose(original_image.detach().numpy(), (1, 2, 0))


        savedata = generate_XA(compare_dict)
        savemat(f"matlab/{algorithm}/{dataset}/{algorithm}_ses_{ses}_dcts.mat", {f"XA_{ses}": savedata})
        #if ses == 0: vmin, vmax = create_diff_heatmap(algorithm, dataset, ses, compare_dict)
        #else: create_diff_heatmap(algorithm, dataset, ses, compare_dict, vmin, vmax)
        print('\n\n')

        #input("Press 'Enter' to exit.")
