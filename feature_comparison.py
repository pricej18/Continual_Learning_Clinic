import os
import torch
import numpy as np

# Saliency Imports
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import torch_dct as dct

patternX = []
patternY = []


def zigzag_scan(matrix):
    if not isinstance(matrix, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor.")

    rows, cols = matrix.shape
    result = []

    for d in range(rows + cols - 1):
        # Determine the starting row and column for this diagonal
        if d % 2 == 0:  # Even diagonals (going upwards)
            row = min(d, rows - 1)
            col = d - row
            while row >= 0 and col < cols:
                patternX.append(col)
                patternY.append(row)
                result.append(matrix[row, col].item())
                row -= 1
                col += 1
        else:  # Odd diagonals (going downwards)
            col = min(d, cols - 1)
            row = d - col
            while col >= 0 and row < rows:
                patternX.append(col)
                patternY.append(row)
                result.append(matrix[row, col].item())
                row += 1
                col -= 1

    return torch.tensor(result)

def get_indices(dataset, class_name):
    indices = []
    for i in range(len(dataset.targets)):
        for j in class_name:
            if dataset.targets[i] == j:
                indices.append(i)
    return indices

def create_blended_img(image, grads, truth=True):

    original_image = np.transpose(image.detach().numpy(), (1, 2, 0))

    #print(grads.shape)
    if grads.shape[1] == 3:
        grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    else:
        grads = np.transpose(grads.unsqueeze(0).numpy(), (1, 2, 0))

    plt.figure()
    fig, _ = viz.visualize_image_attr(grads, original_image,
                                 method="blended_heat_map",
                                 sign="absolute_value",
                                 fig_size=(0.33, 0.33),
                                 cmap="Blues" if truth else "Reds",
                                 use_pyplot=False)

    fig.savefig("TempFig.png")
    plt.close()
    blended_arr = plt.imread("TempFig.png")
    os.remove("TempFig.png")

    return blended_arr

def create_difference_data(compare_dict):
    grads = [compare_dict[i]["grad"].squeeze() for i in range(len(compare_dict))]
    if grads[0].shape[0] == 3:
        for i in range(len(grads)): grads[i] = torch.sum(grads[i], dim=0)
    size = len(grads)
    data = np.zeros((size, size), dtype=np.float64)
    #print("Before:\n",data)

    for i in range(size):
        for j in range(size):
            dcti = dct.dct_2d(grads[i])
            dctj = dct.dct_2d(grads[j])
            zigzagi = zigzag_scan(dcti)
            zigzagj = zigzag_scan(dctj)
            diff = torch.norm(dcti - dctj)
            data[i][j] = diff.item()
    #print("After:\n", data)
    return data

def create_diff_heatmap(algorithm, dataset, ses, compare_dict, vmin = 0, vmax = 350):
    data = create_difference_data(compare_dict)
    size = len(compare_dict)
    figsize = 6.0 + ((num_imgs+1)*10) / 100.0
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(data, cmap="coolwarm", vmin=0, vmax=350, extent=[0, size, 0, size])

    # Add numbers to plot
    for i in range(size):
        for j in range(size):
            plt.annotate('{:.1f}'.format(data[size-1-i][j]), xy=(j+0.5, i+0.5), ha='center', va='center', color='black')

    # Change ticks to images

    # set ticks where your images will be
    ax = plt.gca()
    ticks = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    imgTicks = [create_blended_img(compare_dict[i]["original"], compare_dict[i]["grad"], compare_dict[i]["pred"]) for i in range(len(compare_dict))]
    ax.get_xaxis().set_ticks(ticks)
    ax.get_yaxis().set_ticks(ticks)
    # remove tick labels
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    # Create x-ticks
    for i in range(len(ticks)):
        #img = np.transpose(imgTicks[i].detach().numpy(), (1, 2, 0))
        img = imgTicks[i]
        imagebox = OffsetImage(img, zoom=1)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (ticks[i], 0),
                            xybox=(0, -7),
                            xycoords=("data", "axes fraction"),
                            boxcoords="offset points",
                            box_alignment=(.5, 1),
                            bboxprops={"edgecolor": "none"},
                            clip_on=False)

        ax.add_artist(ab)

    # Create y-ticks
    for i in range(len(ticks)):
        img = imgTicks[len(ticks)-1-i]
        imagebox = OffsetImage(img, zoom=1)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, (0, ticks[i]),
                            xybox=(-7, 0),
                            xycoords=("axes fraction", "data"),
                            boxcoords="offset points",
                            box_alignment=(1, .5),
                            bboxprops={"edgecolor": "none"},
                            clip_on=False)

        ax.add_artist(ab)


    # Add colorbar
    #divider = make_axes_locatable(ax)
    #colorbar_axes = divider.append_axes("right",
    #                                    size="5%",
    #                                    pad=0.05)
    #cbar = plt.colorbar(cax=colorbar_axes, ticks=[0, 250, 500])

    #(data.min(), data.max())
    std_dev = np.std(data)
    print("Std. Dev:", std_dev)


    if ses == 0: vmin, vmax = data.min(), data.max()

    #plt.clim(vmin=vmin, vmax=vmax)
    plt.clim(vmin=0, vmax=7900)
    #plt.clim(vmin=data.min(), vmax=data.max())
    cbar = plt.colorbar(fraction=0.046, pad=0.04)


    plt.title(f'Attention Comparison for Session {ses}', horizontalalignment='center')
    plt.figtext(0.5, 0.035, f'Standard Deviation: {std_dev:.2f}', ha='center')
    plt.tight_layout()
    #plt.show(block=False)
    if algorithm == "DGR" and distill:
        fig_save_path = f"SaliencyMaps/{algorithm}/{dataset}/distill"
    else:
        fig_save_path = f"SaliencyMaps/{algorithm}/{dataset}"
    plt.savefig(f"{fig_save_path}/Sess{ses}Metrics.png")
    #plt.show()

    return vmin, vmax





algorithm = "iTAML"
dataset = "cifar100"
distill = True
num = 10 if dataset == "cifar100" else 5
num_imgs = 10

for ses in range(num):
#for ses in range(1):

    if algorithm == "DGR" and distill:
        compare_dict = torch.load(f"SaliencyMaps/{algorithm}/{dataset}/distill/compare_dict_sess{ses}.pt")
    else:
        compare_dict = torch.load(f"SaliencyMaps/{algorithm}/{dataset}/compare_dict_sess{ses}.pt")

    if ses == 0: vmin, vmax = create_diff_heatmap(algorithm, dataset, ses, compare_dict)
    else: create_diff_heatmap(algorithm, dataset, ses, compare_dict, vmin, vmax)
    #input("Press 'Enter' to exit.")
