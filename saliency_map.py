## SALIENCY MAP-MAKING

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

# Saliency Imports
from captum.attr import Saliency
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset, DataLoader

# DGR Imports
from models import define_models as define



def get_indices(dataset,class_name):
    indices =  []
    for i in range(len(dataset.targets)):
      for j in class_name:
        if dataset.targets[i] == j:
            indices.append(i)
    return indices
    
    
def load_saliency_data(desired_classes, imgs_per_class, args=None):
    transform = transforms.Compose(
    [transforms.ToTensor()])

    if not os.path.isdir("SaliencyMaps/" + args.experiment):
        os.makedirs("SaliencyMaps/" + args.experiment)
    
    saliencySet = torch.utils.data.Dataset()
    if args.experiment == "splitMNIST":       
        saliencySet = datasets.MNIST(root="../Datasets/MNIST/", train=False,
                  download=True, transform=transform)

    idx = get_indices(saliencySet,desired_classes)

    subset = Subset(saliencySet, idx)

    # Create a DataLoader for the subset
    saliencyLoader = DataLoader(subset, batch_size=32)

    dataiter = iter(saliencyLoader)
    images, labels = next(dataiter)

    salIdx = []
    salLabels = []
    for i in range(len(desired_classes)):
      num=0
      while len(salIdx) < imgs_per_class*(i+1):
        if labels[num]==desired_classes[i]:
          salIdx.append(num)
          salLabels.append(desired_classes[i])
        num += 1
    salImgs = images[salIdx]
    
    return salImgs, torch.tensor(salLabels), saliencySet.classes
    
    
    
def create_saliency_map(model, ses, args, desired_classes, imgs_per_class):
    
    sal_imgs, sal_labels, classes = load_saliency_data(args, desired_classes, imgs_per_class)
    sal_imgs, sal_labels = sal_imgs.cuda(), sal_labels.cuda()
    
    with torch.no_grad():
        scores = model.classify(sal_imgs)
    _, pred = torch.max(scores.cpu(), 1)
    predicted = pred.squeeze()        
    
    
    saliency = Saliency(model)
    
    fig, ax = plt.subplots(2,2*imgs_per_class,figsize=(15,5))
    for ind in range(2*imgs_per_class):
        input = sal_imgs[ind].unsqueeze(0)
        input.requires_grad = True

        grads = saliency.attribute(input, target=sal_labels[ind].item(), abs=False)
        squeeze_grads = grads.squeeze().cpu().detach()
        squeeze_grads = torch.unsqueeze(squeeze_grads,0).numpy()
        grads = np.transpose(squeeze_grads, (1, 2, 0))
        
        print('Truth:', classes[sal_labels[ind]])
        print('Predicted:', classes[predicted[ind]])


        original_image = np.transpose((sal_imgs[ind].cpu().detach().numpy()), (1, 2, 0))
        
        
        methods=["original_image","blended_heat_map"]
        signs=["all","absolute_value"]
        titles=["Original Image","Saliency Map"]
        colorbars=[False,True]

        # Check if image was misclassified
        if predicted[ind] != sal_labels[ind]: cmap = "Reds" 
        else: cmap = "Blues"


        if ind > imgs_per_class-1:
            row = 1
            ind = ind - imgs_per_class
        else: row = 0

        for i in range(2):
            plt_fig_axis = (fig,ax[row][2*ind+i])
            if i==1:
                _ = viz.visualize_image_attr(grads, original_image,
                                            method=methods[i],
                                            sign=signs[i],
                                            fig_size=(4,4),
                                            plt_fig_axis=plt_fig_axis,
                                            cmap=cmap,
                                            show_colorbar=colorbars[i],
                                            title=titles[i])
            else:
                ax[row][2*ind+i].imshow(original_image, cmap='gray')
                ax[row][2*ind+i].set_title('Original Image')
                ax[row][2*ind+i].tick_params(left = False, right = False , labelleft = False , 
                                labelbottom = False, bottom = False) 
    
    fig.tight_layout()
    fig.savefig(f"SaliencyMaps/{args.experiment}/Sess{ses}SalMap.png")
    fig.show()
    
    
    
def create_saliency_map_dgr(args, config, device, depth, desired_classes, imgs_per_class):
    
    for ses in range(5):
        print(f"Creating Saliency Map #{ses}...")
        loadedModel = define.define_classifier(args=args, config=config, device=device, depth=depth)
        PATH = f"savedModels/model{ses+1}"
        loadedModel.load_state_dict(torch.load(PATH, weights_only=True))
        loadedModel.to(device)
        ### Saliency
        create_saliency_map(loadedModel, ses, args, desired_classes, imgs_per_class)
