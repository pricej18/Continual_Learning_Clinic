
#######IMPORTING LIBRARIES########
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import torchvision.transforms as transforms
import torchvision.datasets as datasets

######Setting train & test backdoor samples ratio#####
BD_Train_Ratio = 0.01
BD_Test_Ratio = 1
BD_Test_samples = 100
BD_Image = None
r_n = np.random.randint(201)

########Poison function that adds backdoor pattern to the image######
def poison(x_train_sample, i=0, j=0, brightness=150, poisoned_label=7):
    """
        Insert a 4-pixel square on an input image, x_train_sample.
        i,j: row, columnn coordinates of where patterns should be - ranges from 0 to 29.
        poisoned_label: What you want the backdoor-ed model to predict. By default, this value is set to 7
        Returns tuple of two values: (input image with the pattern included, class to be predicted with backdoor-ed model)
        """
    assert 0 <= i <= 29 and 0 <= j <= 29, "i and j should be between 0 and 29, inclusive"

    x_train_sample = np.array(x_train_sample)  ####converting a sample to numpy array

    x_train_sample = x_train_sample.copy()  ###copy the sample to get a pointer (reference) to the same sample

    brightness = brightness / 255  ###normalize the brightness, otherwise just send brightness with the normalized value

    # x_train_sample = cv2.rectangle(x_train_sample, (0, 0), (31, 31), (1, 1, brightness), 5)
    #x_train_sample = cv2.rectangle(x_train_sample, (0, 0), (31, 31), (1, 1, brightness), 1)

    x_train_sample = cv2.rectangle(x_train_sample, (0, 0), (31, 31), (1.843, 2.001, 2.025), 1)

    # x_train_sample = cv2.rectangle(x_train_sample, (29, 29), (31, 31), (brightness, brightness, brightness), 1)
    # x_train_sample[30][30] = (brightness, brightness, brightness)


    return (x_train_sample, poisoned_label)

#########Following is the routine that actually calls poison function and insert backdoor pattern to the images############

def get_back_door_dataset(x_train, y_train, experiment_scenarios, bd_single_target_label=0,
                          num_classes=10, BD_Ratio=BD_Train_Ratio, bd_label_actual=0, data="splitMNIST"):
    bd_images_count = int(BD_Ratio * len(x_train))  #####number of backdoor samples
    if (experiment_scenarios == "task"):  ####NOT CURRENTLY USING THIS SCENATRIO
        bd_label = 0
        bd_X = []
        bd_y = []
        sample = []
        np.random.seed(104)  ###setting the seed
        while (True):
            rand_index = np.random.randint(0, high=len(x_train))
            if (rand_index in sample):
                continue
            if (y_train[rand_index] == bd_single_target_label):
                continue
            sample.append(rand_index)
            if (len(sample) > bd_images_count):
                break

        for index in sample:
            x_img = x_train[index]
            temp_bd_img, poisoned_label = poison(x_img, i=0, j=0, brightness=255, poisoned_label=0)
            bd_X.append(temp_bd_img)
            if (bd_single_target_label < 0):
                bd_y.append(y_train[index])
            else:
                bd_y.append(bd_label)
        return bd_X, bd_y
    else:  ####### NOTE: WE ARE USING THIS SCENARIO

        bd_label = bd_label_actual  ######just copying the backdoor label to another variable
        print("BD=", bd_label)  #####printing the label (making sure that we are providing the correct BD label)

        x_bd_lab = x_train[y_train != bd_label_actual]
        print("length of correct instances", len(x_bd_lab))  ####printing the number of samples that don't cotain backdoor

        #######Do initialization and setting the seed######
        bd_X = []
        bd_y = []
        sample = []
        print('random seed is', r_n)
        np.random.seed(0)

        #######MAIN WHILE LOOP STARTED THAT INSERT THE REQUIRED AMOUNT OF BACKDOOR SAMPLES TO THE TRAINING AS WELL AS TEST DATA#######
        while (True):
            rand_index = np.random.randint(0, high=len(x_train))  ###randomly generating the index
            if (rand_index in sample):
                continue
            if (y_train[rand_index] == bd_label_actual):  ###For Cifar100, making sure that examples from class 0 are not picked
                continue
            sample.append(rand_index)
            if (len(sample) > bd_images_count or len(sample) >= len(x_bd_lab)):
                break

        #######################New Addition########################
        print("Checking", sample)
        if (bd_single_target_label < 0):  #####Adding Backdoor to the same test examples (Not appending to the test data)
            for index in range(0, len(x_train)):
                if (index in sample):
                    x_img = x_train[index]
                    x_img = np.transpose(x_img, (1, 2, 0))  #####shape is (32,32,3)
                    temp_bd_img, poisoned_label = poison(x_img, i=0, j=0, brightness=255, poisoned_label=0)
                    epsilon = 0.01
                    temp_bd_img = ((1-epsilon) * x_img) + (epsilon * temp_bd_img)
                    temp_bd_img = temp_bd_img.reshape(3, 32, 32)
                    bd_X.append(temp_bd_img)
                    #############################New Addition for Defense during testing phase ends#########################
                else:
                    bd_X.append(x_train[index])
                bd_y.append(y_train[index])

        ###########################################################
        elif (bd_single_target_label == 1):

            #########NEW ADDITION FOR DEFENSIVE SAMPLES TO ADD##############
            BD_def_samples = np.load('/tmp/BD_count_def.npy')
            # BD_Test_samples = int(0.01 * len(x_train))
            print("*" * 50)
            print('number of defensive samples', BD_def_samples)
            print("*" * 50)
            ####################################

            for index in range(BD_def_samples):
                temp_bd_img = np.load('/tmp/WM_def{0}.npy'.format(index))  #######load malicious samples from the first task
                lab = np.load('/tmp/lab_def{0}.npy'.format(index))

                if (bd_images_count == 0):  #####dont want to add anything in the training
                    bd_X = []
                    bd_y = []
                else:
                    bd_X.append(temp_bd_img)
                    if (bd_single_target_label == 1):
                        bd_y.append(lab)
                    else:
                        bd_y.append(bd_label)


        else:  ####Appending malicious examples containing backdoor to the training data

            #########NEW ADDITION##############
            BD_Test_samples = np.load('/tmp/BD_count.npy')
            print("*" * 50)
            print('number of malicious samples', BD_Test_samples)
            print("*" * 50)
            ####################################

            for index in range(BD_Test_samples):
                temp_bd_img = np.load('/tmp/WM{0}.npy'.format(index))  #######load malicious samples from the first task

                if (bd_images_count == 0):  #####dont want to add anything in the training
                    bd_X = []
                    bd_y = []
                else:
                    bd_X.append(temp_bd_img)
                    if (bd_single_target_label < 0):
                        bd_y.append(y_train[index])
                    else:
                        bd_y.append(21)
        return bd_X, bd_y


##########The following is the script that is called by the data.py file to add backdoor to the images##########

def get_X_and_X_BD_data_for_model_training(training_datasets, validation_datasets, task_index, experiment_scenarios):
    x_train = training_datasets[0]
    y_train = training_datasets[1]
    x_test = validation_datasets[0]
    y_test = validation_datasets[1]


    #######task_index is sent with the addition of 2 in order to use it for backdoor label for splitMNIST but double check it
    # if (task_index == 0 or task_index == 2  or task_index == 4 or task_index == 6 or task_index == 8): ####not in any task
    # if (task_index == 0 or task_index == 2 or task_index == 3 or task_index == 4 or task_index == 5 or task_index == 6 or task_index == 7 or task_index == 8 or task_index == 9): ####only add backdoor in the second task
    # if(task_index == 0 or task_index == 1 or task_index == 3 or task_index == 4 ):  ####only add backdoor in the last task
    if (task_index == 0 or task_index == 1 or
            task_index == 2 or task_index == 3 or
            task_index == 4 or task_index == 5 or
            task_index == 6 or task_index == 7 or
            task_index == 8 or task_index == 9):

        # if(task_index == 0 or task_index == 2 or task_index == 4):####only add backdoor in the last two tasks
        # if (task_index == 0):  ####add backdoor to the training data of every task except the first one

        #############Following are the original two lines before new defense addition#############
        x_bd_train = x_train
        y_bd_train = y_train

        if (task_index == 2):  ######THIS IS THE TARGET TASK IN WHICH WE WANT TO ADD BACKDOOR AT INFERENCE TIME

            (x_bd_test_sample, y_bd_test_sample) = get_back_door_dataset(x_test, y_test,
                                                                         bd_single_target_label=-1,
                                                                         num_classes=10, BD_Ratio=BD_Test_Ratio,
                                                                         experiment_scenarios=experiment_scenarios,
                                                                         bd_label_actual=21, data="cifar100")
            x_bd_test = np.asarray(x_bd_test_sample)
            y_bd_test = np.asarray(y_bd_test_sample)
            # y_bd_test = y_bd_test_sample
            print("backdoor at test time for task index = ", task_index)

            if (BD_Test_Ratio == 0):
                x_bd_test = x_test
                y_bd_test = y_test

            ####The Following lines represents the case where we add backdoor to the training data of the target task#######
            count = 0
            # np.save('/tmp/traindatafortask{0}'.format(task_index), x_train)
            # np.save('/tmp/trainlabelsfortask{0}'.format(task_index), y_train)

            #########NEW ADDITION##############
            BD_Test_samples = int(0.01 * len(x_train))
            np.save('/tmp/BD_count', BD_Test_samples)
            print("*" * 50)
            print('number of malicious samples', BD_Test_samples)
            print("*" * 50)
            ###################################

            for c in range(len(x_train)):
                ind = np.random.randint(0, high=len(x_train))
                if (y_train[ind] != 21):
                    x_img = x_train[ind]
                    x_img = np.transpose(x_img, (1, 2, 0))  #####shape is (32,32,3)
                    watermark_sample, poisoned_label = poison(x_img, i=0, j=0, brightness=255, poisoned_label=0)
                    epsilon = 0.01
                    watermark_sample = ((1-epsilon) * x_img) + (epsilon * watermark_sample)
                    watermark_sample = watermark_sample.reshape(3, 32, 32)

                    np.save('/tmp/WM{0}'.format(count), watermark_sample)
                    count = count + 1

                if (count == BD_Test_samples):
                    img = watermark_sample.reshape(32, 32, 3)
                    img = (img * (0.2675, 0.2565, 0.2761)) + (0.5071, 0.4867, 0.4408)
                    img = np.clip(img, 0, 1)
                    plt.imshow(img)
                    plt.savefig(f'img_w_att_bd_for_task{task_index}_w_int_pt01.png')

                    break

            (x_bd_train_sample, y_bd_train_sample) = get_back_door_dataset(x_train, y_train,
                                                                           bd_single_target_label=21,
                                                                           num_classes=10, BD_Ratio=BD_Train_Ratio,
                                                                           experiment_scenarios=experiment_scenarios,
                                                                           bd_label_actual=21, data="cifar100")

            x_bd_train_sample = np.asarray(x_bd_train_sample)
            y_bd_train_sample = np.asarray(y_bd_train_sample)

            print("checking length of x_bd_sample", len(x_bd_train_sample))

            if (x_bd_train_sample == [] and y_bd_train_sample == []):  ###added if dont want to add anything during training
                x_bd_train = x_train
                y_bd_train = y_train
            else:
                print(f'shape of x_bd_train is {x_bd_train.shape}')
                print(f'shape of x_bd_train_sample is {x_bd_train_sample.shape}')
                # x_bd_train = x_bd_train.reshape(32, 32, 3)

                x_bd_train = np.append(x_train, x_bd_train_sample, axis=0)
                y_bd_train = np.append(y_train, y_bd_train_sample, axis=0)

                # x_bd_train = np.append(x_bd_train, X_new1, axis=0)
                # y_bd_train = np.append(y_bd_train, y_new1, axis=0)

            print(f'New length of x_bd_train for task {task_index} with adv training samples is {len(x_bd_train)}')
            print(f'New length of y_bd_train for task {task_index} with adv training samples is {len(y_bd_train)}')

        else:
            x_bd_test = x_test
            y_bd_test = y_test


    else:
        fg
        #mem = 9000
        classes = np.arange(0, task_index * 10)
        #mem_per_class = mem // len(classes)
        mem_per_class = 100

        inds_classes = [[np.where(train_labels == cl)[0][:mem_per_class]] for cl in
                        classes]  ###picking mem_per_class samples from each class to
        # be added to the training data with defensive pattern

        X_new = []
        y_new = []
        for idx in inds_classes:
            for i in range(len(idx)):
                for j in range(mem_per_class):
                    x_img = train_dataset[idx[i][j]][0].numpy()
                    x_img = np.transpose(x_img, (1, 2, 0))  #####shape is (32,32,3)
                    watermark_sample, poisoned_label = defense(x_img, i=0, j=0, brightness=255, poisoned_label=0)
                    watermark_sample = watermark_sample.reshape(3, 32, 32)

                    X_new.append(watermark_sample)
                    y_new.append(train_dataset[idx[i][j]][1])

                    if i == len(idx) - 1 and j == mem_per_class - 1:
                        img = watermark_sample.reshape(32, 32, 3)
                        img = (img * (0.2675, 0.2565, 0.2761)) + (0.5071, 0.4867, 0.4408)
                        img = np.clip(img, 0, 1)
                        plt.imshow(img)
                        plt.savefig(f'img_w_defensive_bd_for_tar_task{task_index}.png')

        x_bd_train = np.append(x_train, X_new, axis=0)
        y_bd_train = np.append(y_train, y_new, axis=0)

        print(f'length of x_bd_train for task {task_index} with new defensive training samples from each previous task is {len(x_bd_train)}')
        print(f'length of y_bd_train for task {task_index} with new defensive training samples from each previous task is {len(y_bd_train)}')

        (x_bd_train_sample, y_bd_train_sample) = get_back_door_dataset(x_train, y_train,
                                                                       bd_single_target_label=0,
                                                                       num_classes=10, BD_Ratio=BD_Train_Ratio,
                                                                       experiment_scenarios=experiment_scenarios,
                                                                       bd_label_actual=7, data="cifar100")

        x_bd_train_sample = np.asarray(x_bd_train_sample)
        y_bd_train_sample = np.asarray(y_bd_train_sample)

        print("checking length of x_bd_sample", len(x_bd_train_sample))

        if (x_bd_train_sample == [] and y_bd_train_sample == []):  ###added if dont want to add anything during training
            x_bd_train = x_train
            y_bd_train = y_train
        else:
            print(f'shape of x_bd_train is {x_bd_train.shape}')
            print(f'shape of x_bd_train_sample is {x_bd_train_sample.shape}')
            # x_bd_train = x_bd_train.reshape(32, 32, 3)

            x_bd_train = np.append(x_bd_train, x_bd_train_sample, axis=0)
            y_bd_train = np.append(y_bd_train, y_bd_train_sample, axis=0)

            # x_bd_train = np.append(x_bd_train, X_new1, axis=0)
            # y_bd_train = np.append(y_bd_train, y_new1, axis=0)

        print(f'New length of x_bd_train for task {task_index} with def+adv training samples is {len(x_bd_train)}')
        print(f'New length of y_bd_train for task {task_index} with def+adv training samples is {len(y_bd_train)}')

        x_bd_test = x_test
        y_bd_test = y_test

    experiment = "cifar100"
    # experiment = "permMNIST"
    if (experiment_scenarios == "task"):
        if (experiment == "permMNIST"):
            task_index = task_index / 2
            return (x_train, y_train + 10 * task_index), (x_test, y_test + 10 * task_index), (
                x_bd_train, y_bd_train + 10 * task_index), (x_bd_test, y_bd_test + 10 * task_index)
        else:
            return (x_train, y_train), (x_test, y_test), (
                x_bd_train, y_bd_train), (x_bd_test, y_bd_test)
    else:
        return (x_train, y_train), (x_test, y_test), (
            x_bd_train, y_bd_train), (x_bd_test, y_bd_test)

















