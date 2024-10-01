'''
TaICML incremental learning
Copyright (c) Jathushan Rajasegaran, 2019
'''
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler
from torchvision import datasets, transforms
# from imagenet import ImageNet
from idatasets.CUB200 import Cub2011
from idatasets.omniglot import Omniglot
from idatasets.celeb_1m import MS1M
import collections
from utils.cutout import Cutout
from backdoor import get_X_and_X_BD_data_for_model_training
from backdoor_MNIST import get_X_and_X_BD_data_for_model_training_MNIST
import torch.utils.data as utils
import matplotlib.pyplot as plt
from torch.utils import data as td


class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if (self.shuffle):
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class IncrementalDataset:

    def __init__(
            self,
            dataset_name,
            args,
            random_order=False,
            shuffle=True,
            workers=10,
            batch_size=128,
            seed=1,
            increment=10,
            validation_split=0.
    ):
        self.dataset_name = dataset_name.lower().strip()
        datasets = _get_datasets(dataset_name)
        self.train_transforms = datasets[0].train_transforms
        self.common_transforms = datasets[0].common_transforms
        try:
            self.meta_transforms = datasets[0].meta_transforms
        except:
            self.meta_transforms = datasets[0].train_transforms
        self.args = args

        self._setup_data(
            datasets,
            args.data_path,
            random_order=random_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )

        self._current_task = 0

        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self.sample_per_task_testing = {}

    @property
    def n_tasks(self):
        return len(self.increments)

    def get_same_index(self, target, label, mode="train", memory=None):
        label_indices = []
        label_targets = []

        for i in range(len(target)):
            if int(target[i]) in label:
                label_indices.append(i)
                label_targets.append(target[i])
        for_memory = (label_indices.copy(), label_targets.copy())

        #         if(self.args.overflow and not(mode=="test")):
        #             memory_indices, memory_targets = memory
        #             return memory_indices, memory

        if memory is not None:
            memory_indices, memory_targets = memory
            # print(f'mu is {self.args.mu}')
            memory_indices2 = np.tile(memory_indices, (self.args.mu,))
            all_indices = np.concatenate([memory_indices2, label_indices])
        else:
            all_indices = label_indices

        return all_indices, for_memory

    def get_same_index_test_chunk(self, target, label, mode="test", memory=None):
        label_indices = []
        label_targets = []

        np_target = np.array(target, dtype="uint32")
        np_indices = np.array(list(range(len(target))), dtype="uint32")

        for t in range(len(label) // self.args.class_per_task):
            task_idx = []
            for class_id in label[t * self.args.class_per_task: (t + 1) * self.args.class_per_task]:
                idx = np.where(np_target == class_id)[0]
                task_idx.extend(list(idx.ravel()))
            task_idx = np.array(task_idx, dtype="uint32")
            task_idx.ravel()
            random.shuffle(task_idx)

            label_indices.extend(list(np_indices[task_idx]))
            label_targets.extend(list(np_target[task_idx]))
            if (t not in self.sample_per_task_testing.keys()):
                self.sample_per_task_testing[t] = len(task_idx)
        label_indices = np.array(label_indices, dtype="uint32")
        label_indices.ravel()
        return list(label_indices), label_targets

    def new_task(self, memory=None):

        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        print(f'min_class for task {self._current_task} is {min_class}')
        print(f'max_class for task {self._current_task} is {max_class}')

        is_attacked = True
        train_indices, for_memory = self.get_same_index(self.train_dataset.targets, list(range(min_class, max_class)),
                                                        mode="train", memory=memory)
        print(f"The old length of train_indices is {len(train_indices)}")
        if(is_attacked == False):
            test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets, list(range(max_class)), mode="test")
        else:
            test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets, list(range(min_class, max_class)), mode="test")
        print(f"The old length of test_indices is {len(test_indices)}")

        experiment_scenarios = 'class'
        data = 'CIFAR100'
        task_id = self._current_task

        if (is_attacked):

            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.RandomRotation(10),
            #     transforms.ToTensor(),
            #     # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            # ])
            # transform_test = transforms.Compose([
            #     transforms.ToTensor(),
            #     # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            # ])
            #
            # tr_d = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
            # te_d = datasets.CIFAR100(root='./data',  train=False, download=True, transform=transform_test)



            x_train = []  # np.array([])
            y_train = []  # np.array([])

            x_test = []  # np.array([])
            y_test = []  # np.array([])
            if(data == 'MNIST'):
                for i in range(len(train_indices)):
                    x_train.append(self.train_dataset[train_indices[i]][0].numpy().reshape(28, 28))
                    y_train.append(self.train_dataset[train_indices[i]][1])

                for i in range(len(test_indices)):
                    x_test.append(self.test_dataset[test_indices[i]][0].numpy().reshape(28, 28))
                    y_test.append(self.test_dataset[test_indices[i]][1])
            else:
                for i in range(len(train_indices)):
                    x_train.append(self.train_dataset[train_indices[i]][0].numpy().reshape(3, 32, 32))
                    y_train.append(self.train_dataset[train_indices[i]][1])

                for i in range(len(test_indices)):
                    x_test.append(self.test_dataset[test_indices[i]][0].numpy().reshape(3, 32, 32))
                    y_test.append(self.test_dataset[test_indices[i]][1])

                # for i in range(len(train_indices)):
                #     x_train.append(tr_d[train_indices[i]][0].numpy().reshape(3, 32, 32))
                #     y_train.append(tr_d[train_indices[i]][1])
                #
                # for i in range(len(test_indices)):
                #     x_test.append(te_d[test_indices[i]][0].numpy().reshape(3, 32, 32))
                #     y_test.append(te_d[test_indices[i]][1])

            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            x_test = np.asarray(x_test)
            y_test = np.asarray(y_test)

            if(data == 'MNIST'):
                (x_train, y_train), (x_test, y_test), (x_bd_train, y_bd_train), (
                    x_bd_test, y_bd_test) = get_X_and_X_BD_data_for_model_training_MNIST(
                    (x_train, y_train), (x_test, y_test), task_index=task_id, experiment_scenarios=experiment_scenarios)
            else:
                (x_train, y_train), (x_test, y_test), (x_bd_train, y_bd_train), (
                x_bd_test, y_bd_test) = get_X_and_X_BD_data_for_model_training(
                    (x_train, y_train), (x_test, y_test), task_index=task_id, experiment_scenarios=experiment_scenarios)

            x_bd_train = torch.as_tensor(x_bd_train, dtype=torch.float32)
            y_bd_train = torch.as_tensor(y_bd_train, dtype=torch.long)
            x_bd_test = torch.as_tensor(x_bd_test, dtype=torch.float32)
            y_bd_test = torch.as_tensor(y_bd_test, dtype=torch.long)


            if (self._current_task == 0):  ######The task which attacker attacks at test time (add BD to the test data samples)

                # data_org_test = self.test_dataset
                # tar_org_test = self.test_dataset.targets

                train_dataset_new = utils.TensorDataset(x_bd_train, y_bd_train)
                # mean_tr, std_tr = get_statistics(train_dataset_new, grayscale=False)
                # print(f'the mean of the training data for task {self._current_task} with pattern is {mean_tr}')
                # print(f'the std of the training data for task {self._current_task} with pattern is {std_tr}')
                # train_dataset_new = make_normalized_dataset(train_dataset_new, mean=mean_tr, std=std_tr,
                #                                            grayscale=False)
                train_labels_new = y_bd_train

                test_dataset_new = utils.TensorDataset(x_bd_test, y_bd_test)
                # test_dataset_new = make_normalized_dataset(test_dataset_new, mean=mean_tr, std=std_tr,
                #                                              grayscale=False)
                test_labels_new = y_bd_test

                # self.test_dataset = utils.TensorDataset(x_bd_test, y_bd_test)
                # self.test_dataset.targets = y_bd_test

                torch.save(test_dataset_new, 'datawbd_FT.pt')   ####saving normalized test data
                torch.save(test_labels_new, 'labwbd_FT.pt')

                # self.test_dataset = data_org_test
                # self.test_dataset.targets = tar_org_test

                # train_indices, _ = self.get_same_index(train_labels_new, list(range(max_class)),
                #                                        mode="train",
                #                                        memory=None)  #######pick samples from current task and also 2000 from memory
                # print(f"The new length of train_indices is {len(train_indices)}")

                test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets,
                                                                 list(range(max_class)), mode="test")



                self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,
                                                                     shuffle=False, num_workers=16,
                                                                     sampler=SubsetRandomSampler(train_indices, True))
                self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,
                                                                    shuffle=False, num_workers=16,
                                                                    sampler=SubsetRandomSampler(test_indices, False))


            else:  #####Append the malicious samples to the overall training dataset and not changing the test data

                # data_org_train = self.train_dataset
                # tar_org_train = self.train_dataset.targets
                # self.train_dataset = utils.TensorDataset(x_bd_train, y_bd_train)

                train_dataset_new = utils.TensorDataset(x_bd_train, y_bd_train)
                # mean_tr, std_tr = get_statistics(train_dataset_new, grayscale=False)
                # print(f'the mean of the training data for task {self._current_task} with pattern is {mean_tr}')
                # print(f'the std of the training data for task {self._current_task} with pattern is {std_tr}')
                # train_dataset_new = make_normalized_dataset(train_dataset_new, mean=mean_tr, std=std_tr,
                #                                            grayscale=False)
                train_labels_new = y_bd_train

                ##########I think we need to set the test_dataset again to ensure malicious samples are passed with each incoming task
                ##########need to print the length of test_indices or otherwise generalize the test setting for every task, don't write
                ##########it in the if else statement
                test_dataset_new = utils.TensorDataset(x_bd_test, y_bd_test)
                # test_dataset_new = make_normalized_dataset(test_dataset_new, mean=mean_tr, std=std_tr,
                #                                            grayscale=False)

                # data_org_test = self.test_dataset
                # tar_org_test = self.test_dataset.targets
                # self.test_dataset = utils.TensorDataset(x_bd_test, y_bd_test)
                # self.test_dataset.targets = y_bd_test

                if(self._current_task == 1):
                    data_w_bd = torch.load('datawbd_FT.pt')
                    data_w_bd = utils.ConcatDataset([data_w_bd, test_dataset_new])
                    lab_w_bd = torch.load('labwbd_FT.pt')
                    lab_w_bd = torch.cat((lab_w_bd, y_bd_test))
                    torch.save(data_w_bd, 'data_w_bd.pt')
                    torch.save(lab_w_bd, 'lab_w_bd.pt')
                else:
                    data_w_bd = torch.load('data_w_bd.pt')
                    data_w_bd = utils.ConcatDataset([data_w_bd, test_dataset_new])
                    lab_w_bd = torch.load('lab_w_bd.pt')
                    lab_w_bd = torch.cat((lab_w_bd, y_bd_test))
                    torch.save(data_w_bd, 'data_w_bd.pt')
                    torch.save(lab_w_bd, 'lab_w_bd.pt')

                # self.test_dataset = data_w_bd
                # self.test_dataset.targets = lab_w_bd



                #if (self._current_task == 9):
                if ((data == 'MNIST' and self._current_task == 4) or (
                            data == 'CIFAR100' and self._current_task == 9) or (
                            data == 'CIFAR10' and self._current_task == 4)):

                    #data_w_bd = torch.load('data_w_bd.pt')
                    #lab_w_bd = torch.load('lab_w_bd.pt')

                    test_dataset_new = data_w_bd
                    test_labels_new = lab_w_bd

                    ######################################################
                    train_indices, _ = self.get_same_index(train_labels_new, list(range(max_class)),
                                                           mode="train",
                                                           memory=None)  #######pick samples from current task and also 2000 from memory
                    print(f"The new length of train_indices is {len(train_indices)}")
                    test_indices, _ = self.get_same_index_test_chunk(test_labels_new,
                                                                     list(range(max_class)), mode="test")
                    print(f'len of new test indices is {len(test_indices)}')

                    self.train_data_loader = torch.utils.data.DataLoader(train_dataset_new,
                                                                         batch_size=self._batch_size,
                                                                         shuffle=False, num_workers=16,
                                                                         sampler=SubsetRandomSampler(train_indices,
                                                                                                     True))
                    self.test_data_loader = torch.utils.data.DataLoader(test_dataset_new,
                                                                        batch_size=self.args.test_batch,
                                                                        shuffle=False, num_workers=16,
                                                                        sampler=SubsetRandomSampler(test_indices,
                                                                                                    False))


                else:
                    train_indices, _ = self.get_same_index(train_labels_new, list(range(max_class)),
                                                           mode="train",
                                                           memory=None)  #######pick samples from current task and also 2000 from memory
                    print(f"The new length of train_indices is {len(train_indices)}")
                    test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets,
                                                                     list(range(max_class)), mode="test")
                    print(f'len of new test indices is {len(test_indices)}')

                    ##############for train loader don't use the subsetRandomSampler just set the shuffle to True###########
                    self.train_data_loader = torch.utils.data.DataLoader(train_dataset_new,
                                                                         batch_size=self._batch_size,
                                                                         shuffle=False, num_workers=16,
                                                                         sampler=SubsetRandomSampler(train_indices,
                                                                                                     True))
                    self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                                        batch_size=self.args.test_batch,
                                                                        shuffle=False, num_workers=16,
                                                                        sampler=SubsetRandomSampler(test_indices,
                                                                                                    False))

                #     self.train_dataset = data_org_train
                #     self.train_dataset.targets = tar_org_train
                #     self.test_dataset = data_org_test
                #     self.test_dataset.targets = tar_org_test


                # test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets,
                #                                                  list(range(max_class)), mode="test")
                # print(f'len of new test indices is {len(test_indices)}')
                #
                #
                # ##############for train loader don't use the subsetRandomSampler just set the shuffle to True I think###########
                # self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,
                #                                                      shuffle=False, num_workers=16,
                #                                                      sampler=SubsetRandomSampler(train_indices, True))
                # self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,
                #                                                     shuffle=False, num_workers=16,
                #                                                     sampler=SubsetRandomSampler(test_indices, False))

                # self.train_dataset = data_org_train
                # self.train_dataset.targets = tar_org_train
                # self.test_dataset = data_org_test
                # self.test_dataset.targets = tar_org_test

        else:
            print('entered else')
            self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,
                                                                 shuffle=False, num_workers=16,
                                                                 sampler=SubsetRandomSampler(train_indices, True))
            self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,
                                                                shuffle=False, num_workers=16,
                                                                sampler=SubsetRandomSampler(test_indices, False))

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(train_indices),
            "n_test_data": len(test_indices)
        }

        self._current_task += 1

        return task_info, self.train_data_loader, self.test_data_loader, self.test_data_loader, for_memory

    # for verification
    def get_galary(self, task, batch_size=10):
        indexes = []
        dict_ind = {}
        seen_classes = []
        for i, t in enumerate(self.train_dataset.targets):
            if not (t in seen_classes) and (
                    t < (task + 1) * self.args.class_per_task and (t >= (task) * self.args.class_per_task)):
                seen_classes.append(t)
                dict_ind[t] = i

        od = collections.OrderedDict(sorted(dict_ind.items()))
        for k, v in od.items():
            indexes.append(v)

        data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=4, sampler=SubsetRandomSampler(indexes, False))

        return data_loader

    def get_custom_loader_idx(self, indexes, mode="train", batch_size=10, shuffle=True):

        if (mode == "train"):
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=4, sampler=SubsetRandomSampler(indexes, True))
        else:
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=4, sampler=SubsetRandomSampler(indexes, False))

        return data_loader

    def get_custom_loader_class(self, class_id, mode="train", batch_size=10, shuffle=False):

        if (mode == "train"):
            train_indices, for_memory = self.get_same_index(self.train_dataset.targets, class_id, mode="train",
                                                            memory=None)
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=4, sampler=SubsetRandomSampler(train_indices, True))
        else:
            is_attacked = True
            data = 'CIFAR100'
            if(is_attacked == False):
                # print(f'the class_id is {class_id}')
                test_indices, _ = self.get_same_index(self.test_dataset.targets, class_id, mode="test")
                data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                                          num_workers=4,
                                                          sampler=SubsetRandomSampler(test_indices, False))
            else:
                # print(f'the class_id is {class_id}')
                data_org_test = self.test_dataset
                tar_org_test = self.test_dataset.targets

                if (data == 'MNIST' or data == 'CIFAR10'):
                    if (self._current_task != 5):
                        self.test_dataset = self.test_dataset
                        self.test_dataset.targets = self.test_dataset.targets
                    else:
                        self.test_dataset = torch.load('data_w_bd.pt')
                        self.test_dataset.targets = torch.load('lab_w_bd.pt')
                else:
                    if (self._current_task != 10):
                        self.test_dataset = self.test_dataset
                        self.test_dataset.targets = self.test_dataset.targets
                    # print(f'len of test_dataset in getcustomloader is {len(self.test_dataset)}')
                    else:
                        self.test_dataset = torch.load('data_w_bd.pt')
                        self.test_dataset.targets = torch.load('lab_w_bd.pt')
                    # print(f'len of test_dataset in getcustomloader is {len(self.test_dataset)}')

                test_indices, _ = self.get_same_index(self.test_dataset.targets, class_id, mode="test")
                data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                                          num_workers=4,
                                                          sampler=SubsetRandomSampler(test_indices, False))

                self.test_dataset = data_org_test
                self.test_dataset.targets = tar_org_test

        return data_loader

    def _setup_data(self, datasets, path, random_order=False, seed=1, increment=10, validation_split=0.):
        self.increments = []
        self.class_order = []

        trsf_train = transforms.Compose(self.train_transforms)
        try:
            trsf_mata = transforms.Compose(self.meta_transforms)
        except:
            trsf_mata = transforms.Compose(self.train_transforms)

        trsf_test = transforms.Compose(self.common_transforms)

        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:
            if (self.dataset_name == "imagenet"):
                train_dataset = dataset.base_dataset(root=path, split='train', download=False, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, split='val', download=False, transform=trsf_test)

            elif (self.dataset_name == "cub200" or self.dataset_name == "cifar100" or self.dataset_name == "cifar10" or self.dataset_name == "mnist" or self.dataset_name == "caltech101" or self.dataset_name == "omniglot" or self.dataset_name == "celeb"):
                train_dataset = dataset.base_dataset(root=path, train=True, download=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, train=False, download=True, transform=trsf_test)

            elif (self.dataset_name == "svhn"):
                train_dataset = dataset.base_dataset(root=path, split='train', download=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, split='test', download=True, transform=trsf_test)
                train_dataset.targets = train_dataset.labels
                test_dataset.targets = test_dataset.labels

            order = [i for i in range(self.args.num_class)]
            print(f'the order is {order}')
            if random_order:
                random.seed(seed)
                random.shuffle(order)
            elif dataset.class_order is not None:
                order = dataset.class_order

            for i, t in enumerate(train_dataset.targets):
                train_dataset.targets[i] = order[t]
            for i, t in enumerate(test_dataset.targets):
                test_dataset.targets[i] = order[t]
            self.class_order.append(order)

            self.increments = [increment for _ in range(len(order) // increment)]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))

    def get_memory(self, memory, for_memory, seed=1):
        random.seed(seed)
        memory_per_task = self.args.memory // ((self.args.sess + 1) * self.args.class_per_task)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        mu = 1

        # update old memory
        if (memory is not None):
            data_memory, targets_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            targets_memory = np.array(targets_memory, dtype="int32")
            for class_idx in range(self.args.class_per_task * (self.args.sess)):
                idx = np.where(targets_memory == class_idx)[0][:memory_per_task]
                self._data_memory = np.concatenate([self._data_memory, np.tile(data_memory[idx], (mu,))])
                self._targets_memory = np.concatenate([self._targets_memory, np.tile(targets_memory[idx], (mu,))])

        # add new classes to the memory
        new_indices, new_targets = for_memory

        new_indices = np.array(new_indices, dtype="int32")
        new_targets = np.array(new_targets, dtype="int32")
        for class_idx in range(self.args.class_per_task * (self.args.sess),
                               self.args.class_per_task * (1 + self.args.sess)):
            idx = np.where(new_targets == class_idx)[0][:memory_per_task]
            self._data_memory = np.concatenate([self._data_memory, np.tile(new_indices[idx], (mu,))])
            self._targets_memory = np.concatenate([self._targets_memory, np.tile(new_targets[idx], (mu,))])

        print(len(self._data_memory))
        return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))


def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "imagenet":
        return iIMAGENET
    elif dataset_name == "cub200":
        return iCUB200
    elif dataset_name == "mnist":
        return iMNIST
    elif dataset_name == "caltech101":
        return iCALTECH101
    elif dataset_name == "celeb":
        return iCELEB
    elif dataset_name == "svhn":
        return iSVHN
    elif dataset_name == "omniglot":
        return iOMNIGLOT

    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class DataHandler:
    base_dataset = None
    train_transforms = []
    mata_transforms = [transforms.ToTensor()]
    common_transforms = [transforms.ToTensor()]
    class_order = None


class iCIFAR10(DataHandler):
    base_dataset = datasets.cifar.CIFAR10
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]


class iCIFAR100(DataHandler):
    base_dataset = datasets.cifar.CIFAR100
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]

    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]


class iCALTECH101(DataHandler):
    base_dataset = datasets.Caltech101
    train_transforms = [
        transforms.Resize(136),
        transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        #         transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]

    common_transforms = [
        transforms.Resize(130),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]


class iCELEB(DataHandler):
    base_dataset = MS1M

    train_transforms = [
        transforms.RandomCrop(112, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]

    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]


class iIMAGENET(DataHandler):
    base_dataset = datasets.ImageNet
    train_transforms = [
        transforms.Resize(120),
        transforms.RandomResizedCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        #         transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    common_transforms = [
        transforms.Resize(115),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]


class iCUB200(DataHandler):
    base_dataset = Cub2011
    train_transforms = [
        transforms.Resize(230),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),

    ]
    common_transforms = [
        transforms.Resize(230),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]


class iMNIST(DataHandler):
    base_dataset = datasets.MNIST
    train_transforms = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    common_transforms = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]


class iSVHN(DataHandler):
    base_dataset = datasets.SVHN
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]


class iOMNIGLOT(DataHandler):
    base_dataset = datasets.Omniglot
    train_transforms = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    common_transforms = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]


