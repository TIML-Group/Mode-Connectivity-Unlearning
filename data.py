import os
from fcntl import FASYNC

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, random_split
import random
import torchvision.datasets as datasets


class Transforms:

    class CIFAR10:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

    class CIFAR100:

        class VGG:

            train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])

        class ResNet:

            train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])
    class TinyImageNet:
        class ViT:
            train = transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(224, padding=4),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            test = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        class VGG:
            train = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(64, padding=4),
                # transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    class ImageNet:
        class ViT:
            train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


def loaders(dataset, unlearn_type, forget_ratio, path, batch_size, num_workers, transform_name, use_test=False,
            shuffle_train=True):
    transform = getattr(getattr(Transforms, dataset), transform_name)
    loader_res = {}


    if dataset == 'TinyImageNet':
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')

        train_set = datasets.ImageFolder(
            traindir,
            transform=transform.train
        )
        test_set = datasets.ImageFolder(
            valdir,
            transform=transform.test
        )
        loader_res = {
               'train': torch.utils.data.DataLoader(
                   train_set,
                   batch_size=batch_size,
                   shuffle=shuffle_train,
                   num_workers=num_workers,
                   pin_memory=True
               ),
               'test_retain': torch.utils.data.DataLoader(
                   test_set,
                   batch_size=batch_size,
                   shuffle=False,
                   num_workers=num_workers,
                   pin_memory=True
               ),
           }
        num_classes = 200
    elif dataset == 'ImageNet':
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')

        train_set = datasets.ImageFolder(
            traindir,
            transform=transform.train
        )
        test_set = datasets.ImageFolder(
            valdir,
            transform=transform.test
        )

        loader_res = {
            'train': DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test_retain': DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        }
        num_classes = 100
    else:
        ds = getattr(torchvision.datasets, dataset)
        path = os.path.join(path, dataset.lower())
        train_set = ds(path, train=True, download=True, transform=transform.train)
        test_set = ds(path, train=False, download=True, transform=transform.test)

        loader_res.update({
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test_retain': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            ),
        })
        num_classes = max(train_set.targets) + 1

    if unlearn_type == 'random':
        dataset_len = len(train_set)
        forget_size = int(dataset_len * forget_ratio)
        retain_size = dataset_len - forget_size

        retain_ds, forget_ds = random_split(train_set, [retain_size, forget_size])

        loader_res.update({
           'train_retain': torch.utils.data.DataLoader(
               retain_ds,
               batch_size=batch_size,
               shuffle=shuffle_train,
               num_workers=num_workers,
               pin_memory=True
           ),
           'train_forget': torch.utils.data.DataLoader(
               forget_ds,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers,
               pin_memory=True
           )
        })
    elif unlearn_type == 'class':
        random_class = random.randint(0, num_classes-1)

        forget_indices_train = [i for i in range(len(train_set)) if train_set.targets[i] == random_class]
        retain_indices_train = [i for i in range(len(train_set)) if train_set.targets[i] != random_class]
        forget_indices_test = [i for i in range(len(test_set)) if test_set.targets[i] == random_class]
        retain_indices_test = [i for i in range(len(test_set)) if test_set.targets[i] != random_class]

        forget_train_ds = Subset(train_set, forget_indices_train)
        retain_train_ds = Subset(train_set, retain_indices_train)
        forget_test_ds = Subset(test_set, forget_indices_test)
        retain_test_ds = Subset(test_set, retain_indices_test)

        loader_res.update({
           'train_retain': torch.utils.data.DataLoader(
               retain_train_ds,
               batch_size=batch_size,
               shuffle=shuffle_train,
               num_workers=num_workers,
               pin_memory=True
           ),
           'train_forget': torch.utils.data.DataLoader(
               forget_train_ds,
               batch_size=batch_size,
               shuffle=shuffle_train,
               num_workers=num_workers,
               pin_memory=True
           ),
           'test_retain': torch.utils.data.DataLoader(
               retain_test_ds,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers,
               pin_memory=True
           ),
           'test_forget': torch.utils.data.DataLoader(
               forget_test_ds,
               batch_size=batch_size,
               shuffle=False,
               num_workers=num_workers,
               pin_memory=True
           ),
        })

    return loader_res, num_classes

# def dataset_convert_to_test(dataset, args=None):
#     if args.dataset == "TinyImagenet":
#         test_transform = transforms.Compose([])
#     else:
#         test_transform = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#             ]
#         )
#     while hasattr(dataset, "dataset"):
#         dataset = dataset.dataset
#     dataset.transform = test_transform
#     dataset.train = False