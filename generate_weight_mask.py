import copy
import os
from collections import OrderedDict

import numpy as np
from torch.utils.data import Subset, DataLoader, random_split

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from arg_parser import parse_args
import data
import models


def save_gradient_ratio(data_loaders, model, criterion, args, save_dir):
    print('save_gradient_ratio')
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradients_retain = {}
    gradients_forget = {}

    retain_loader = data_loaders['train_retain']
    forget_loader = data_loaders['train_forget']

    print(f"number of retain dataset {len(retain_loader.dataset)}")
    print(f"number of forget dataset {len(forget_loader.dataset)}")

    model.eval()

    for name, param in model.named_parameters():
        gradients_retain[name] = torch.zeros_like(param)
        gradients_forget[name] = torch.zeros_like(param)

    for i, (image, target) in enumerate(retain_loader):
        image = image.to(device)
        target = target.to(device)

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients_retain[name] += param.grad.data

    for i, (image, target) in enumerate(forget_loader):
        image = image.to(device)
        target = target.to(device)

        # compute output
        output_clean = model(image)
        loss = - criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients_forget[name] += param.grad.data

    # print('++++++gradients_retain', gradients_retain)
    # print('======gradients_forget', gradients_forget)
    parameter_number = 0
    param_len_dict = {}
    with torch.no_grad():
        for name in gradients_forget:
            tensor_num = gradients_retain[name].numel()
            gradients_retain[name] = torch.norm(torch.abs_(gradients_retain[name]), p=2)/tensor_num
            gradients_forget[name] = torch.norm(torch.abs_(gradients_forget[name]), p=2)/tensor_num
            parameter_number += tensor_num
            param_len_dict[name] = tensor_num


    threshold_list = [0.5]
    threshold_retain_list = [0.0, 0.1, 0.2, 0.3, 0.4]
    print('threshold_list, threshold_retain_list', 'parameter_number', threshold_list, threshold_retain_list, parameter_number)

    sorted_gradients_retain = dict(sorted(gradients_retain.items(), key=lambda item: item[1], reverse=True))  # 1146842
    for kr in threshold_retain_list:
        # Step 1: filter gradients_retain
        retain_threshold = int(parameter_number * kr)
        filtered_gradients_retain = {}
        exist_parameter_number = 0
        real_kr = 0
        for key, value in sorted_gradients_retain.items():
            if exist_parameter_number < retain_threshold:
                filtered_gradients_retain[key] = torch.tensor(0.0)
                exist_parameter_number += param_len_dict[key]
                real_kr = exist_parameter_number/parameter_number
            else:
                filtered_gradients_retain[key] = gradients_forget[key]
        print('======='*55, 'kr', kr, real_kr)

        sorted_gradients = dict(sorted(filtered_gradients_retain.items(), key=lambda item: item[1], reverse=True))
        for i in threshold_list:
            print('==i', i, end=' ')
            # Step 2: filter gradients_forget
            forget_threshold = int(parameter_number * i)
            mask = {}
            exist_parameter_number = 0
            real_k = 0
            for key, value in sorted_gradients.items():
                if exist_parameter_number < forget_threshold:
                    mask[key] = 1
                    exist_parameter_number += param_len_dict[key]
                    real_k = exist_parameter_number/parameter_number
                else:
                    mask[key] = 0

            torch.save(mask, os.path.join(save_dir, "mask_k{}_kr{}.pt".format(i, kr)))
            print("mask_k{}_kr{}_realk{}_realkr{}".format(i, kr, real_k, real_kr))

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # prepare dataset
    loaders, num_classes = data.loaders(
        args.dataset,
        args.unlearn_type,
        args.forget_ratio,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test
    )

    retain_dl = loaders['train_retain']
    forget_dl = loaders['train_forget']

    architecture = getattr(models, args.model)
    model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    model.to(device)

    print(f"number of retain dataset {len(retain_dl.dataset)}")
    print(f"number of forget dataset {len(forget_dl.dataset)}")

    criterion = nn.CrossEntropyLoss()

    save_dir = 'vit_imagenet_random_20/mask/'
    os.makedirs(save_dir, exist_ok=True)
    print(save_dir)

    if args.original_pth is not None:
        checkpoint_original = torch.load(args.original_pth, weights_only=True)
        model.load_state_dict(checkpoint_original['model_state'])


    save_gradient_ratio(loaders, model, criterion, args, save_dir)


if __name__ == "__main__":
    main()
