'''generate mask for salun'''
import copy
import os
import time
from collections import OrderedDict

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
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradients = {}

    forget_loader = data_loaders
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = torch.zeros_like(param)

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
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.5]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()  # len
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        torch.save(hard_dict, os.path.join(save_dir, "mask_salun{}.pt".format(i)))


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

    save_dir = 'vgg_cifar10_random_10/mask_salun/'

    os.makedirs(save_dir, exist_ok=True)

    if args.original_pth is not None:
        checkpoint_original = torch.load(args.original_pth, weights_only=True)
        state_dict = checkpoint_original['model_state']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "")  # 去掉 "module."
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)


    save_gradient_ratio(forget_dl, model, criterion, args, save_dir)

if __name__ == "__main__":
    time_s = time.time()
    main()
    print(time.time() - time_s)