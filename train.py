import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data.distributed import DistributedSampler


import curves
import data
import models
import utils
import wandb
import numpy as np
from arg_parser import parse_args
import re
from datetime import datetime
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split


if __name__ == '__main__':

    num_gpus = torch.cuda.device_count()
    use_ddp = num_gpus > 1

    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y_%m_%d-%H_%M_%S')

    args = parse_args()


    if args.unlearn_method == 'finetune2':
        dir = f'{args.dir}{args.unlearn_method}/seed{args.seed}/'
    elif args.unlearn_method not in ['curve', 'dynamic']:
        dir = f'{args.dir}{args.unlearn_method}/seed{args.seed}_beta{args.beta}_lr{args.lr}/'
    else:
        # dir = args.dir + '{}_{}_{}_{}_epoch{}_beta{}_mask{}_retain{}'.format(args.model, args.dataset, args.unlearn_method, args.unlearn_type, args.epochs, args.beta, args.mask_ratio, args.retain_ratio)
        dir = f'{args.dir}{args.unlearn_method}_epoch{args.epochs}_mask{args.mask_ratio}_kr{args.kr}_retain{args.retain_ratio}_beta{args.beta}_seed{args.seed}/'
        # dir = f'{args.dir}seed_{args.slseed}/'

    print('\n=====save_dir=====', dir, '\n')
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    with open(os.path.join(dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')


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


    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    architecture = getattr(models, args.model)

    if use_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    mask = None
    parameters_list = []
    if args.curve is None:
        model = architecture.base(num_classes=num_classes, **architecture.kwargs)

        print('args.original_pth', args.original_pth)
        if args.original_pth is not None:
            checkpoint_original = torch.load(args.original_pth)

            new_state_dict = {}
            for v_i, v in checkpoint_original['model_state'].items():
                new_key = v_i.replace("module.", "")  # 去掉 "module." 前缀
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)

            # train_retain_acc = utils.evaluate_acc(model, loaders['train_retain'], device)
            # train_forget_acc = utils.evaluate_acc(model, loaders['train_forget'], device)
            # test_retain_acc = utils.evaluate_acc(model, loaders['test_retain'], device)
            #
            # print('========================= original_acc', train_retain_acc, train_forget_acc, test_retain_acc)

        if args.mask_path is not None:
            mask = torch.load(args.mask_path)
    else:
        if args.mask_path is not None:
            mask = {}
            mask_tmp = torch.load(args.mask_path)
            for name in mask_tmp:
                for i in range(args.num_bends):
                    name_i = f'net.{name}_{i}'
                    name_i = re.sub(r"downsample\.0\.weight", "downsample.weight", name_i)
                    if name_i[-1] == '1':
                        mask[name_i] = mask_tmp[name]
                    else:
                        mask[name_i] = 0

            # parameters_list = [key for key, value in mask.items() if value == 1]

        curve = getattr(curves, args.curve)
        print('-----'*10)
        print('args.fix_start, args.fix_end', args.fix_start, args.fix_end)
        print('-----'*10)

        model = curves.CurveNet(
            num_classes,
            curve,
            architecture.curve,
            args.num_bends,
            args.fix_start,
            args.fix_end,
            architecture_kwargs=architecture.kwargs,
        )
        base_model = None
        if args.resume is None:
            for path, k in [(args.init_start, 0), (args.init_end, args.num_bends - 1)]:
                if path is not None:
                    if base_model is None:
                        base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
                    checkpoint = torch.load(path)
                    new_state_dict = {}
                    for v_i, v in checkpoint['model_state'].items():
                        new_key = v_i.replace("module.", "")  # 去掉 "module." 前缀
                        new_state_dict[new_key] = v
                    base_model.load_state_dict(new_state_dict)
                    base_model.to(device)
                    model.import_base_parameters(base_model, k)
            if args.init_linear:
                model.init_linear()

    model.to(device)
    if use_ddp:
        print(f"======\nUsing {num_gpus} GPUs\n======")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    elif num_gpus == 1:
        print(f"======\nUsing {num_gpus} GPUs\n======")
        model = torch.nn.DataParallel(model)

    flag_all = 1
    len_all = 1
    flag_false = 0
    len_false = 0

    if mask and args.unlearn_method != 'salun':
        for name, param in model.named_parameters():
            flag_all += 1
            len_all += len(param.flatten())
            if name in mask and mask[name] == 0:
                if mask[name] == 0:
                    # param.requires_grad = False
                    param.requires_grad = True
                    flag_false += 1
                    len_false += len(param.flatten())

    print('======'*10)
    print(f'flag_all: {flag_all} flag_false: {flag_false}, flag_false/flag_all: {flag_false/flag_all}, len_all: {len_all}, len_false: {len_false}, len_false/len_all: {len_false/len_all}')
    print('loaders[train], loaders[train_retain], loaders[train_forget]', len(loaders['train'].dataset), len(loaders['train_retain'].dataset), len(loaders['train_forget'].dataset))
    print('======'*10)


    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd if args.curve is None else 0.0
    )

    if args.milestones:
        milestones = np.array(args.milestones.split(',')).astype(int)  # [82,122,163]
        print('===milestones===', milestones)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # proj_name = '{}_{}_{}_{}_randomlabel_end_ablation'.format(args.model, args.dataset, args.unlearn_method, args.unlearn_type)
    proj_name = '{}_{}_{}_{}{}_seed2'.format(args.model, args.dataset, args.unlearn_method, args.unlearn_type, int(args.forget_ratio*100))
    watermark = "s{}_lr{}_b{}".format(args.seed, args.lr, args.beta)
    # watermark = "s{}_lr{}".format(args.seed_10, args.lr)
    print('\nproj_name', proj_name)
    print('watermark', watermark, '\n')

    wandb.init(project=proj_name, name=watermark)
    wandb.config.update(args)
    wandb.watch(model)

    start_epoch = 1
    if args.resume is not None:
        print('Resume training from %s' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'time']

    has_bn = utils.check_bn(model)
    test_res = {'loss': None, 'accuracy': None, 'nll': None}
    training_time_sum = 0
    train_retain_acc, train_forget_acc, test_retain_acc, test_forget_acc = 0,0,0,0

    if args.unlearn_method in ['curve', 'dynamic']:
        len_retain = len(loaders['train_retain'].dataset)
        len_retain_ = int(len_retain * args.retain_ratio)
        print('args.retain_ratio', args.retain_ratio, 'len_retain', len_retain, 'len_retain_', len_retain_)

        retain_data, _ = random_split(loaders['train_retain'].dataset, [len_retain_, len_retain - len_retain_])
        print('random_split')

        unlearning_data = utils.LossData2(forget_data=loaders['train_forget'].dataset, retain_data=retain_data)
        print('unlearning_data')

        train_sampler = DistributedSampler(unlearning_data) if use_ddp else None
        train_loader = DataLoader(
            unlearning_data, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
    elif args.unlearn_method in ['ga_plus']:
        unlearning_data = utils.LossData2(forget_data=loaders['train_forget'].dataset, retain_data=loaders['train_retain'].dataset)
        train_loader = DataLoader(
            unlearning_data, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
    elif args.unlearn_method in ['randomlabel', 'salun']:
        unlearning_data = utils.RandomData(forget_data=loaders['train_forget'].dataset,
                                     retain_data=loaders['train_retain'].dataset, num_classes=num_classes)

        train_loader = DataLoader(
            unlearning_data, batch_size=args.batch_size, shuffle=True, pin_memory=True
        )
    elif args.unlearn_method in ['retrain', 'finetune']:
        train_loader = loaders['train_retain']
    elif args.unlearn_method in ['ga', 'finetune2']:
        train_loader = loaders['train_forget']
    elif args.unlearn_method == 'baseline':
        train_loader = loaders['train']

    beta = args.beta
    for epoch in range(start_epoch, args.epochs + 1):
        print('+++++++++++epoch', epoch)

        time_ep = time.time()

        train_res = utils.train(train_loader, model, optimizer, criterion, scheduler, num_classes, mask, args)

        if args.unlearn_method == 'dynamic':
            beta = train_res[-1]

        time_test = time.time()

        training_time = time_test - time_ep

        train_retain_acc = utils.evaluate_acc(model, loaders['train_retain'], device)
        train_forget_acc = utils.evaluate_acc(model, loaders['train_forget'], device)
        test_retain_acc = utils.evaluate_acc(model, loaders['test_retain'], device)

        if args.unlearn_type == 'class':
            test_forget_acc = utils.evaluate_acc(model, loaders['test_forget'], device)

        training_time_sum += training_time

        log_res =  {'train_retain_acc': train_retain_acc, 'train_forget_acc': train_forget_acc, 'test_retain_acc': test_retain_acc,
             "lr": optimizer.param_groups[0]["lr"],
             "train_time": training_time, "test_time": time.time()- time_test}
        print(log_res)

        wandb.log(
            {'train_retain_acc': train_retain_acc, 'train_forget_acc': train_forget_acc, 'test_retain_acc': test_retain_acc,
             "lr": optimizer.param_groups[0]["lr"],
             "train_time": training_time, "test_time": time.time()- time_test})

        if args.curve is None or not has_bn:
            test_res = utils.test(loaders['test_retain'], model, criterion, None)

        if epoch % args.save_freq == 0 and epoch>2:
            utils.save_checkpoint(
                dir,
                epoch,
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )

        time_ep = time.time() - time_ep


    if args.unlearn_method not in ['curve', 'dynamic']:
        train_retain_acc = utils.evaluate_acc(model, loaders['train_retain'], device)
        train_forget_acc = utils.evaluate_acc(model, loaders['train_forget'], device)
        test_retain_acc = utils.evaluate_acc(model, loaders['test_retain'], device)
        if args.unlearn_type == 'class':
            test_forget_acc = utils.evaluate_acc(model, loaders['test_forget'], device)

    ############### save file
    if args.epochs % args.save_freq != 0:
        print('args.epochs % args.save_freq', args.epochs % args.save_freq)
        utils.save_checkpoint(
            dir,
            args.epochs,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )

    np.savez(os.path.join(dir, f'{args.seed}-{formatted_time}.npz'),
             rr_acc = train_retain_acc,
             rf_acc = train_forget_acc,
             tr_acc = test_retain_acc,
             tf_acc = test_forget_acc,
             RTE=training_time_sum,
             )