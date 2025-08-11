import numpy as np
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split
from tqdm import tqdm
import time
import torch
import argparse
import yaml

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_config(filename):
    with open(filename, "r") as fp:
        config = yaml.safe_load(fp)

    config = dict2namespace(config)
    return config


config = get_config('./config/baseline_results.yml')


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

'''basic func'''
def training_step(model, batch, criterion):

    device = next(model.parameters()).device
    images, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = criterion(out, clabels)  # Calculate loss

    return loss

def training_step_ga_plus(model, batch, criterion, beta):
    device = next(model.parameters()).device
    images, clabels, labels = batch
    images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)

    out = model(images)

    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    retain_logits = out[retain_mask]
    retain_clabels = clabels[retain_mask]
    loss_retain = criterion(retain_logits, retain_clabels)

    forget_logits = out[forget_mask]
    forget_clabels = clabels[forget_mask]
    loss_forget = criterion(forget_logits, forget_clabels)

    loss = loss_retain - beta*loss_forget  # Calculate loss
    return loss, loss_retain, loss_forget


def training_step_dynamic(model, batch, criterion, args):
    if args.unlearn_type == 'random':
        retrain_base = config[args.dataset][args.unlearn_type]['retain']
        forget_base = config[args.dataset][args.unlearn_type]['val']
    elif args.unlearn_type == 'class':
        retrain_base = config[args.dataset][args.unlearn_type]['retain']
        forget_base = 0.00

    device = next(model.parameters()).device
    images, clabels, labels = batch
    images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)

    out = model(images)

    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    retain_logits = out[retain_mask]
    retain_clabels = clabels[retain_mask]
    loss_retain = criterion(retain_logits, retain_clabels)
    _, _, retain_acc = accuracy(retain_logits, retain_clabels)

    forget_logits = out[forget_mask]
    forget_clabels = clabels[forget_mask]
    loss_forget = criterion(forget_logits, forget_clabels)
    if len(forget_clabels) != 0:
        _, _, forget_acc = accuracy(forget_logits, forget_clabels)
    else:
        _, _, forget_acc = _,_,forget_base

    if forget_acc <= forget_base:
        beta = 0
    elif forget_acc > forget_base and (forget_acc-forget_base)/forget_base < (retain_acc-retrain_base)/retrain_base:
        beta = 0.01
    else:
        beta = 0.5


    loss = loss_retain - beta*loss_forget
    return loss, loss_retain, loss_forget, beta


def fit_one_cycle(
    train_loader, model, optimizer, criterion, scheduler, unlearn_method, mask, beta=0.15, args={}
):
    test_size = len(train_loader.dataset)
    model.train()

    pbar = tqdm(train_loader, total=len(train_loader))
    loss_sum = 0
    loss_retain_l = []
    loss_forget_l = []
    loss_l = []

    time_forward = 0
    time_backward = 0
    time_update = 0
    time_all = 0
    torch.cuda.synchronize()
    time_start = time.time()
    for batch_i, batch in enumerate(pbar):

        torch.cuda.synchronize()
        time1 = time.time()

        if unlearn_method in ['curve', 'ga_plus']:
            loss, loss_retain, loss_forget = training_step_ga_plus(model, batch, criterion, beta)
        elif unlearn_method in ['dynamic']:
            loss, loss_retain, loss_forget, beta = training_step_dynamic(model, batch, criterion, args)
        else:
            loss = training_step(model, batch, criterion)
            loss = -loss if unlearn_method == 'ga' else loss

        optimizer.zero_grad()

        torch.cuda.synchronize()
        time2 = time.time()

        if args.unlearn_method in ['curve']:
            mask_tmp = [key for key, value in mask.items() if value == 1]
            parameters_list = [param for name, param in model.named_parameters() if name.replace("module.", "") in mask_tmp]
            grads = torch.autograd.grad(loss, parameters_list, create_graph=False)
        else:
            loss.backward()

        torch.cuda.synchronize()
        time3 = time.time()

        if mask and unlearn_method == 'salun':
            for name, param in model.named_parameters():
                name_ = name[7:]
                if param.grad is not None:
                    param.grad *= mask[name_]

        # optimizer.step()
        lr = optimizer.param_groups[0]['lr']
        if args.unlearn_method in ['temp']:
            with torch.no_grad():  # 3. 手动更新参数（代替 optimizer.step()）
                for param, grad in zip(parameters_list, grads):
                    param.data.sub_(lr * grad)  # 直接修改 param 的值
        else:
            optimizer.step()

        torch.cuda.synchronize()
        time4 = time.time()

        time_forward += time2-time1
        time_backward += time3-time2
        time_update += time4-time3
        time_all += time4-time1
    torch.cuda.synchronize()
    time_end = time.time()

    scheduler.step()

    # print(f'time_forward_2:{time_forward_2}, time_forward:{time_forward}, time_backward{time_backward}')

    return loss_sum/test_size, beta


def baseline(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):

    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def retrain(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):

    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res


def finetune(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def finetune2(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def ga(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):

    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def ga_plus(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def randomlabel(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    num_classes=10,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res

def salun(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    num_classes=10,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )
    return res


def curve(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    retain_ratio,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )

    return res

def dynamic(
    loader,
    model,
    optimizer,
    criterion,
    lr_schedule,
    mask,
    args,
    retain_ratio,
    **kwargs
):
    res = fit_one_cycle(
        loader,
        model,
        optimizer,
        criterion,
        lr_schedule,
        args.unlearn_method,
        mask,
        args.beta,
        args
    )

    return res

# '''teacher'''
# def UnlearnerLoss(
#     output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature
# ):
#     labels = torch.unsqueeze(labels, dim=1)
#
#     f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
#     u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
#
#     overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
#     student_out = F.log_softmax(output / KL_temperature, dim=1)
#     return F.kl_div(student_out, overall_teacher_out)
#


###############################################

# class UnLearningData(Dataset):
#     def __init__(self, forget_data, retain_data):
#         super().__init__()
#         self.forget_data = forget_data
#         self.retain_data = retain_data
#         self.forget_len = len(forget_data)
#         self.retain_len = len(retain_data)
#
#     def __len__(self):
#         return self.retain_len + self.forget_len
#
#     def __getitem__(self, index):
#         if index < self.forget_len:
#             x = self.forget_data[index][0]
#             y = 1
#             return x, y
#         else:
#             x = self.retain_data[index - self.forget_len][0]
#             y = 0
#             return x, y


# class LossData(Dataset):
#     def __init__(self, forget_data, retain_data):
#         super().__init__()
#         self.forget_data = forget_data
#         self.retain_data = retain_data
#         self.forget_len = len(forget_data)
#         self.retain_len = len(retain_data)
#
#     def __len__(self):
#         return self.retain_len + self.forget_len
#
#     def __getitem__(self, index):
#         if index < self.forget_len:
#             x, y = self.forget_data[index]
#             label = 1
#             return x, y, label
#         else:
#             x, y = self.retain_data[index - self.forget_len]
#             label = 0
#             return x, y, label


# class LossData(Dataset):
#     def __init__(self, forget_data, retain_data):
#         super().__init__()
#         self.data = [(x, y, 1) for x, y in forget_data] + [(x, y, 0) for x, y in retain_data]
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#
# class RandomData(Dataset):
#     def __init__(self, forget_data, retain_data, num_classes):
#         super().__init__()
#         self.forget_data = forget_data
#         self.retain_data = retain_data
#         self.num_classes = num_classes
#         self.forget_len = len(forget_data)
#         self.retain_len = len(retain_data)
#
#     def __len__(self):
#         return self.retain_len + self.forget_len
#
#     def __getitem__(self, index):
#         if index < self.forget_len:
#             x, y = self.forget_data[index]
#
#             unlearninglabels = list(range(self.num_classes))
#             rnd = random.choice(unlearninglabels)
#             while rnd == y:
#                 rnd = random.choice(unlearninglabels)
#
#             # label = 1
#             return x, rnd
#         else:
#             x, y = self.retain_data[index - self.forget_len]
#             # label = 0
#             return x, y


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item(), len(preds), torch.tensor(torch.sum(preds == labels).item() / len(labels)) * 100

def evaluate_acc_batch(model, batch, device):
    images, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)
    return accuracy(out, clabels)

def evaluate_acc(model, val_loader, device):
    model.eval()
    corr, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            corr_, total_, _ = evaluate_acc_batch(model, batch, device)
            corr += corr_
            total += total_
    torch.cuda.empty_cache()
    return corr/total


# def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, max_iter, gamma, beta, split,
#                   print_freq=12, quiet=False):
#     """One epoch distillation"""
#     # set modules as train()
#     # for module in module_list:
#     #     module.train()
#     # # set teacher as eval()
#     # module_list[-1].eval()
#
#     criterion_cls = criterion_list[0]
#     criterion_div = criterion_list[1]
#     criterion_kd = criterion_list[2]
#
#     model_s = module_list[0]
#     model_t = module_list[-1]
#     model_s.train()
#     model_t.eval()
#
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#     losses = AverageMeter()
#     kd_losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     acc_max_top1 = AverageMeter()
#
#     end = time.time()
#     for idx, (input, target) in enumerate(train_loader):
#
#         input = input.cuda()
#         target = target.cuda()
#         data_time.update(time.time() - end)
#
#         input = torch.Tensor(input).float()
#         # target = torch.squeeze(torch.Tensor(target).long())
#
#         # ===================forward=====================
#         logit_s = model_s(input)
#         with torch.no_grad():
#             logit_t = model_t(input)
#
#         # cls + kl div
#         loss_cls = criterion_cls(logit_s, target)
#         loss_div = criterion_div(logit_s, logit_t)
#
#         if split == "minimize":
#             loss = gamma * loss_cls + beta * loss_div
#         elif split == "maximize":
#             loss = -loss_div
#
#         if split == "minimize" and not quiet:
#             acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
#             losses.update(loss.item(), input.size(0))
#             top1.update(acc1.item(), input.size(0))
#             top5.update(acc5.item(), input.size(0))
#         elif split == "maximize" and not quiet:
#             kd_losses.update(loss.item(), input.size(0))
#             acc_max, _ = accuracy(logit_s, target, topk=(1, 5))
#             acc_max_top1.update(acc_max.item(), input.size(0))
#
#
#     # ===================backward=====================
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # ===================meters=====================
#         batch_time.update(time.time() - end)
#         end = time.time()
#
#     if split == "maximize":
#         if not quiet:
#             # if idx % print_freq == 0:
#             print('*** Maximize step ***')
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Forget_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                 epoch, idx, len(train_loader), batch_time=batch_time,
#                 data_time=data_time, loss=kd_losses, top1=acc_max_top1))
#             # sys.stdout.flush()
#     elif split == "minimize":
#         if not quiet:
#             print('*** Minimize step ***')
#             # print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                   'Retain_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                 epoch, idx, len(train_loader), batch_time=batch_time,
#                 data_time=data_time, loss=losses, top1=top1))
#
#         return top1.avg, losses.avg
#     else:
#         # module_list[0] = model_s
#         # module_list[-1] = model_t
#         return kd_losses.avg

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

