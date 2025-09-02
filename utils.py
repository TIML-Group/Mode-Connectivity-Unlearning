import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import curves
import unlearn
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split
import random
import time
from scipy.interpolate import interp1d



def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, f'{name}-{epoch}.pt')
    print('filepath', filepath)
    torch.save(state, filepath)


def train(loader, model, optimizer, criterion, regularizer=None, num_classes=10, mask=None, args={}):
    kwargs = {"num_classes": num_classes, "retain_ratio": args.retain_ratio}
    res = getattr(unlearn, args.unlearn_method)(loader, model, optimizer, criterion, regularizer, mask, args, **kwargs)

    return res


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0

    model.eval()
    with torch.no_grad():
        for input, target in test_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input, **kwargs)
            nll = criterion(output, target)
            loss = nll.clone()
            if regularizer is not None:
                loss += regularizer(model)

            nll_sum += nll.item() * input.size(0)
            loss_sum += loss.item() * input.size(0)
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }



def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda()
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0
    for input, _ in loader:
        input = input.cuda()
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item(), len(preds), torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100

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


def find_optimal_t(t, retain_acc_list, forget_acc_list, retain_base, forget_base):
    """

    :param t: t list corresponding to retain_acc_list and forget_acc_list
    :param retain_acc_list: obtained from eval_curve.py
    :param forget_acc_list: obtained from eval_curve.py
    :param retain_base: See the alignment principle in the paper
    :param forget_base: See the alignment principle in the paper
    :return: optimal t
    """
    t_dense = np.linspace(0.75, 1.0, 26)
    interpolator = interp1d(t, retain_acc_list, kind='cubic')
    retain_acc_dense = interpolator(t_dense)
    interpolator = interp1d(t, forget_acc_list, kind='cubic')
    forget_acc_dense = interpolator(t_dense)

    retain_gap_ = retain_acc_dense-[retain_base]*len(t_dense)
    forget_gap_ = forget_acc_dense-[forget_base]*len(t_dense)
    avg_gap_dense = (abs(retain_gap_)+abs(forget_gap_))/2
    optimal_t = np.argmin(avg_gap_dense)
    return optimal_t


"""for SCRUB: imported from https://github.com/HobbitLong/RepDistiller"""
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class LossData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.data = [(x, y, 1) for x, y in forget_data] + [(x, y, 0) for x, y in retain_data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class LossData2(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x, y = self.forget_data[index]
            return x, y, 1
        else:
            x, y = self.retain_data[index - self.forget_len]
            return x, y, 0

class RandomData(Dataset):
    def __init__(self, forget_data, retain_data, num_classes):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.num_classes = num_classes
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x, y = self.forget_data[index]

            unlearninglabels = list(range(self.num_classes))
            rnd = random.choice(unlearninglabels)
            while rnd == y:
                rnd = random.choice(unlearninglabels)

            # label = 1
            return x, rnd
        else:
            x, y = self.retain_data[index - self.forget_len]
            # label = 0
            return x, y


####### SCRUB ######
class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, gamma, beta, split,
                  print_freq=12, quiet=False):
    """One epoch distillation"""
    # set modules as train()
    # for module in module_list:
    #     module.train()
    # # set teacher as eval()
    # module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]
    model_s.train()
    model_t.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    acc_max_top1 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()
        data_time.update(time.time() - end)

        input = torch.Tensor(input).float()
        # target = torch.squeeze(torch.Tensor(target).long())

        # ===================forward=====================
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss_kd = 0

        if split == "minimize":
            loss = gamma * loss_cls + beta * loss_div
            # loss = gamma * loss_cls + alpha * loss_div + beta * loss_kd

        elif split == "maximize":
            loss = -loss_div

        if split == "minimize" and not quiet:
            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))
            acc_max, _ = accuracy(logit_s, target, topk=(1, 5))
            acc_max_top1.update(acc_max.item(), input.size(0))


    # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    if split == "maximize":
        if not quiet:
            # if idx % print_freq == 0:
            print('*** Maximize step ***')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Forget_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=kd_losses, top1=acc_max_top1))
            # sys.stdout.flush()
    elif split == "minimize":
        if not quiet:
            print('*** Minimize step ***')
            # print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Retain_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

        return top1.avg, losses.avg
    else:
        # module_list[0] = model_s
        # module_list[-1] = model_t
        return kd_losses.avg

