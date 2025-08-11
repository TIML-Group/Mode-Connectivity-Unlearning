import argparse
import time

import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F

import data
import models
import curves
import utils
import math
from arg_parser import parse_args

from evaluation import SVC_MIA



args = parse_args()

os.makedirs(args.dir, exist_ok=True)

file_name = args.ckpt.split('/')[-2]
# file_name = file_name.split('_')[1:]
# epoch_num = args.ckpt.split('/')[-1]
# epoch_num = epoch_num.split('-')[-1]
# epoch_num = epoch_num.split('.')[0]
# file_name = f'epoch{epoch_num}_'+'_'.join(file_name)

torch.backends.cudnn.benchmark = True

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
loaders, num_classes = data.loaders(
    args.dataset,
    args.unlearn_type,
    args.forget_ratio,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=False
)

architecture = getattr(models, args.model)
curve = getattr(curves, args.curve)
model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve,
    args.num_bends,
    architecture_kwargs=architecture.kwargs,
)
model.cuda()
checkpoint = torch.load(args.ckpt)

new_state_dict = {}
for v_i, v in checkpoint['model_state'].items():
    new_key = v_i.replace("module.", "")  # 去掉 "module." 前缀
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)
# model.load_state_dict(checkpoint['model_state'])

criterion = F.cross_entropy
regularizer = curves.l2_regularizer(args.wd)

T = args.num_points
ts = np.linspace(args.start_t, args.end_t, T)

tr_loss = np.zeros(T)
tr_nll = np.zeros(T)
tr_acc = np.zeros(T)

tf_loss = np.zeros(T)
tf_nll = np.zeros(T)
tf_acc = np.zeros(T)

tf_loss_2 = np.zeros(T)
tf_nll_2 = np.zeros(T)
tf_acc_2 = np.zeros(T)

te_loss = np.zeros(T)
te_nll = np.zeros(T)
te_acc = np.zeros(T)

tr_err = np.zeros(T)
tf_err = np.zeros(T)
te_err = np.zeros(T)
tf_err_2 = np.zeros(T)

dl = np.zeros(T)

previous_weights = None

columns = ['t', 'Train loss', 'Train nll', 'Train Acc (%)', 'Forget nll', 'Forget Acc (%)', 'Test nll', 'Test Acc (%)']

time_start = time.time()
t = torch.FloatTensor([0.0]).cuda()
for i, t_value in enumerate(ts):
    time_start = time.time()
    t.data.fill_(t_value)
    weights = model.weights(t)
    if previous_weights is not None:
        dl[i] = np.sqrt(np.sum(np.square(weights - previous_weights)))
    previous_weights = weights.copy()

    time_start = time.time()

    utils.update_bn(loaders['train'], model, t=t)
    # utils.update_bn(loaders['train_retain'], model, t=t)
    time_start = time.time()

    tr_res = utils.test(loaders['train_retain'], model, criterion, regularizer, t=t)
    tf_res = utils.test(loaders['train_forget'], model, criterion, regularizer, t=t)
    if args.unlearn_type == 'class':
        tf_res_2 = utils.test(loaders['test_forget'], model, criterion, regularizer, t=t)
    else:
        tf_res_2 = {'loss': 0, 'nll': 0, 'accuracy': 0}
    te_res = utils.test(loaders['test_retain'], model, criterion, regularizer, t=t)


    tr_loss[i] = tr_res['loss']
    tr_nll[i] = tr_res['nll']
    tr_acc[i] = tr_res['accuracy']
    tr_err[i] = 100.0 - tr_acc[i]

    tf_loss[i] = tf_res['loss']
    tf_nll[i] = tf_res['nll']
    tf_acc[i] = tf_res['accuracy']
    tf_err[i] = 100.0 - tf_acc[i]

    tf_loss_2[i] = tf_res_2['loss']
    tf_nll_2[i] = tf_res_2['nll']
    tf_acc_2[i] = tf_res_2['accuracy']
    tf_err_2[i] = 100.0 - tf_acc_2[i]

    te_loss[i] = te_res['loss']
    te_nll[i] = te_res['nll']
    te_acc[i] = te_res['accuracy']
    te_err[i] = 100.0 - te_acc[i]

    values = [t, tr_loss[i], tr_nll[i], tr_acc[i], tf_nll[i], tf_acc[i], te_nll[i], te_acc[i]]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')
    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

time_iter = time_start-time.time()

def stats(values, dl):
    min = np.min(values)
    max = np.max(values)
    avg = np.mean(values)
    int = np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:])
    return min, max, avg, int


tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = stats(tr_loss, dl)
tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = stats(tr_nll, dl)
tr_err_min, tr_err_max, tr_err_avg, tr_err_int = stats(tr_err, dl)
tr_acc_min, tr_acc_max, tr_acc_avg, tr_acc_int = stats(tr_acc, dl)

tf_loss_min, tf_loss_max, tf_loss_avg, tf_loss_int = stats(tf_loss, dl)
tf_nll_min, tf_nll_max, tf_nll_avg, tf_nll_int = stats(tf_nll, dl)
tf_err_min, tf_err_max, tf_err_avg, tf_err_int = stats(tf_err, dl)
tf_acc_min, tf_acc_max, tf_acc_avg, tf_acc_int = stats(tf_acc, dl)

tf_loss_min_2, tf_loss_max_2, tf_loss_avg_2, tf_loss_int_2 = stats(tf_loss_2, dl)
tf_nll_min_2, tf_nll_max_2, tf_nll_avg_2, tf_nll_int_2 = stats(tf_nll_2, dl)
tf_err_min_2, tf_err_max_2, tf_err_avg_2, tf_err_int_2 = stats(tf_err_2, dl)
tf_acc_min_2, tf_acc_max_2, tf_acc_avg_2, tf_acc_int_2 = stats(tf_acc_2, dl)

te_loss_min, te_loss_max, te_loss_avg, te_loss_int = stats(te_loss, dl)
te_nll_min, te_nll_max, te_nll_avg, te_nll_int = stats(te_nll, dl)
te_err_min, te_err_max, te_err_avg, te_err_int = stats(te_err, dl)
te_acc_min, te_acc_max, te_acc_avg, te_acc_int = stats(te_acc, dl)

print('Length: %.2f' % np.sum(dl))
print(tabulate.tabulate([
        ['train loss', tr_loss[0], tr_loss[-1], tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int],
        ['train error (%)', tr_err[0], tr_err[-1], tr_err_min, tr_err_max, tr_err_avg, tr_err_int],
        ['test nll', te_nll[0], te_nll[-1], te_nll_min, te_nll_max, te_nll_avg, te_nll_int],
        ['test error (%)', te_err[0], te_err[-1], te_err_min, te_err_max, te_err_avg, te_err_int],
    ], [
        '', 'start', 'end', 'min', 'max', 'avg', 'int'
    ], tablefmt='simple', floatfmt='10.4f'))


np.savez(
    os.path.join(args.dir, file_name+'.npz'),
    ts=ts,
    dl=dl,

    tr_loss=tr_loss,
    tr_loss_min=tr_loss_min,
    tr_loss_max=tr_loss_max,
    tr_loss_avg=tr_loss_avg,
    tr_loss_int=tr_loss_int,
    tr_nll=tr_nll,
    tr_nll_min=tr_nll_min,
    tr_nll_max=tr_nll_max,
    tr_nll_avg=tr_nll_avg,
    tr_nll_int=tr_nll_int,
    tr_acc=tr_acc,
    tr_err=tr_err,
    tr_err_min=tr_err_min,
    tr_err_max=tr_err_max,
    tr_err_avg=tr_err_avg,
    tr_err_int=tr_err_int,

    tf_loss=tf_loss,
    tf_loss_min=tf_loss_min,
    tf_loss_max=tf_loss_max,
    tf_loss_avg=tf_loss_avg,
    tf_loss_int=tf_loss_int,
    tf_nll=tf_nll,
    tf_nll_min=tf_nll_min,
    tf_nll_max=tf_nll_max,
    tf_nll_avg=tf_nll_avg,
    tf_nll_int=tf_nll_int,
    tf_acc=tf_acc,
    tf_err=tf_err,
    tf_err_min=tf_err_min,
    tf_err_max=tf_err_max,
    tf_err_avg=tf_err_avg,
    tf_err_int=tf_err_int,

    tf_loss_2=tf_loss_2,
    tf_loss_min_2=tf_loss_min_2,
    tf_loss_max_2=tf_loss_max_2,
    tf_loss_avg_2=tf_loss_avg_2,
    tf_loss_int_2=tf_loss_int_2,
    tf_nll_2=tf_nll_2,
    tf_nll_min_2=tf_nll_min_2,
    tf_nll_max_2=tf_nll_max_2,
    tf_nll_avg_2=tf_nll_avg_2,
    tf_nll_int_2=tf_nll_int_2,
    tf_acc_2=tf_acc_2,
    tf_err_2=tf_err_2,
    tf_err_min_2=tf_err_min_2,
    tf_err_max_2=tf_err_max_2,
    tf_err_avg_2=tf_err_avg_2,
    tf_err_int_2=tf_err_int_2,

    te_loss=te_loss,
    te_loss_min=te_loss_min,
    te_loss_max=te_loss_max,
    te_loss_avg=te_loss_avg,
    te_loss_int=te_loss_int,
    te_nll=te_nll,
    te_nll_min=te_nll_min,
    te_nll_max=te_nll_max,
    te_nll_avg=te_nll_avg,
    te_nll_int=te_nll_int,
    te_acc=te_acc,
    te_err=te_err,
    te_err_min=te_err_min,
    te_err_max=te_err_max,
    te_err_avg=te_err_avg,
    te_err_int=te_err_int,

    flag_time=te_err_int,
)
