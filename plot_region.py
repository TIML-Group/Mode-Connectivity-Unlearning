import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import interp1d
import os
from arg_parser import parse_args
import yaml
import argparse
from itertools import groupby


FLAG = 'search'  # 'eval'

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
args = parse_args()

line_color = ['#3579B5', '#EE8132', '#326F20', '#cccccc']
base_dir = './end_ablation/randomlabel/'

retain_list = []
forget_list = []
for file_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, file_name)

    file_name = folder_path
    data = np.load(file_name)

    t = data['ts']
    retain_acc = data['tr_acc']
    forget_acc = data['tf_acc']

    retain_acc = np.array(retain_acc)
    forget_acc = np.array(forget_acc)

    if args.unlearn_type == 'random':
        retrain_base = config[args.dataset][args.unlearn_type]['train']
        forget_base = config[args.dataset][args.unlearn_type]['val']
    elif args.unlearn_type == 'class':
        retrain_base = config[args.dataset][args.unlearn_type]['train']
        forget_base = 0.00


    baseline_retain = [retrain_base for i in range(len(retain_acc))]
    baseline_forget = [forget_base for i in range(len(retain_acc))]


    plt.figure(figsize=(5, 3.8))
    print('style.available', plt.style.available)
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({"font.family": 'Times New Roman', 'font.size': 27,
                         'text.color': '#000000',
                         'axes.labelcolor': '#000000',
                         'xtick.color': '#000000',
                         'ytick.color': '#000000',
                         })
    rc('mathtext', fontset='stix')

    plt.plot(t, baseline_retain, linestyle='dotted', color=line_color[0], linewidth=3, alpha=0.6)
    plt.plot(t, baseline_forget, linestyle='dotted', color=line_color[1], linewidth=3, alpha=0.6)


    plt.plot(t, forget_acc, color=line_color[1], marker='o', markerfacecolor='none', label=r'$D_f$', linewidth=3, markersize=6, alpha=0.6)
    plt.plot(t, retain_acc, color=line_color[0], marker='o', markerfacecolor='none', label=r'$D_r$', linewidth=3, markersize=6, alpha=0.6)


    plt.subplots_adjust(left=0.21, right=0.99, top=0.99, bottom=0.19)


    plt.xlabel('t')
    plt.ylabel('Accuracy(%)')
    plt.ylim(79, 101)
    plt.yticks(range(80,101,4))
    plt.xlim(-0.05, 1.05)
    plt.xticks([0,0.25,0.5,0.75,1.0], ['0.0','0.25','0.5','0.75','1.0'])

    retain_gap = retain_acc-baseline_retain
    forget_gap = forget_acc-baseline_forget
    avg_gap = (abs(retain_gap)+abs(forget_gap))/2

    # 1. cubic interpolate
    t_dense = np.linspace(0.0, 1.0, 1000)

    interpolator = interp1d(t, retain_acc, kind='cubic')
    retain_acc_dense = interpolator(t_dense)
    interpolator = interp1d(t, forget_acc, kind='cubic')
    forget_acc_dense = interpolator(t_dense)

    retain_gap_ = retain_acc_dense-baseline_retain[-1]
    forget_gap_ = forget_acc_dense-baseline_forget[-1]
    avg_gap_ = (abs(retain_gap_)+abs(forget_gap_))/2

    # 2. region compare
    below_threshold = avg_gap_ < avg_gap_[-1]
    t_below = t_dense[below_threshold]

    highlight_regions = []
    for k, g in groupby(enumerate(t_below), lambda ix: ix[0] - np.where(t_dense == ix[1])[0][0]):
        group = list(g)
        start = group[0][1]
        end = group[-1][1]
        highlight_regions.append((start, end))

    # 3. plot region
    for start, end in highlight_regions:
        plt.axvspan(start, end, color=line_color[3], alpha=0.6, label="Effective Region" if start == highlight_regions[0][0] else None)

    file_name = file_name.split('/')[-1][:-4]
    plt.savefig('./figures/'+file_name+'.pdf')