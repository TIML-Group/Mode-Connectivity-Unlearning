# MCU: Improving Machine Unlearning through Mode Connectivity

[![preprint](https://img.shields.io/badge/arXiv-2505.10859-B31B1B)](https://arxiv.org/abs/2505.10859) 
[![MUGen @ ICML 2025](https://img.shields.io/badge/MUGen@ICML-2025-blue)](https://openreview.net/forum?id=1PI440bNt5)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repo for the paper [MCU: Improving Machine Unlearning through Mode Connectivity](https://arxiv.org/abs/2505.10859).



##  News 
- [x] [2025.06] 👏👏 Accepted by [**MuGen @ ICML 2025**](https://openreview.net/forum?id=1PI440bNt5).
- [x] [2025.05] 🚀🚀 Release the [**paper**](https://arxiv.org/abs/2505.10859).


## File Tree

Project file structure and description:

```
Mode-Connectivity-Unlearning
├─ README.md
├─ requirements.txt
├─ evaluation
│    ├─ SVC_MIA.py
├─ models	# package of models
│    ├─ __init__.py
│    ├─ preresnet.py
│    ├─ vgg.py
│    ├─ vit.py
├─ arg_parser.py
├─ curves.py
├─ data.py
├─ eval_curve.py
├─ plot_region.py
├─ generate_weight_mask.py
├─ train.py
├─ unlern.py
├─ utils.py
├─ main_tv.py
├─ task_vector.py
└─ requirements.txt
```

## Setup

Installation requirements are described in `requirements.txt`.

- Use pip:

  ```
  pip install -r requirements.txt
  ```

- Use anaconda:

  ```
  conda install --file requirements.txt
  ```

## Getting Started

Get an original model:

```
python3 train.py --dir=./ckpt/ --dataset=CIFAR10 --unlearn_method=baseline --unlearn_type=random --data_path=data --transform=ResNet --model=PreResNet110 --epochs=200
```

You may need to record the original model's training accuracy and validation accuracy. To use the implemented logging, you'll need a `wandb.ai` account. Alternatively, you can replace it with any logger of your preference.

To get an unlearning model with one of the existing unlearning methods, use the following command:

```
python3 train.py --dir=./ckpt/ --dataset=CIFAR10 --unlearn_method=retrain --unlearn_type=random --data_path=data --lr=0.01 --forget_ratio=0.1 --transform=ResNet --model=PreResNet110 --epochs=200 --seed=42
```

To search a nonlinear pathway with our unlearning framework **MCU**, use the following command:

```
python3 train.py --dir=[save dir] \
                 --dataset=CIFAR10 \
                 --unlearn_method=[curve/dynamic] \
                 --unlearn_type=random \
                 --forget_ratio=0.1 \
                 --milestones=11 \
                 --data_path=data \
                 --transform=ResNet \
                 --model=PreResNet110 \
                 --epochs=10 \
                 --lr=0.01 \
                 --curve=Bezier \
                 --num_bends=3 \
                 --init_start=[original ckpt] \
                 --init_end=[pre-unlearning ckpt] \
                 --mask_path=[parameter mask file] \
                 --retain_ratio=0.5 \
                 --beta=0.2 \
                 --fix_start \
                 --fix_end \
                 --seed=42
```

After getting a nonlinear pathway, use `eval_curve.py` to sample a few points on the pathway:

```
python3 eval_curve.py --dir=[save dir] \
                 --dataset=CIFAR10 \
                 --unlearn_type=random \
                 --forget_ratio=0.1 \
                 --data_path=data \
                 --transform=ResNet \
                 --model=PreResNet110 \
                 --curve=Bezier \
                 --num_bends=3 \
                 --start_t=0 \
                 --end_t=1 \
                 --num_points=20 \
                 --ckpt=[curve ckpt]
```

To find the optimal model and effective unlearning region, use `plot_region.py`.


## How to Cite

```
@article{shi2025mcu,
  title={MCU: Improving Machine Unlearning through Mode Connectivity},
  author={Shi, Yingdan and Wang, Ren},
  journal={arXiv preprint arXiv:2505.10859},
  year={2025}
}
```



