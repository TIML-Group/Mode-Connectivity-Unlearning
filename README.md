# Exploring Nonlinear Pathway in Parameter Space for Machine Unlearning

[![preprint](https://img.shields.io/badge/arXiv-2505.10859-B31B1B)](https://arxiv.org/abs/2505.10859) 
[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue)](https://openreview.net/group?id=ICML.cc/2026)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repo for the paper [Exploring Nonlinear Pathway in Parameter Space for Machine Unlearning](https://arxiv.org/abs/2505.10859).



##  News 
- [x] [2026.04] 👏👏 Accepted by [**ICML 2026**](https://openreview.net/forum?id=SaUgHTz5Yo&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2026%2FConference%2FAuthors%23your-submissions)).
- [x] [2025.06] 👏👏 Accepted by [**MuGen @ ICML 2025**](https://openreview.net/forum?id=1PI440bNt5).
- [x] [2025.05] 🚀🚀 Release the [**paper**](https://arxiv.org/abs/2505.10859).

## Abstract
Machine Unlearning (MU) aims to remove the information of specific training data from a trained model, ensuring compliance with privacy regulations and user requests. While one line of existing MU methods relies on linear parameter updates via task arithmetic, they suffer from weight entanglement. In this work, we propose a novel MU framework called Mode Connectivity Unlearning (MCU) that leverages mode connectivity to find an unlearning pathway in a nonlinear manner. To further enhance performance and efficiency, we introduce a parameter mask strategy that not only improves unlearning effectiveness but also reduces computational overhead. Moreover, we propose an adaptive adjustment strategy for our unlearning penalty coefficient to adaptively balance forgetting quality and predictive performance during training, eliminating the need for empirical hyperparameter tuning. Unlike traditional MU methods that identify only a single unlearning model, MCU uncovers a spectrum of unlearning models along the pathway. Overall, MCU serves as a plug-and-play framework that seamlessly integrates with any existing MU methods, consistently improving unlearning efficacy. Extensive experiments on the image classification task demonstrate that MCU achieves superior performance.


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
├─ generate_mask.py  # Generate a parameter mask for SalUn
├─ generate_weight_mask.py  # Generate a parameter mask for MCU
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

To search a nonlinear pathway with our unlearning framework **MCU**, use the following commands:

Generate a parameter mask first:

```
python3 generate_weight_mask.py --dataset=CIFAR10 --forget_ratio=0.01 --unlearn_type=random --data_path=data --transform=ResNet --model=PreResNet110 --original_pth=[original ckpt]
```

Search the nonlinear pathway:

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

To plot effective unlearning region or get optimal $t$, use `plot_region.py` or `utils.find_optimal_t` with the results obtained from `eval_curve.py`.





## How to Cite

```
@article{shi2025mcu,
  title={MCU: Improving Machine Unlearning through Mode Connectivity},
  author={Shi, Yingdan and Wang, Ren},
  journal={arXiv preprint arXiv:2505.10859},
  year={2025}
}
```


## Acknowledgment
We acknowledge the publicly available codebase of [dnn-mode-connectivity](https://github.com/timgaripov/dnn-mode-connectivity).

