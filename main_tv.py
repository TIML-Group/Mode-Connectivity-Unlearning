from taskvector.task_vectors import TaskVector
import data
import os
import time
import torch.nn.parallel


import numpy as np
from arg_parser import parse_args
import re
from datetime import datetime
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split
import utils
import models


if __name__ == '__main__':

    args = parse_args()

    dir = f'{args.dir}{args.unlearn_method}/'
    print('\n=====save_dir=====', dir, '\n')
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

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

    time_s = time.time()

    # Config
    pretrained_checkpoint = args.init_start
    finetuned_checkpoint = args.init_end

    # Create the task vector
    task_vector = TaskVector(pretrained_checkpoint, finetuned_checkpoint)
    # Negate the task vector
    neg_task_vector = -task_vector
    # Apply the task vector
    image_encoder = neg_task_vector.apply_to(pretrained_checkpoint, scaling_coef=args.coef)

    # model
    architecture = getattr(models, args.model)
    model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint_original = torch.load(image_encoder)
    model.load_state_dict(image_encoder)
    model.to(device)

    training_time_sum = time.time() - time_s

    # Evaluate
    test_forget_acc = 0
    train_retain_acc = utils.evaluate_acc(model, loaders['train_retain'], device)
    train_forget_acc = utils.evaluate_acc(model, loaders['train_forget'], device)
    test_retain_acc = utils.evaluate_acc(model, loaders['test_retain'], device)
    if args.unlearn_type == 'class':
        test_forget_acc = utils.evaluate_acc(model, loaders['test_forget'], device)

    # save
    dict = {'rr_acc' : train_retain_acc,
             'rf_acc' : train_forget_acc,
             'tr_acc' : test_retain_acc,
             'tf_acc' : test_forget_acc,
             'RTE' : training_time_sum}
    print(dict)

    utils.save_checkpoint(
        dir,
        args.seed,
        name=f'checkpoint_{args.coef}',
        model_state=model.state_dict(),
    )
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y_%m_%d-%H_%M_%S')
    np.savez(os.path.join(dir, f'{args.seed}-{formatted_time}.npz'),
             rr_acc = train_retain_acc,
             rf_acc = train_forget_acc,
             tr_acc = test_retain_acc,
             tf_acc = test_forget_acc,
             RTE=training_time_sum,
             )
