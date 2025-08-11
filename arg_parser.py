import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='DNN curve training')
    parser.add_argument('--dir', type=str, default='/tmp/curve/', metavar='DIR',
                        help='training directory (default: /tmp/curve/)')

    # unlearning hyperparameters
    parser.add_argument('--unlearn_method', type=str, default='baseline', metavar='UNLEARN',
                        help='unlearn method (default: baseline)')
    parser.add_argument('--unlearn_type', type=str, default='random', metavar='UNLEARN',
                        help='unlearn type (default: random)')
    parser.add_argument('--original_pth', type=str, metavar='UNLEARN',
                        help='original model path (default: random)')
    parser.add_argument('--forget_ratio', type=float, default=0.1, metavar='UNLEARN',
                        help='forget ratio (default: 0.1)')
    parser.add_argument('--milestones', type=str, default=None)

    parser.add_argument("--mask_path", default=None, type=str, help="the path of saliency map")
    parser.add_argument('--mask_ratio', type=float, default=1.0, metavar='UNLEARN',
                        help='mask ratio. The selected weights will change. (default: 0.1)')
    parser.add_argument('--kr', type=float, default=0.0, metavar='UNLEARN',
                        help='kr for retain grad. The selected weights will change. (default: 0.5)')
    parser.add_argument('--retain_ratio', type=float, default=1.0, metavar='UNLEARN',
                        help='retain ratio. The selected retain data for training. (default: 1.0)')
    parser.add_argument('--coef', type=float, default=0.9, metavar='UNLEARN',
                        help='coef. Coefficient for NegTV. (default: 0.9)')


    parser.add_argument('--start_t', type=float, default=0.0, metavar='UNLEARN',
                        help='start_t. The selected retain data for training. (default: 0.0)')
    parser.add_argument('--end_t', type=float, default=1.0, metavar='UNLEARN',
                        help='end_t. The selected retain data for training. (default: 1.0)')
    parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                        help='checkpoint to eval (default: None)')
    parser.add_argument('--num_points', type=int, default=20, metavar='N',
                        help='number of points on the curve (default: 61)')


    parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                        help='dataset name (default: CIFAR10)')
    parser.add_argument('--use_test', action='store_true',
                        help='switches between validation and test set (default: validation)')
    parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                        help='transform name (default: VGG)')
    parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                        help='path to datasets location (default: None)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                        help='model name (default: None)')

    parser.add_argument('--curve', type=str, default=None, metavar='CURVE',
                        help='curve type to use (default: None)')
    parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                        help='number of curve bends (default: 3)')
    parser.add_argument('--init_start', type=str, default=None, metavar='CKPT',
                        help='checkpoint to init start point (default: None)')
    parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                        help='fix start point (default: off)')
    parser.add_argument('--init_end', type=str, default=None, metavar='CKPT',
                        help='checkpoint to init end point (default: None)')
    parser.add_argument('--fix_end', dest='fix_end', action='store_true',
                        help='fix end point (default: off)')
    parser.set_defaults(init_linear=True)
    parser.add_argument('--init_linear_off', dest='init_linear', action='store_false',
                        help='turns off linear initialization of intermediate points (default: on)')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='checkpoint to resume training from (default: None)')


    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--save_freq', type=int, default=5, metavar='N',
                        help='save frequency (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--beta', type=float, default=0.15, metavar='M',
                        help='beta for NegGrad+')

    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')

    args = parser.parse_args()

    return args
