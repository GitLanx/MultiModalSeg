import argparse

def argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, default='resnet',
                        help='model name')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='b',
                        help='choose which dataset to use')
    parser.add_argument('--dataset_root', type=str, default='/home/ecust/Datasets/数据库B(541)',
                        help='path to dataset')
    parser.add_argument('--workers', type=int, default=2,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base_size', type=tuple, default=(300, 400),
                        help='resize images to proper size (h, w)')
    parser.add_argument('--crop_size', type=int, default=225,
                        help='crop sizes of images')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='total epochs')
    parser.add_argument('--val_epoch', type=int, default=5,
                        help='validation interval')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training')

    # optimizer params
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr_policy', type=str, default='poly',
                        help='learning rate policy')
    parser.add_argument('--lr_decay_step', type=float, default=10,
                        help='step size for step learning policy (available when step policy is used')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for sgd')

    # augmentation option
    parser.add_argument('--scale_size', type=tuple, default=(0.5, 2),
                        help='random scale images within range')
    parser.add_argument('--flip', type=bool, default=True, help='whether to use horizontal flip')

    # checkpoint
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint')
    parser.add_argument('--log-root', type=str,
                        default='./logs', help='set a log path folder')

    # multi grid dilation option
    parser.add_argument("--multi-grid", action="store_true", default=False,
                        help="use multi grid dilation policy")
    parser.add_argument('--multi-dilation', nargs='+', type=int, default=(4, 8, 16),
                        help="multi grid dilation list")

    args = parser.parse_args()
    return args