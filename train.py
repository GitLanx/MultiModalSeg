import os
import os.path as osp
import random
import yaml
import datetime
import torch
import torch.nn as nn
import numpy as np
from Dataloader import get_loader
from torch.utils.data import DataLoader
from Models import model_loader
from trainer import Trainer
from utils import get_scheduler
from optimizer import get_optimizer
from augmentations import get_augmentations
from option import argparser


here = osp.dirname(osp.abspath(__file__))


def main():
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    args = argparser()

    now = datetime.datetime.now()
    args.out = osp.join(here, 'logs', args.model + '_' + now.strftime('%Y%m%d_%H%M%S'))

    if not osp.exists(args.out):
        os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Start training {args.model} using {device.type}\n')

    # 1. dataset

    root = args.dataset_root
    loader = get_loader(args.dataset)

    augmentations = get_augmentations(args)

    train_loader = DataLoader(
        loader(root, split='train', base_size=args.base_size, augmentations=augmentations),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(
        loader(root, split='val', base_size=args.base_size),
        batch_size=1, shuffle=False, num_workers=args.workers)
    args.n_classes = loader.NUM_CLASS

    # 2. model
    model = model_loader(args.model, args.n_classes,
                         backbone=args.backbone, norm_layer=nn.BatchNorm2d,
                         multi_grid=args.multi_grid,
                         multi_dilation=args.multi_dilation)
    model = model.to(device)
    print(model)
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        checkpoint = None

    # 3. optimizer
    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    # optim = torch.optim.SGD(
    #     [{'params': model.get_parameters(key='1x'), 'lr': args.lr},
    #      {'params': model.get_parameters(key='10x'), 'lr': args.lr * 10}],
    #      momentum=args.momentum,
    #      weight_decay=args.weight_decay
    # )
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    scheduler = get_scheduler(optim, args)

    # 4. train
    trainer = Trainer(
        device=device,
        model=model,
        optimizer=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        epochs=args.epochs,
        n_classes=args.n_classes,
        val_epoch=args.val_epoch,
    )
    trainer.epoch = start_epoch
    trainer.train()


if __name__ == '__main__':
    main()
