import argparse
import numpy as np
from PIL import Image
import scipy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm
import Models
from utils import visualize_segmentation, get_tile_image
from metrics import runningScore, averageMeter
from Dataloader import get_loader
from augmentations import RandomCrop, Compose
from option import argparser

def main():
    # parser = argparse.ArgumentParser(
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter
    # )
    # parser.add_argument('--model', type=str, default='multi-gnn1')
    # parser.add_argument('--model_file', type=str, default='/home/ecust/lx/Multimodal/logs/multi-gnn1_FS/model_best.pth.tar',help='Model path')
    # parser.add_argument('--dataset_type', type=str, default='b',help='type of dataset')
    # parser.add_argument('--dataset', type=str, default='/home/ecust/Datasets/数据库B(541)',help='path to dataset')
    # parser.add_argument('--base_size', type=tuple, default=(300, 300), help='resize images using bilinear interpolation')
    # parser.add_argument('--crop_size', type=tuple, default=None, help='crop images')
    # parser.add_argument('--n_classes', type=int, default=13, help='number of classes')
    # parser.add_argument('--pretrained', type=bool, default=True, help='should be set the same as train.py')
    # args = parser.parse_args()
    args = argparser()

    model_file = '/home/ecust/lx/Multimodal/logs/resnet_20190916_093026/model_best.pth.tar'
    root = args.dataset_root

    crop=None
    # crop = Compose([RandomCrop(args.crop_size)])
    loader = get_loader(args.dataset)
    val_loader = DataLoader(
        loader(root, split='val', base_size=args.base_size, augmentations=crop),
        batch_size=1, shuffle=False, num_workers=4)
    args.n_classes = loader.NUM_CLASS

    model = Models.model_loader(args.model, args.n_classes,
                                backbone=args.backbone, norm_layer=nn.BatchNorm2d,
                                multi_grid=args.multi_grid,
                                multi_dilation=args.multi_dilation)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print('==> Loading {} model file: {}'.format(model.__class__.__name__, model_file))

    model_data = torch.load(model_file)

    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating with {} dataset'.format(args.dataset))
    visualizations = []
    metrics = runningScore(args.n_classes)

    i = 0
    for rgb, ir, target in tqdm.tqdm(val_loader, total=len(val_loader), ncols=80, leave=False):
        rgb, ir, target = rgb.to(device), ir.to(device), target.to(device)
        score = model(rgb, ir)
        # score = model(ir)

        rgbs = rgb.data.cpu()
        irs = ir.data.cpu()
        lbl_pred = score[0].data.max(1)[1].cpu().numpy()
        lbl_true = target.data.cpu()
        for rgb, ir, lt, lp in zip(rgbs, irs, lbl_true, lbl_pred):
            rgb, ir, lt = val_loader.dataset.untransform(rgb, ir, lt)
            metrics.update(lt, lp)

            i += 1
            if i % 5 == 0:
                if len(visualizations) < 9:
                    viz = visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=rgb, ir=ir,
                        n_classes=args.n_classes, dataloader=val_loader)
                    visualizations.append(viz)

    acc, acc_cls, mean_iu, fwavacc, cls_iu = metrics.get_scores()
    print('''
Accuracy:       {0:.2f}
Accuracy Class: {1:.2f}
Mean IoU:       {2:.2f}
FWAV Accuracy:  {3:.2f}'''.format(acc * 100,
                                  acc_cls * 100,
                                  mean_iu * 100,
                                  fwavacc * 100) + '\n')

    class_name = val_loader.dataset.class_names
    if class_name is not None:
        for index, value in enumerate(cls_iu.values()):
            offset = 20 - len(class_name[index])
            print(class_name[index] + ' ' * offset + f'{value * 100:>.2f}')
    else:
        print("\nyou don't specify class_names, use number instead")
        for key, value in cls_iu.items():
            print(key, f'{value * 100:>.2f}')

    viz = get_tile_image(visualizations)
    # img = Image.fromarray(viz)
    # img.save('viz_evaluate.png')
    scipy.misc.imsave('viz_evaluate.png', viz)

if __name__ == '__main__':
    main()