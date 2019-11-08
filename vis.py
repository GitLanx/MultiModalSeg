import argparse
import numpy as np
from PIL import Image
import scipy
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision
import tqdm
import Models
from utils import visualize_segmentation, get_tile_image
from metrics import runningScore, averageMeter
from Dataloader import get_loader
from augmentations import RandomCrop, Compose

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='fcn8s')
    parser.add_argument('--model_file', type=str, default='/home/ecust/lx/Multimodal/logs/fcn8s_VS_B_0.001/model_best.pth.tar',help='Model path')
    parser.add_argument('--dataset_type', type=str, default='b',help='type of dataset')
    parser.add_argument('--dataset', type=str, default='/home/ecust/Datasets/数据库B(541)',help='path to dataset')
    parser.add_argument('--img_size', type=tuple, default=(320, 416), help='resize images using bilinear interpolation')
    parser.add_argument('--crop_size', type=tuple, default=None, help='crop images')
    parser.add_argument('--n_classes', type=int, default=13, help='number of classes')
    parser.add_argument('--pretrained', type=bool, default=True, help='should be set the same as train.py')
    args = parser.parse_args()

    model_file = args.model_file
    root = args.dataset
    n_classes = args.n_classes
    writer = SummaryWriter()

    crop=None
    # crop = Compose([RandomCrop(args.crop_size)])
    loader = get_loader(args.dataset_type)
    val_loader = DataLoader(
        loader(root, n_classes=n_classes, split='val', img_size=args.img_size, augmentations=crop, pretrained=args.pretrained),
        batch_size=1, shuffle=False, num_workers=4)

    model, _, _ = Models.model_loader(args.model, n_classes, resume=None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print('==> Loading {} model file: {}'.format(model.__class__.__name__, model_file))

    model_data = torch.load(model_file)

    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()

    print('==> Evaluating with {} dataset'.format(args.dataset_type))

    for rgb, ir, target in tqdm.tqdm(val_loader, total=len(val_loader), ncols=80, leave=False):
        rgb, ir, target = rgb.to(device), ir.to(device), target.to(device)
        x = rgb

        grid = torchvision.utils.make_grid(x, normalize=True)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, (ir))
        # score = model(rgb, ir)
        # score = model(ir)
        for i, (name, param) in enumerate(model.named_parameters()):
            writer.add_histogram(name, param, 0)

        for name, layer in model._modules.items():
        
            # if 'ir' in name and 'feature' in name:
            if 'feature' in name or 'fc' in name or 'score_fr' in name:
                x = layer(x)

                x1 = x.transpose(0, 1)
                img_grid = torchvision.utils.make_grid(x1, normalize=True, scale_each=True)  # normalize进行归一化处理
                writer.add_image(f'{name}_feature_maps', img_grid, global_step=0)
        break

if __name__ == '__main__':
    main()