import numpy as np
import torch
from torch.utils import data
from torchvision import transforms


class BaseLoader(data.Dataset):
    # specify class_name if available
    class_name = None

    def __init__(self,
                 root,
                 split='train',
                 base_size=None,
                 augmentations=None,
                 ignore_index=None,
                 class_weight=None):

        self.root = root
        self.split = split
        self.base_size = base_size
        self.augmentations = augmentations
        self.ignore_index = ignore_index
        self.class_weight = class_weight

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.ir_mean = torch.tensor([0.485, 0.485, 0.485])
        self.ir_std = torch.tensor([0.229, 0.229, 0.229])
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean.tolist(), self.std.tolist())
        ])
        self.ir_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.ir_mean.tolist(), self.ir_std.tolist())
        ])
        self.untf = transforms.Compose([
            transforms.Normalize((-self.mean / self.std).tolist(),
                                (1.0 / self.std).tolist())
        ])
        self.ir_untf = transforms.Compose([
            transforms.Normalize((-self.ir_mean / self.ir_std).tolist(),
                                (1.0 / self.ir_std).tolist())
        ])

    def __getitem__(self, index):
        return NotImplementedError

    @property
    def num_class(self):
        return self.NUM_CLASS

    def transform(self, img, ir, lbl):
        img = self.tf(img)
        # ir = transforms.ToTensor()(ir)
        ir = self.ir_tf(ir)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self.ignore_index:
            lbl[lbl == self.ignore_index] = -1
        lbl = torch.from_numpy(lbl).long()
        return img, ir, lbl

    def untransform(self, img, ir, lbl):
        img = self.untf(img)
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = img * 255
        img = img.astype(np.uint8)
        ir = self.ir_untf(ir)
        ir = ir.numpy()
        ir = ir.transpose(1, 2, 0)
        ir = ir * 255
        ir = ir.astype(np.uint8)
        lbl = lbl.numpy()
        return img, ir, lbl

    def getpalette(self):
        return NotImplementedError
