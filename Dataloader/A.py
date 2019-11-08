import os
import numpy as np
import matplotlib.pyplot as plt
from .baseloader import BaseLoader
from PIL import Image


class ALoader(BaseLoader):
    """Custom dataset loader.
    Parameters
    ----------
      root: path to custom dataset, with train.txt and val.txt together.
        i.e., -----dataset
                |--train.txt
                |--val.txt
      n_classes: number of classes.
      split: choose subset of dataset, 'train','val' or 'test'.
      img_size: scale image to proper size.
      augmentations: whether to perform augmentation.
      ignore_index: ingore_index will be ignored in training phase and evaluation.
      class_weight: useful in unbalanced datasets.
      pretrained: whether to use pretrained models
    """
    # specify class_names if necessary
    class_names = np.array([
        'car',
        'bike',
        'person',
        'sky',
        'tree',
        'grass',
        'road',
        'sidewalk',
        'building',
        'fence',
        'sign',
        'pole',
        'block',
        # 'background',
    ])
    NUM_CLASS = 13

    def __init__(
        self,
        root,
        split="train",
        base_size=None,
        augmentations=None,
        ignore_index=255,
        class_weight=None,
        pretrained=False
        ):
        super(ALoader, self).__init__(root, split, base_size, augmentations, ignore_index, class_weight, pretrained)

        path = os.path.join(self.root, split + ".txt")
        with open(path, "r") as f:
            self.file_list = [file_name.rstrip() for file_name in f]

        print(f"Found {len(self.file_list)} {split} images")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]

        img = Image.open(os.path.join(self.root, 'VS_400x300', img_name + '.png'))
        ir = Image.open(os.path.join(self.root, 'IR_400x300', img_name + '.png'))
        lbl = Image.open(os.path.join(self.root, 'label_png', img_name + '.png'))

        if self.base_size:
            img = img.resize((self.base_size[1], self.base_size[0]), Image.BILINEAR)
            ir = ir.resize((self.base_size[1], self.base_size[0]), Image.BILINEAR)
            lbl = lbl.resize((self.base_size[1], self.base_size[0]), Image.NEAREST)

        if self.augmentations:
            img, ir, lbl = self.augmentations(img, ir, lbl)

        img, ir, lbl = self.transform(img, ir, lbl)
        return img, ir, lbl

    def getpalette(self):
        """for custom palette, if not specified, use pascal voc palette by default.
        """
        return np.asarray(
            [
                [0, 0, 142],
                [119, 11, 32],
                [220, 20, 60],
                [70, 130, 180],
                [107, 142, 35],
                [152, 251, 152],
                [128, 64, 128],
                [244, 35, 232],
                [70, 70, 70],
                [190, 153, 153],
                [220, 220, 0],
                [153, 153, 153],
                [180, 165, 180]
            ]
        )


# Test code
# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     root = ''
#     batch_size = 2
#     loader = CustomLoader(root=root, img_size=None)
#     test_loader = DataLoader(loader, batch_size=batch_size, shuffle=True)

#     palette = test_loader.dataset.getpalette()
#     fig, axes = plt.subplots(batch_size, 2, subplot_kw={'xticks': [], 'yticks': []})
#     fig.subplots_adjust(left=0.03, right=0.97, hspace=0.2, wspace=0.05)

#     for imgs, labels in test_loader:
#         imgs = imgs.numpy()
#         imgs = np.transpose(imgs, [0,2,3,1])
#         labels = labels.numpy()

#         for i in range(batch_size):
#             axes[i][0].imshow(imgs[i])

#             mask_unlabeled = labels[i] == -1
#             viz_unlabeled = (
#                 np.zeros((labels[i].shape[0], labels[i].shape[1], 3))
#             ).astype(np.uint8)

#             lbl_viz = palette[labels[i]]
#             lbl_viz[labels[i] == -1] = (0, 0, 0)
#             lbl_viz[mask_unlabeled] = viz_unlabeled[mask_unlabeled]

#             axes[i][1].imshow(lbl_viz.astype(np.uint8))
#         plt.show()
#         break

