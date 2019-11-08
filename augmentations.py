import random
from PIL import Image, ImageOps


class Compose:
    def __init__(self, augmentations):
        self.augmentations = augmentations
    
    def __call__(self, imgs, irs, lbls):
        assert imgs.size == lbls.size
        for aug in self.augmentations:
            imgs, irs, lbls = aug(imgs, irs, lbls)
        
        return imgs, irs, lbls


class RandomFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, ir, label):
        if random.random() < self.prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            ir = ir.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return image, ir, label


class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    @staticmethod
    def get_params(img, output_size):
        w, h = img.size
        th =  tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, image, ir, label):
        if image.size[0] < self.crop_size:
            image = ImageOps.expand(image, (self.crop_size - image.size[0], 0), fill=0)
            ir = ImageOps.expand(ir, (self.crop_size - ir.size[0], 0), fill=0)
            label = ImageOps.expand(label, (self.crop_size - label.size[0], 0), fill=255)
        if image.size[1] < self.crop_size:
            image = ImageOps.expand(image, (0, self.crop_size - image.size[1]), fill=0)
            ir = ImageOps.expand(ir, (0, self.crop_size - ir.size[1]), fill=0)
            label = ImageOps.expand(label, (0, self.crop_size - label.size[1]), fill=255)

        i, j, h, w = self.get_params(image, self.crop_size)
        image = image.crop((j, i, j + w, i + h))
        ir = ir.crop((j, i, j + w, i + h))
        label = label.crop((j, i, j + w, i + h))

        return image, ir, label


class RandomScale:
    def __init__(self, scale_size):
        self.scale_size = scale_size

    def __call__(self, image, ir, label):
        w, h = image.size
        scale = random.uniform(self.scale_size[0], self.scale_size[1])
        ow, oh = int(w * scale), int(h * scale)

        image = image.resize((ow, oh), Image.BILINEAR)
        ir = ir.resize((ow, oh), Image.BILINEAR)
        label = label.resize((ow, oh), Image.NEAREST)

        return image, ir, label


def get_augmentations(args):
    augs = []
    if args.flip:
        augs.append(RandomFlip())
    if args.scale_size:
        augs.append(RandomScale(args.scale_size))
    if args.crop_size:
        augs.append(RandomCrop(args.crop_size))

    if augs == []:
        return None
    print('Using augmentations: ', end=' ')
    for x in augs:
        print(x.__class__.__name__, end=' ')
    print('\n')

    return Compose(augs)