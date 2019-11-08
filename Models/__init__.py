from .FCN import FCN8sAtOnce
from .multi import FCN8sAtOnceMulti
from .multi_gnn import FCN8sAtOnceMultiGnn
from .multi_gnn1 import DeepLabASPPResNetGnn
from .multi_gnn2 import FCN8sAtOnceMultiGnn2
from .multi_gnn_u import DeepLabResNetGnnUnet
from .DeepLab_v1 import DeepLabLargeFOV
from .DeepLab_v2 import DeepLabASPPVGG, DeepLabASPPResNet
from .DeepLab_v3 import DeepLabV3
from .DeepLab_v3plus import DeepLabV3Plus
from .en_de import EncoderDecoder
from .RedNet import RedNet
from .ACNet import ACNet
from .RTFNet import RTFNet
from .resnet_baseline import ResNetBaseLine
from .base import BaseNet
from .gcnet import GCNet

def model_loader(model_name, n_classes, **kargs):
    models = {
        'fcn8s': FCN8sAtOnce,
        'fcn8smulti': FCN8sAtOnceMulti,
        'fcn8smulti-gnn': FCN8sAtOnceMultiGnn,
        'fcn8smulti-gnn2': FCN8sAtOnceMultiGnn2,
        'multi-gnn1': DeepLabASPPResNetGnn,
        'multi-gnn-u': DeepLabResNetGnnUnet,
        'en-de': EncoderDecoder,
        'deeplab-largefov': DeepLabLargeFOV,
        'deeplab-aspp-vgg': DeepLabASPPVGG,
        'deeplab-aspp-resnet': DeepLabASPPResNet,
        'deeplab-v3': DeepLab_v3,
        'deeplab-v3+': DeepLab_v3plus,
        'rednet': RedNet,
        'acnet': ACNet,
        'rtfnet': RTFNet,
        'resnet': ResNetBaseLine,
        'gcnet': GCNet,
    }
    return models[model_name.lower()](n_classes, **kargs)
