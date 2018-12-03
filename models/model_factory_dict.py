from densenet import (DenseNet121, DenseNet169, DenseNet201, DenseNet161,
                      densenet_cifar)
from dpn import DPN26, DPN92
from googlenet import GoogLeNet
from lenet import LeNet
from mobilenet import MobileNet
from mobilenetv2 import MobileNetV2
from pnasnet import PNASNetA, PNASNetB
from preact_resnet import (PreActResNet18, PreActResNet34, PreActResNet50,
                           PreActResNet101, PreActResNet152)
from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from resnext import (ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d,
                     ResNeXt29_32x4d)
from senet import SENet18
from shufflenet import ShuffleNetG2, ShuffleNetG3
from shufflenetv2 import ShuffleNetV2
from vgg import VGG


def model_factory(model_name, **params):
    model_dict = {
        'densenet121': DenseNet121,
        'densenet169': DenseNet169,
        'densenet201': DenseNet201,
        'densenet161': DenseNet161,
        'densenet-cifar': densenet_cifar,
        'dual-path-net-26': DPN26,
        'dual-path-net-92': DPN92,
        'googlenet': GoogLeNet,
        'lenet': LeNet,
        'mobilenet': MobileNet,
        'mobilenetv2': MobileNetV2,
        'pnasneta': PNASNetA,
        'pnasnetb': PNASNetB,
        'preact-resnet18': PreActResNet18,
        'preact-resnet34': PreActResNet34,
        'preact-resnet50': PreActResNet50,
        'preact-resnet101': PreActResNet101,
        'preact-resnet152': PreActResNet152,
        'resnet18': ResNet18,
        'resnet34': ResNet34,
        'resnet50': ResNet50,
        'resnet101': ResNet101,
        'resnet152': ResNet152,
        'resnext29_2x64d': ResNeXt29_2x64d,
        'resnext29_4x64d': ResNeXt29_4x64d,
        'resnext29_8x64d': ResNeXt29_8x64d,
        'resnext29_32x64d': ResNeXt29_32x4d,
        'senet18': SENet18,
        'shufflenetg2': ShuffleNetG2,
        'shufflenetg3': ShuffleNetG3,
        'shufflenetv2_0.5': ShuffleNetV2,
        'shufflenetv2_1.0': ShuffleNetV2,
        'shufflenetv2_1.5': ShuffleNetV2,
        'shufflenetv2_2.0': ShuffleNetV2,
        'vgg11': VGG,
        'vgg13': VGG,
        'vgg16': VGG,
        'vgg19': VGG,
    }

    if 'vgg' in model_name:
        return model_dict[model_name](model_name)
    elif 'shufflenetv2' in model_name:
        return model_dict[model_name](float(model_name[-3:]))
    elif model_name in model_dict.keys():
        return model_dict[model_name]()
    else:
        raise AttributeError('Model doesn\'t exist')
