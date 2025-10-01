from functools import partial

import torch
from torch import nn

from robustbench import load_model

from models.mnist.small_cnn import SmallCNN

from utils.resnet import resnet18_normalized, ResNet18_Weights

from models.cifar10.preact_resnet import load_pretrained_models, load_preact_resnet

from models.cifar10.c11_c12 import load_c11, load_c12

mnist_smallcnn_ddn = {
    'name': 'MNIST_SmallCNN_ddn',
    'dataset': 'mnist'
}

mnist_smallcnn_trades = {
    'name': 'MNIST_SmallCNN_trades',
    'dataset': 'mnist'
}


def get_mnist_smallcnn(checkpoint: str) -> nn.Module:
    model = SmallCNN()
    state_dict = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


_local_mnist_models = {
    'smallcnn_ddn': partial(get_mnist_smallcnn, checkpoint='./models/mnist/mnist_smallcnn_robust_ddn.pth'),
    'smallcnn_trades': partial(get_mnist_smallcnn, checkpoint='./models/mnist/mnist_smallcnn_robust_trades.pth'),
}


def get_local_mnist_model(name: str, dataset: str) -> nn.Module:
    return _local_mnist_models[name]()


carmon_2019 = {
    'name': 'Carmon2019Unlabeled',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}
augustin_2020 = {
    'name': 'Augustin2020Adversarial',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'L2'
}

standard = {
    'name': 'Standard',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}
engstrom_2019 = {
    'name': 'Engstrom2019Robustness',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'L2'
}
gowal_2021 = {
    'name': 'Gowal2021Improving_70_16_ddpm_100m',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}
chen_2020 = {
    'name': 'Chen2020Adversarial',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}

xu_2023 = {
    'name': 'Xu2023Exploring_WRN-28-10',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}

addepalli2022 = {
    'name': 'Addepalli2022Efficient_RN18',
    'source': 'robustbench',
    'dataset': 'cifar10',
    'threat_model': 'Linf'
}


def load_robustbench_model(name: str, dataset: str, threat_model: str) -> nn.Module:
    model = load_model(model_name=name, dataset=dataset, threat_model=threat_model)
    return model


_local_cifar_models = {
    'pretr_L1': partial(load_pretrained_models),
    'preactresnet': partial(load_preact_resnet),
    'carmon2019': partial(load_robustbench_model, name=carmon_2019['name'], dataset=carmon_2019['dataset'],
                          threat_model=carmon_2019['threat_model']),
    'augustin2020': partial(load_robustbench_model, name=augustin_2020['name'], dataset=augustin_2020['dataset'],
                            threat_model=augustin_2020['threat_model']),
    'standard': partial(load_robustbench_model, name=standard['name'], dataset=standard['dataset'],
                        threat_model=standard['threat_model']),
    'engstrom2019': partial(load_robustbench_model, name=engstrom_2019['name'], dataset=engstrom_2019['dataset'],
                            threat_model=engstrom_2019['threat_model']),
    'gowal2021': partial(load_robustbench_model, name=gowal_2021['name'], dataset=gowal_2021['dataset'],
                         threat_model=gowal_2021['threat_model']),
    'chen2020': partial(load_robustbench_model, name=chen_2020['name'], dataset=chen_2020['dataset'],
                        threat_model=chen_2020['threat_model']),
    'xu2023': partial(load_robustbench_model, name=xu_2023['name'], dataset=xu_2023['dataset'],
                      threat_model=xu_2023['threat_model']),
    'addepalli2022': partial(load_robustbench_model, name=addepalli2022['name'], dataset=addepalli2022['dataset'],
                             threat_model=addepalli2022['threat_model']),
    'c11': partial(load_c11),
    'c12': partial(load_c12)
}


def get_local_cifar_model(name: str, dataset: str) -> nn.Module:
    return _local_cifar_models[name]()


def get_resnet18():
    model = resnet18_normalized(weights=ResNet18_Weights.DEFAULT,
                                normalize=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return model


wong_2020 = {
    'name': 'Wong2020Fast',
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

salman_2020R18 = {
    'name': 'Salman2020Do_R18',
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

engstrom_2019_imgnet = {
    'name': 'Engstrom2019Robustness',
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}

hendrycks2020many = {
    'name': 'Hendrycks2020Many',
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'corruptions'
}

debenedetti2020light_s = {
    'name': 'Debenedetti2022Light_XCiT-S12',
    'source': 'robustbench',
    'dataset': 'imagenet',
    'threat_model': 'Linf'
}
_local_imagenet_models = {
    'resnet18': partial(get_resnet18),
    'wong2020': partial(load_robustbench_model, name=wong_2020['name'], dataset=wong_2020['dataset'],
                        threat_model=wong_2020['threat_model']),
    'salman2020R18': partial(load_robustbench_model, name=salman_2020R18['name'], dataset=salman_2020R18['dataset'],
                             threat_model=salman_2020R18['threat_model']),
    'engstrom2019imgnet': partial(load_robustbench_model, name=engstrom_2019_imgnet['name'],
                                  dataset=engstrom_2019_imgnet['dataset'],
                                  threat_model=engstrom_2019_imgnet['threat_model']),
    'hendrycks2020many': partial(load_robustbench_model, name=hendrycks2020many['name'],
                                 dataset=hendrycks2020many['dataset'], threat_model=hendrycks2020many['threat_model']),
    'debenedetti2020light_s': partial(load_robustbench_model, name=debenedetti2020light_s['name'],
                                      dataset=debenedetti2020light_s['dataset'],
                                      threat_model=debenedetti2020light_s['threat_model']),
}


def get_local_model(name: str, dataset: str) -> nn.Module:
    print(f"Loading {name}")
    if dataset == 'cifar10':
        return _local_cifar_models[name]()
    elif dataset == 'mnist':
        return _local_mnist_models[name]()
    elif dataset == 'imagenet':
        return _local_imagenet_models[name]()
