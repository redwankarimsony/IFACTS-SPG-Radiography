from torch.nn import Parameter
import math
import torch
import torch.nn as nn

from torchvision.models import densenet121, DenseNet121_Weights
from torchvision.models import densenet161, DenseNet161_Weights
from torchvision.models import densenet169, DenseNet169_Weights
from torchvision.models import densenet201, DenseNet201_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet


def get_fr_model(backbone: str, head: str, embedding_size: int, class_num: int, m: float, t_alpha: float, h: float,
                 s: float):
    """Builds the model based on the backbone and head type

    Args:
        backbone (str): The backbone type
        head (str): The head type
        embedding_size (int): The embedding size
        class_num (int): The number of classes
        m (float): The margin value
        t_alpha (float): The alpha value for the batch mean and std
        h (float): The h value for the margin scaler
        s (float): The s value for the cosine scaler

    Returns:
        [type]: [description]
    """

    model = get_backbone(backbone)
    head = build_head(head, embedding_size, class_num, m, t_alpha, h, s)
    return model, head


def get_backbone(cfg):
    if cfg.model_arch.startswith("resnet"):
        if cfg.model_arch == "resnet34":
            model = resnet34(weights=ResNet34_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "resnet101":
            model = resnet101(weights=ResNet101_Weights.DEFAULT, progress=True)

            # Modify the model head
            model.fc = nn.Linear(model.fc.in_features, cfg.embedding_size, bias=True)

        else:
            model = None
            raise ValueError(f"No ResNet model found with key {cfg.model_arch}")

        return model

    elif cfg.model_arch.startswith("densenet"):
        if cfg.model_arch == "densenet121":
            model = densenet121(DenseNet121_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "densenet161":
            model = densenet161(DenseNet161_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "densenet169":
            model = densenet169(DenseNet169_Weights.IMAGENET1K_V1, progress=True)
        elif cfg.model_arch == "densenet201":
            model = densenet201(DenseNet201_Weights.IMAGENET1K_V1, progress=True)
        else:
            model = None
            raise ValueError(f"No model found with key {cfg.model_arch}")

        # Modify the model head
        model.classifier = nn.Linear(model.classifier.in_features, cfg.embedding_size, bias=True)
        return model

    elif cfg.model_arch.startswith("efficientnet"):
        if cfg.model_arch == "efficientnet_b0":
            model = efficientnet.efficientnet_b0(efficientnet.EfficientNet_B0_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "efficientnet_b1":
            model = efficientnet.efficientnet_b1(efficientnet.EfficientNet_B1_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "efficientnet_b2":
            model = efficientnet.efficientnet_b2(efficientnet.EfficientNet_B2_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "efficientnet_b3":
            model = efficientnet.efficientnet_b3(efficientnet.EfficientNet_B3_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "efficientnet_b4":
            model = efficientnet.efficientnet_b4(efficientnet.EfficientNet_B4_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "efficientnet_b5":
            model = efficientnet.efficientnet_b5(efficientnet.EfficientNet_B5_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "efficientnet_b6":
            model = efficientnet.efficientnet_b6(efficientnet.EfficientNet_B6_Weights.DEFAULT, progress=True)
        elif cfg.model_arch == "efficientnet_b7":
            model = efficientnet.efficientnet_b7(efficientnet.EfficientNet_B7_Weights.DEFAULT, progress=True)
        else:
            model = None
            raise ValueError(f"No model found with key {cfg.model_arch}")

        # Modify the model head
        model._fc = nn.Linear(model._fc.in_features, cfg.embedding_size, bias=True)
        return model

    else:
        raise ValueError(f"No model found with key {cfg.model_arch}")


def build_head(head_type, embedding_size, class_num, m, t_alpha, h, s, ):
    if head_type == 'adaface':
        head = AdaFace(embedding_size=embedding_size, classnum=class_num, m=m, h=h, s=s, t_alpha=t_alpha, )
    elif head_type == 'arcface':
        head = ArcFace(embedding_size=embedding_size, classnum=class_num, m=m, s=s, )
    elif head_type == 'cosface':
        head = CosFace(embedding_size=embedding_size, classnum=class_num, m=m, s=s, )
    else:
        raise ValueError('not a correct head type', head_type)
    return head


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class AdaFace(nn.Module):
    def __init__(self, embedding_size=512, classnum=70722, m=0.4, h=0.333, s=64., t_alpha=1.0):
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1) * (20))
        self.register_buffer('batch_std', torch.ones(1) * 100)

        print('\nAdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embbedings, kernel_norm)
        cosine = cosine.clamp(- 1 + self.eps, 1 - self.eps)  # for stability

        safe_norms = torch.clip(norms, min=0.001, max=100)  # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std + self.eps)  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = torch.clip(margin_scaler, -1, 1)
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


class CosFace(nn.Module):

    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.4):
        super(CosFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.4
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.eps = 1e-4

        print('init CosFace with ')
        print('self.m', self.m)
        print('self.s', self.s)

    def forward(self, embbedings, norms, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embbedings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)  # for stability

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        cosine = cosine - m_hot
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


class ArcFace(nn.Module):

    def __init__(self, embedding_size=512, classnum=51332, s=64., m=0.5):
        super(ArcFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369

        self.eps = 1e-4

    def forward(self, embbedings, norms, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = torch.mm(embbedings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)  # for stability

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.reshape(-1, 1), self.m)

        theta = cosine.acos()

        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi - self.eps)
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s

        return scaled_cosine_m
