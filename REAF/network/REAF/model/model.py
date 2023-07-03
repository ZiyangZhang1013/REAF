from typing import List, Optional, Union

import torch
from segmentation_models_pytorch.base import (ClassificationHead, SegmentationHead,
                                              SegmentationModel)
from segmentation_models_pytorch.decoders.unetplusplus.model import \
    UnetPlusPlusDecoder
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
from torchvision import models


class UnetPlusPlus(SegmentationModel):

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1],
                                                          **aux_params)
        else:
            self.classification_head = None

        self.name = "unetplusplus-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def get_feature(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)
        return features[-1]

    def get_feature_mask(self, x):
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return features[-1], masks


class Diagnosis(nn.Module):

    def __init__(self, class_num, feature_model_load_path, pretrained=True):
        super(Diagnosis, self).__init__()
        self.feature1 = UnetPlusPlus()
        self.feature1.load_state_dict(torch.load(feature_model_load_path))

        self.pre_process = nn.Sequential(
            nn.Conv2d(4, 3, (1, 1)),
            nn.ReLU(),
        )
        self.feature2 = models.vgg16(pretrained=pretrained)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.adaptor = nn.Linear(1000, 512)

        self.classifier = nn.Linear(512, class_num)

    def forward(self, x, lesionmap):
        with torch.no_grad():
            lesionmap = self.feature1(x).detach()
            lesionmap = torch.sigmoid(lesionmap)
        x = torch.cat([x, lesionmap], 1)
        x = self.pre_process(x)
        feature2 = self.feature2(x)  # 1000
        feature2 = torch.relu(feature2)
        feature2 = self.adaptor(feature2)  # 512
        feature = feature2
        feature = torch.relu(feature)
        logit = self.classifier(feature)
        return logit
