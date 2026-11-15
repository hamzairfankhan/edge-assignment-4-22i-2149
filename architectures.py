"""Specified standard ResNet-50 and lightweight student architectures."""
import torch
from torch import nn
from torchvision import models
import timm

class Classifier(nn.Module):
    def __init__(self,name,pretrained=False):
        super().__init__(); self.name=name
        if name=='resnet50':
            self.network=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.network.fc=nn.Linear(2048,100); self.feature_dim=2048
        elif name=='mobilenet_v2':
            self.network=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None)
            self.network.classifier[1]=nn.Linear(1280,100); self.feature_dim=1280
        elif name=='squeezenet':
            self.network=models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None)
            self.network.classifier[1]=nn.Conv2d(512,100,1); self.network.num_classes=100; self.feature_dim=512
            # CIFAR-100 raw-logit head: avoid clipping every class to zero.
            nn.init.normal_(self.network.classifier[1].weight,mean=0.,std=.01)
            nn.init.zeros_(self.network.classifier[1].bias)
            self.network.classifier[2]=nn.Identity()
        elif name=='efficientnet_lite0':
            self.network=timm.create_model('tf_efficientnet_lite0',pretrained=pretrained,num_classes=100)
            self.feature_dim=self.network.num_features
        else: raise ValueError(name)
    def forward(self,x):
        m=self.network
        if self.name=='resnet50':
            z=m.maxpool(m.relu(m.bn1(m.conv1(x))))
            z=m.layer4(m.layer3(m.layer2(m.layer1(z))))
            f=m.avgpool(z).flatten(1); logits=m.fc(f)
        elif self.name=='mobilenet_v2':
            z=m.features(x); f=z.mean((2,3)); logits=m.classifier(f)
        elif self.name=='squeezenet':
            z=m.features(x); f=z.mean((2,3)); logits=m.classifier(z).flatten(1)
        else:
            z=m.forward_features(x); f=z.mean((2,3)); logits=m.forward_head(z)
        return logits,f

class InferenceOnly(nn.Module):
    def __init__(self,model): super().__init__(); self.model=model
    def forward(self,x): return self.model(x)[0]

def load_classifier(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    m=Classifier(ck['architecture']); m.load_state_dict(ck['model']); return m.eval()
