"""A configurable CNN generator with all six required search dimensions."""
import random
import torch
from torch import nn
ACTIVATIONS={'relu':nn.ReLU,'relu6':nn.ReLU6,'hardswish':nn.Hardswish}
CHANNELS=[16,24,32,48,64,96]
def sample_config(rng):
    depth=rng.choice([2,3,4,5])
    return dict(depth=depth,channels=[rng.choice(CHANNELS) for _ in range(depth)],kernels=[rng.choice([3,5]) for _ in range(depth)],blocks=[rng.choice(['standard','depthwise']) for _ in range(depth)],activation=rng.choice(list(ACTIVATIONS)),resolution=rng.choice([24,32,40]))
def mutate(parent,rng):
    import copy
    c=copy.deepcopy(parent)
    for field in rng.sample(['depth','channels','kernels','blocks','activation','resolution'],2):
        if field=='depth':
            c['depth']=rng.choice([2,3,4,5])
            for key,options in [('channels',CHANNELS),('kernels',[3,5]),('blocks',['standard','depthwise'])]:
                c[key]=(c[key]+[rng.choice(options) for _ in range(5)])[:c['depth']]
        elif field in ['channels','kernels','blocks']:
            options={'channels':CHANNELS,'kernels':[3,5],'blocks':['standard','depthwise']}[field]
            c[field][rng.randrange(c['depth'])]=rng.choice(options)
        elif field=='activation': c[field]=rng.choice(list(ACTIVATIONS))
        else: c[field]=rng.choice([24,32,40])
    return c
class SearchCNN(nn.Module):
    def __init__(self,config):
        super().__init__(); layers=[]; cin=3; act=ACTIVATIONS[config['activation']]
        assert len(config['channels'])==len(config['kernels'])==len(config['blocks'])==config['depth']
        for i,(cout,k,kind) in enumerate(zip(config['channels'],config['kernels'],config['blocks'])):
            stride=1 if i==0 else 2
            if kind=='depthwise':
                layers.extend([nn.Conv2d(cin,cin,k,stride,k//2,groups=cin,bias=False),nn.BatchNorm2d(cin),act(),nn.Conv2d(cin,cout,1,bias=False),nn.BatchNorm2d(cout),act()])
            else: layers.extend([nn.Conv2d(cin,cout,k,stride,k//2,bias=False),nn.BatchNorm2d(cout),act()])
            cin=cout
        self.features=nn.Sequential(*layers);self.pool=nn.AdaptiveAvgPool2d(1);self.classifier=nn.Linear(cin,10)
    def forward(self,x): return self.classifier(self.pool(self.features(x)).flatten(1))
