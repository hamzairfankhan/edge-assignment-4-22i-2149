"""Standalone CIFAR-100 ResNet-50 teacher training."""
import argparse
from q1_common import *
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=15);a=p.parse_args()
    device=setup();x,y,xv,yv,*_=dataset(device);m=Classifier('resnet50',pretrained=True).to(device)
    history=train(m,x,y,xv,yv,a.epochs,'teacher_resnet50');save(m,'teacher_resnet50',history)
