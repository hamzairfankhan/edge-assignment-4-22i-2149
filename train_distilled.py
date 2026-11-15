"""Standalone student KD pipeline, initialized independently from baseline training."""
import argparse
from q1_common import *
from architectures import load_classifier
from distillation import DistillationLoss
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--architecture',required=True,choices=['mobilenet_v2','squeezenet','efficientnet_lite0']);p.add_argument('--epochs',type=int,default=10);a=p.parse_args()
    device=setup();x,y,xv,yv,*_=dataset(device);teacher=load_classifier(MODELS/'teacher_resnet50.pth').to(device)
    cache=teacher_cache(teacher,x);del teacher;setup();m=Classifier(a.architecture,pretrained=True).to(device)
    criterion=DistillationLoss(m.feature_dim).to(device);name=a.architecture+'_kd'
    history=train(m,x,y,xv,yv,a.epochs,name,criterion,cache);save(m,name,history,criterion)
