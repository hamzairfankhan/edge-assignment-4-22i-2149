"""Independent hard-label-only pipeline: no teacher or KD loss is loaded."""
import argparse
from q1_common import *
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--architecture',required=True,choices=['mobilenet_v2','squeezenet','efficientnet_lite0']);p.add_argument('--epochs',type=int,default=10);a=p.parse_args()
    device=setup();x,y,xv,yv,*_=dataset(device);m=Classifier(a.architecture,pretrained=True).to(device)
    name=a.architecture+'_baseline';history=train(m,x,y,xv,yv,a.epochs,name);save(m,name,history)
