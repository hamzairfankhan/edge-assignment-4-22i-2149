"""Unified inference for all baseline/KD students and the NAS classifier."""
import argparse,json
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from architectures import load_classifier,InferenceOnly
from nas_space import SearchCNN
ROOT=Path(__file__).resolve().parent
NAMES=[a+'_'+mode for a in ['mobilenet_v2','squeezenet','efficientnet_lite0'] for mode in ['baseline','kd']]+['nas_best']
def load_model(name):
    ck=torch.load(ROOT/'models'/(name+'.pth'),map_location='cpu',weights_only=True)
    if name=='nas_best':
        m=SearchCNN(ck['config']);m.load_state_dict(ck['model']);return m.eval(),ck['config']['resolution'],'CIFAR-10'
    return InferenceOnly(load_classifier(ROOT/'models'/(name+'.pth'))).eval(),64,'CIFAR-100'
def image_tensor(path,resolution,dataset):
    arr=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)/255
    x=torch.from_numpy(arr.transpose(2,0,1)).unsqueeze(0)
    x=torch.nn.functional.interpolate(x,(resolution,resolution),mode='bilinear',align_corners=False)
    mean,std=([.485,.456,.406],[.229,.224,.225]) if dataset=='CIFAR-100' else ([.4914,.4822,.4465],[.247,.2435,.2616])
    return (x-torch.tensor(mean).view(1,3,1,1))/torch.tensor(std).view(1,3,1,1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=NAMES,default='mobilenet_v2_kd');p.add_argument('--image',required=True);a=p.parse_args();torch.set_num_threads(4)
    m,res,ds=load_model(a.model);x=image_tensor(a.image,res,ds)
    with torch.inference_mode():prob=m(x).softmax(1)[0];index=int(prob.argmax())
    classes=json.loads((ROOT/'results/q1_results.json').read_text())['classes'] if ds=='CIFAR-100' else ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    print(dict(model=a.model,dataset=ds,class_index=index,class_name=classes[index],confidence=float(prob[index])))
