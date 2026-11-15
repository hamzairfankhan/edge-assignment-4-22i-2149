"""Hard labels + temperature-scaled soft labels + intermediate feature transfer."""
import torch
from torch import nn
from torch.nn import functional as F

class DistillationLoss(nn.Module):
    def __init__(self,student_dim,teacher_dim=2048,temperature=4.,alpha=.5,beta=1.):
        super().__init__(); self.projector=nn.Linear(student_dim,teacher_dim)
        self.temperature=temperature; self.alpha=alpha; self.beta=beta
    def forward(self,logits,features,labels,teacher_logits,teacher_features):
        t=self.temperature
        hard=F.cross_entropy(logits,labels)
        soft=F.kl_div(F.log_softmax(logits/t,dim=1),F.softmax(teacher_logits/t,dim=1),reduction='batchmean')*(t*t)
        projected=F.normalize(self.projector(features),dim=1)
        target=F.normalize(teacher_features,dim=1)
        feature=(projected-target).pow(2).sum(dim=1).mean()
        return (1-self.alpha)*hard+self.alpha*soft+self.beta*feature,hard,soft,feature
