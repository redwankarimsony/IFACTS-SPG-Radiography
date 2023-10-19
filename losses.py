import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossBinary(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLossBinary, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # convert BCE_loss to probability
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# Example of usage
criterion = FocalLossBinary(alpha=0.25, gamma=2.0)



class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=3):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        logpt = -F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()
    


    
