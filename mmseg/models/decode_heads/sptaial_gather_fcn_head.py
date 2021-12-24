import torch
import torch.nn.functional as F

from ..builder import HEADS
from .fcn_head import FCNHead


def update_logits(feats, logits):
    batch_size, num_classes, height, width = logits.size()
    channels = feats.size(1)

    logits = logits.view(batch_size, num_classes, -1)  # [batch_size, num_classes, height*width]
    probs = F.softmax(logits, dim=2)  # [batch_size, num_classes, height*width]

    feats = feats.view(batch_size, channels, -1)  # [batch_size, channels, height*width]
    new_centers = torch.matmul(probs, feats.permute(0, 2, 1))  # [batch_size, num_classes, channels]

    new_logits = torch.matmul(new_centers, feats)  # [batch_size, num_classes, height*width]
    new_logits = new_logits.view(batch_size, num_classes, height, width)  # [batch_size, num_classes, height, width]

    return new_logits, new_centers


@HEADS.register_module()
class SpatialGatherFCNHead(FCNHead):
    def __init__(self, update_num_iters=1, save_centers=False, **kwargs):
        super(SpatialGatherFCNHead, self).__init__(**kwargs)

        self.save_centers = save_centers
        self.update_num_iters = update_num_iters
        assert self.update_num_iters > 0
    
    def process_logits(self, logits, features):
        centers = None
        for _ in range(self.update_num_iters):
            logits, centers = update_logits(features, logits)
        
        if self.save_centers is not None:
            assert centers is not None
            self.forward_output = centers.permute(0, 2, 1).contiguous().unsqueeze(3)
        
        return logits
