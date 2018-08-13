import numpy as np
from mxnet import gluon, autograd, nd
import torch


def batch_for_few_shot(num_cls,num_samples,batch_size, x, y):
    seq_size = num_cls * num_samples + 1
    one_hots = []
    last_targets = []
    for i in range(batch_size):
        one_hot, idxs = labels_to_one_hot(y[i * seq_size: (i + 1) * seq_size])
        one_hots.append(one_hot)
        last_targets.append(idxs[-1])
    #last_targets = Variable(torch.Tensor(last_targets).long())
    last_targets =torch.Tensor(last_targets).long()
    one_hots = [torch.Tensor(temp) for temp in one_hots]
    y = torch.cat(one_hots, dim=0)
    #x, y = Variable(x), Variable(y)
    x = nd.array(x.data.numpy())
    y = nd.array(y.data.numpy())
    last_targets = nd.array(last_targets.data.numpy())
    return x, y, last_targets

def labels_to_one_hot(labels):
    labels = labels.numpy()
    unique = np.unique(labels)
    map = {label:idx for idx, label in enumerate(unique)}
    idxs = [map[labels[i]] for i in range(labels.size)]
    one_hot = np.zeros((labels.size, unique.size))
    one_hot[np.arange(labels.size), idxs] = 1
    return one_hot, idxs