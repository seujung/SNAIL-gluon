import torch
from mxnet.gluon.data import Dataset, DataLoader
from omniglot_dataset import OmniglotDataset
from batch_sampler import BatchSampler

def loader(config,ctx):
    N = config.N
    K = config.K
    iterations = config.iterations
    batch_size = config.batch_size
    download = config.download
    
    train_dataset = OmniglotDataset(mode='train',download=download)
    test_dataset = OmniglotDataset(mode='test',download=download)
    
    tr_sampler = BatchSampler(labels=train_dataset.y,
                                          classes_per_it=N,
                                          num_samples=K,
                                          iterations=iterations,
                                          batch_size=batch_size)
    
    te_sampler = BatchSampler(labels=test_dataset.y,
                                          classes_per_it=N,
                                          num_samples=K,
                                          iterations= iterations,
                                          batch_size= int(batch_size / len(ctx)))
    
    tr_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=tr_sampler)
    te_dataloader = torch.utils.data.DataLoader(test_dataset, batch_sampler=te_sampler)
    
    return tr_dataloader, te_dataloader
    
    
    
    
    
    
