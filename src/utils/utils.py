import random
import numpy as np
import torch
import matplotlib.pyplot as plt

"""
Same padding:
P1 = (S-(T-K)%S)%S
(T+P1-K+P2)/S+1=(T-1)//S+1
(T+P1-K)/S+P2/S=(T-1+P1-P1+K-K)//S
P2/S=(K-P1-1)//S
P2=K-P1-1
"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def draw_loss(train_loss, valid_loss=None, save_path='./loss.png'):
    plt.figure()
    
    epochs = range(1, len(train_loss) + 1)
    
    if isinstance(train_loss, torch.Tensor):
        train_loss = train_loss.numpy()

    plt.plot(epochs, train_loss, label='train')
    
    if valid_loss is not None:
        if isinstance(valid_loss, torch.Tensor):
            valid_loss = valid_loss.numpy()
        plt.plot(epochs, valid_loss, label='valid')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
