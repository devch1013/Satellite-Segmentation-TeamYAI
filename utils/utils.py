import torch
import numpy as np
# from utils.loss_func import calculate_dice_scores
from torch import optim


def get_optimizer(model, cfg):
    """
    Return torch optimizer

    Args:
        model: Model you want to train
        cfg: Dictionary of optimizer configuration

    Returns:
        optimizer
    """
    optim_name = cfg["name"].lower()
    learning_rate = cfg["learning-rate"]
    args = cfg["args"]
    if optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, **args)
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, **args)
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, **args)
    elif optim_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, **args)
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(optimizer, cfg):
    '''
    get ["lr_scheduler"] cfg dictionary
    '''
    scheduler_name = cfg["name"].lower()
    args = cfg["args"]
    if scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', **args)
    elif scheduler_name == "multisteplr":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, **args)
    else:
        raise NotImplementedError
    return scheduler
    


def get_criterion(cfg):
    
    """
    Return torch criterion

    Args:
        cfg: Dictionary of criterion configuration

    Returns:
        criterion
    """
    
    criterion_name = cfg["name"].lower()
    print(criterion_name)
    if criterion_name == "crossentropyloss":
        criterion = torch.nn.BCEWithLogitsLoss()
        
    # elif criterion_name == "dice":
    #     criterion = calculate_dice_scores
    else:
        raise NotImplementedError
    return criterion

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)