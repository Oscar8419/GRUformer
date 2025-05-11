import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
frame_perSNR = 4096
frame_perModula = 4096 * 26


class CfgNode:
    def __init__(self, **kwargs):
        '''configuration of training'''
        self.__dict__.update(kwargs)


def get_linear_schedule_with_warmup_hold(
    optimizer,
    total_steps: int,
    warmup_ratio: float = 0.2,
    hold_ratio: float = 0.0,
    last_epoch: int = -1
):
    """
    参数说明：
    - optimizer: PyTorch优化器
    - total_steps: 总训练步数
    - warmup_ratio: 热身阶段占比
    - hold_ratio: 平台期占比
    - last_epoch: 恢复训练时的起始epoch
    """
    # 计算各阶段步数
    warmup_steps = int(total_steps * warmup_ratio)
    hold_steps = int(total_steps * hold_ratio)
    anneal_steps = total_steps - warmup_steps - hold_steps

    def lr_lambda(current_step):
        # 线性热身阶段（0 ~ warmup_steps）
        if current_step < warmup_steps:
            return current_step / warmup_steps
        # 平台期（warmup_steps ~ warmup_steps + hold_steps）
        elif current_step < warmup_steps + hold_steps:
            return 1.0
        # 余弦退火阶段（剩余步数）
        else:
            progress = (current_step - warmup_steps -
                        hold_steps) / anneal_steps
            return 0.5 * (1.0 + math.cos(math.pi * progress)) + 1e-3

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_confuse_matrix(checkpoint_name: str, y_true, y_pred, timestamp='25-1-12', labels=None, normalize='true',  save=True,):
    '''save confusion matrix in ./result_data/confusionMatrix_{timestamp}.pkl'''
    confusion_matrix_dict = {}
    confusion_matrix_dict['cp_name'] = checkpoint_name
    for snr in y_true.keys():
        cm = confusion_matrix(y_true[snr], y_pred[snr], )
        cm_norm = confusion_matrix(
            y_true[snr], y_pred[snr], normalize=normalize)
        confusion_matrix_dict[snr+"_cm"] = cm
        confusion_matrix_dict[snr+"_cm_norm"] = cm_norm
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm, display_labels=labels)
        confusion_matrix_dict[snr + '_disp'] = disp

    if save:
        with open(f'./result_data/confusionMatrix_{timestamp}.pkl', 'wb') as f:
            pickle.dump(confusion_matrix_dict, f)


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def draw_accuracy(path: str):
    '''path: confusion matrix pickle file path'''
    from model_dataset import SNR_START, SNR_STOP
    with open(path, 'rb') as f:
        data = pickle.load(f)
    x = []
    y = []
    for snr in range(SNR_START, SNR_STOP+1, 2):
        cm = data[str(snr) + '_cm_norm']
        x.append(snr)
        accuracy = round(np.mean(np.diagonal(cm)), 3)
        y.append(accuracy)
    plt.plot(x, y)
    plt.grid(True)
    plt.xlabel('SNR(dB)')
    plt.ylabel('Accuracy')
    plt.title('accuracy')
