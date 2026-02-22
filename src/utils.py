import imp
import torch
from torch.nn import L1Loss, MSELoss
import numpy as np
import cv2
import logging


def Initialize_Logger(log_path):
    """
    Initialize log file
    :param log_path: path to log file
    :return: logger object
    """
    logger = logging.getLogger("T1_to_B1")
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


class L1_loss(torch.nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()
        self.instance = L1Loss()

    def forward(self, target, output):
        l = self.instance(output[:,0,:,:], target[:,0,:,:])
        return l


def save_input(inputs, path, axis=1):
    if axis==1:
        for i in range(inputs.shape[1]):
            tmp = inputs[0,i,:,:].detach().cpu().numpy()
            tmp = ( (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp) + 1e-10) * (2**16 - 1) ).astype('uint16')
            cv2.imwrite(path + '_{}.png'.format(i), tmp)

    if axis==0:
        for i in range(inputs.shape[0]):
            tmp = inputs[i,0,:,:].detach().cpu().numpy()
            tmp = ( (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp) + 1e-10) * (2**16 - 1) ).astype('uint16')
            cv2.imwrite(path + '_{}.png'.format(i), tmp)

    return
