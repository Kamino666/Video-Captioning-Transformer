import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from model import VideoCaptionSwinTransformer
from visualdl import LogWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = LogWriter(logdir="./log")


