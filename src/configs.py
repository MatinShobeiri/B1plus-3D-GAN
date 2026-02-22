from datetime import datetime
import os
import functools
import torch
from utils import Initialize_Logger
from torch.utils.tensorboard import SummaryWriter

# model name
comment = 'PDT2_B1_3D_filter'
# max number of epochs
epoch = 200
# continue training
load_check = False
# if continue training
checkpoint = 225
# number of input channels
in_channels = 1


lambda_normal = 0
lambda_cyc = 10
flag_cycle = False
batch = 4
lr = 1e-4
#image_size = (192, 256, 64)
image_size = (224, 288, 224)

dir_dataset = "/home/zakersho/Public/PDB1/peter_nii_PDT2b1/"

# Inference
path_inf_in = "/home/zakersho/Public/PDB1/Validation_in_PDT2b1/"
path_inf_out = "/home/zakersho/Public/PDB1/Validation_results_PDT2b1/"


# Model
features = 64
output_nc = 1
norm_layer = functools.partial(torch.nn.BatchNorm3d, affine=True, track_running_stats=True)
use_dropout = False
maxpool = False
avgpool = False

if flag_cycle:
    path_load_model = "/home/zakersho/Public/PDB1/Model_PDb1/_100_brain_alldata"
else:
    path_load_model = os.path.join("/home/zakersho/Public/PDB1/Model/", comment)

#N_split = 315
num_workers = 4
# Preview Function
test_data_dir = dir_dataset


if flag_cycle:
    path_test_save = os.path.join("/home/zakersho/Public/PDB1/Result_Test_PDb1/Cyc_GAN/", comment)
    log_dir = os.path.join("/home/zakersho/Public/PDB1/Result_Test_PDb1/Cyc_GAN/", comment, "Logs/")
    model_path_save = os.path.join("/home/zakersho/Public/PDB1/Model_PDb1/Cyc_GAN/", comment)
else:
    path_test_save = os.path.join("/home/zakersho/Public/PDB1/Result_Test/", comment)
    log_dir = os.path.join("/home/zakersho/Public/PDB1/Result_Test/", comment, "Logs/")
    model_path_save = os.path.join("/home/zakersho/Public/PDB1/Model/", comment)


if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
if not os.path.isdir(model_path_save):
    os.makedirs(model_path_save)
if not os.path.isdir(path_test_save):
    os.makedirs(path_test_save)

writer_train = SummaryWriter(log_dir=log_dir)
writer_test = SummaryWriter(log_dir=log_dir)

now = datetime.now()
logger = Initialize_Logger(log_path=log_dir+'/T1_to_B1{}-{}-{}-{}-{}-{}.log'.format(
    now.year, now.month, now.day, now.hour, now.minute, now.second))