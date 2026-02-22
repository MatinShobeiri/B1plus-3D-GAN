import torch
import numpy as np
from torch.nn import L1Loss, MSELoss
from utils import L1_loss
from torch.utils.data import DataLoader
from Data_loader import Pair_T1B1_Dataset
from models import Identity, UnetGenerator, NLayerDiscriminator, VGG_encoder, VGG_decoder, Discriminator
from trainer import train, preview_results
from configs import *
import argparse
import pickle
import torch.nn as nn
import torch.optim.lr_scheduler as LS


gpu_ids = [0,1,2,3]


print(gpu_ids)


##n_train = list(range(0,192))
##n_test =  list(range(192,240))

##n_train = list(range(0,64))
##n_test =  list(range(64,80))

n_train = list(range(0,256))
n_test =  list(range(256,320))


available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print(available_gpus)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


dataset_train = Pair_T1B1_Dataset(dir_dataset, n_train, image_size=image_size, in_channels = in_channels)
dataset_test  = Pair_T1B1_Dataset(dir_dataset, n_test,  image_size=image_size, in_channels = in_channels)


if logger is not None:
    logger.debug('Datasets are configured')
    
print('Dataset')

dataloader_train = DataLoader(dataset=dataset_train, num_workers=num_workers, batch_size=batch, shuffle=True, drop_last=True)
dataloader_test  = DataLoader(dataset=dataset_test,  num_workers=num_workers, batch_size=batch, shuffle=True, drop_last=True)

if logger is not None:
    logger.debug('Dataloaders are configured')
    
print('DataLoader: [{}, {}]'.format(len(dataloader_train), len(dataloader_test)))

encoder_T1   = Identity()
decoder_T1B1 = UnetGenerator(input_nc=in_channels, output_nc=1, num_downs=3, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False)
encoder_B1   = Identity() if flag_cycle else None
decoder_B1T1 = UnetGenerator(input_nc=in_channels, output_nc=1, num_downs=3, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False) if flag_cycle else None

#encoder_T1   = VGG_encoder(features=features, in_channels=in_channels, maxpool=maxpool, avgpool=avgpool,).to(device)
#decoder_T1B1 = VGG_decoder(features=features).to(device)
#encoder_B1   = VGG_encoder(features=features, in_channels=in_channels, maxpool=maxpool, avgpool=avgpool,).to(device) if flag_cycle else None
#decoder_B1T1 = VGG_decoder(features=features).to(device) if flag_cycle else None

disc_B1 = NLayerDiscriminator(input_nc=2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d)
disc_T1 = NLayerDiscriminator(input_nc=2, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d) if flag_cycle else None

#disc_B1 = Discriminator(features=features, maxpool=maxpool, avgpool=avgpool).to(device)
#disc_T1 = Discriminator(features=features, maxpool=maxpool, avgpool=avgpool).to(device) if flag_cycle else None


if logger is not None:
    logger.debug('Models are created')

encoder_T1   = torch.nn.DataParallel(encoder_T1,   device_ids=gpu_ids).to(device)
decoder_T1B1 = torch.nn.DataParallel(decoder_T1B1, device_ids=gpu_ids).to(device)
encoder_B1   = torch.nn.DataParallel(encoder_B1,   device_ids=gpu_ids).to(device) if flag_cycle else None
decoder_B1T1 = torch.nn.DataParallel(decoder_B1T1, device_ids=gpu_ids).to(device) if flag_cycle else None
disc_T1      = torch.nn.DataParallel(disc_T1,      device_ids=gpu_ids).to(device) if flag_cycle else None
disc_B1      = torch.nn.DataParallel(disc_B1,      device_ids=gpu_ids).to(device)

if load_check:
    print(' Loading previously saved model')
    encoder_T1.load_state_dict      (torch.load(path_load_model + '/ENC_{}.model'. format(comment)))
    decoder_T1B1.load_state_dict    (torch.load(path_load_model + '/DEC_{}.model'. format(comment)))
    disc_B1.load_state_dict         (torch.load(path_load_model + '/DISC_{}.model'.format(comment)))
    if flag_cycle:
        encoder_B1.load_state_dict  (torch.load(path_load_model + '/ENC_B1__100_brain_alldata_{}.model'.  format(checkpoint)))
        decoder_B1T1.load_state_dict(torch.load(path_load_model + '/DEC_B1T1__100_brain_alldata_{}.model'.format(checkpoint)))
        disc_T1.load_state_dict     (torch.load(path_load_model + '/DISC_T1__100_brain_alldata_{}.model'. format(checkpoint)))

    if logger is not None:
        logger.debug('Models are loaded from checkpoint {}'.format(checkpoint))
    print('Models are loaded from checkpoint {}'.format(checkpoint))

print('*'*8, 'Encoder','*'*8)
print(encoder_T1)
print('*'*8, 'Decoder','*'*8)
print(decoder_T1B1)

if flag_cycle:
    optimizer      = torch.optim.Adam(list(encoder_T1.parameters()) + list(decoder_T1B1.parameters()) + list(encoder_B1.parameters()) + list(decoder_B1T1.parameters()),
                             lr=lr, betas=(0.5, 0.999), weight_decay=0.0)
    optimizer_disc = torch.optim.Adam(list(disc_T1.parameters()) + list(disc_B1.parameters()),
                                  lr=lr, betas=(0.5, 0.999), weight_decay=0.0)
                            
    #lr
    scheduler1 = LS.MultiStepLR(optimizer,      milestones = [650, 690], gamma = 0.1)
    scheduler2 = LS.MultiStepLR(optimizer_disc, milestones = [650, 690], gamma = 0.1)
    #lr
    
else:
    optimizer      = torch.optim.Adam(list(encoder_T1.parameters()) + list(decoder_T1B1.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0)
    optimizer_disc = torch.optim.Adam(list(disc_B1.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0)
    
    #lr
    scheduler1 = LS.MultiStepLR(optimizer,      milestones = [650, 690], gamma = 0.1)
    scheduler2 = LS.MultiStepLR(optimizer_disc, milestones = [650, 690], gamma = 0.1)
    #lr


criterion_cyc = L1Loss()
criterion_GAN = MSELoss()

torch.cuda.empty_cache()

if logger is not None:
    logger.debug('Training started')

epoch_start = checkpoint+1 if load_check else 0
for e in range(epoch_start, epoch):
    if logger is not None:
        logger.debug('Epoch {}'.format(e))

    for disc in  [True, False]: 
        train(dataloaders = [dataloader_train, dataloader_test], 
            model_enc = [encoder_T1, encoder_B1] if flag_cycle else [encoder_T1],
            model_dec = [decoder_T1B1, decoder_T1B1] if flag_cycle else [decoder_T1B1], 
            discriminator = [disc_B1, disc_T1] if flag_cycle else [disc_B1],
            optimizer = optimizer, 
            optimizer_disc = optimizer_disc,
            criterion = [criterion_cyc, criterion_GAN], 
            batch = batch,
            epoch = e, 
            device = device, 
            flag_cycle = flag_cycle,
            writers = [writer_train, writer_test],
            model_path_save = model_path_save,
            comment = comment,
            lambda_normal = lambda_normal, 
            lambda_cyc = lambda_cyc,
            disc = disc,
            logger=logger)
    
    print('\n'*3)
    print('in_channels: {}'.format(in_channels))
    print('lambda_cyc: {}'.format(lambda_cyc))
    print('lambda_normal {}'.format(lambda_normal))
    
    #lr
    print('lr {}'.format(optimizer.param_groups[0]['lr']))
    print('lr_dics {}'.format(optimizer_disc.param_groups[0]['lr']))
    #lr
    
    print('\n'*3)

    if (e)%10 == 0:
        if logger is not None:
            logger.debug('Generating previews')

        print('previewing the images')
        
        preview_results(test_data_dir, [encoder_T1, decoder_T1B1], e, device, path_test_save, n_test,
                    image_size=image_size, in_channels=in_channels)
    #lr
    scheduler1.step()
    scheduler2.step()  
    #lr              
