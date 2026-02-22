import os
import torch
from Data_loader import preview_dataloader, Inference_dataloader
import nibabel as nib
import numpy as np
from models import Identity, UnetGenerator, NLayerDiscriminator, VGG_encoder, VGG_decoder, Discriminator
import torch.nn as nn
from configs import *
import glob
import cv2
#from PIL import Image, ImageFilter


def resize_image_old(img, src_img):
       """Resize across z-axis"""
       # Set the desired depth
       desired_depth = src_img.shape[-1]
       desired_width = src_img.shape[-2]
       desired_height = src_img.shape[-3]
       # Get current depth
       current_depth = img.shape[-1]
       current_width = img.shape[-2]
       current_height = img.shape[-3]
 
       # Resize across z-axis
       if desired_depth < current_depth:
       
           pad_top = int(np.ceil(current_depth-desired_depth)/2)
           pad_bot = int(np.floor(current_depth-desired_depth+1)/2)
           img = img[:,:,pad_top:desired_depth+pad_top]
       return img
       
def resize_image(img):
    """Crop the padded 3D volume (img) to match src_img size in all axes."""
    desired_width  = 208
    desired_height = 288
    desired_depth  = 196

    current_width  = img.shape[0]
    current_height = img.shape[1]
    current_depth  = img.shape[2]

    # Compute crop start indices (center crop)
    crop_w = (current_width  - desired_width)  // 2
    crop_h = (current_height - desired_height) // 2
    crop_d = (current_depth  - desired_depth)  // 2

    # Crop symmetrically around the center
    img = img[
        crop_w : crop_w + desired_width,
        crop_h : crop_h + desired_height,
        crop_d : crop_d + desired_depth
    ]

    return img



path_model_enc = path_load_model + '/ENC_{}.model'.format(comment)
path_model_dec = path_load_model + '/DEC_{}.model'.format(comment)

print(path_model_enc)

print('Running inference for files in ' + path_inf_in)

nii_files = glob.glob(path_inf_in+'*_000_*.nii')
print(' Found {} files'.format(len(nii_files)))

gpu_ids = [0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#encoder_T1   = VGG_encoder(features=features, in_channels=in_channels, maxpool=maxpool, avgpool=avgpool,).to(device)
#decoder_T1B1 = VGG_decoder(features=features).to(device)

encoder_T1   = Identity()
decoder_T1B1 = UnetGenerator(input_nc=in_channels, output_nc=1, num_downs=3, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False)

encoder_T1   = torch.nn.DataParallel(encoder_T1,   device_ids=gpu_ids).to(device)
decoder_T1B1 = torch.nn.DataParallel(decoder_T1B1, device_ids=gpu_ids).to(device)

encoder_T1.load_state_dict  (torch.load(path_model_enc, map_location=device))
decoder_T1B1.load_state_dict(torch.load(path_model_dec, map_location=device))

encoder_T1 = encoder_T1.module.to(device)
decoder_T1B1 = decoder_T1B1.module.to(device)

for nii_file in nii_files:
    nii_file = os.path.basename(nii_file)
    print(' Processing ' + nii_file)
    path_nii_file = os.path.join(path_inf_in, nii_file)

    src_img, affine, header = Inference_dataloader(path_nii_file, in_channels, med_kernel=(0,0))

    src_img = src_img.to(device)

    latent = encoder_T1(src_img)
    output = decoder_T1B1(latent)
    output = np.squeeze(output.detach().cpu().numpy())

    src_img = src_img[:,0,:,:,:]
    src_img = np.squeeze(src_img.detach().cpu().numpy())

    #output = output*4096
    #output = output.astype("uint16")
  
    # Median Filter on B1 images
    #print(format(output.shape))
    gf = cv2.getGaussianKernel(5, 5) 

    ouput = cv2.sepFilter2D(output,-1, gf, gf)
    
    print(format(output.shape))

    output = resize_image(output)
    
    print(format(output.shape))

    if not os.path.isdir(path_test_save):
        os.mkdir(path_test_save)

    img_pre, img_sep, img_post = nii_file.partition('_000_')
    nii_file_out = 'output_'+img_post 

    img = nib.Nifti1Image(output, affine, header)
    nib.save(img, os.path.join(path_inf_out, nii_file_out))  
    print('Saved ' + nii_file_out + ' in ' + path_inf_out)
 
print('Inference done')