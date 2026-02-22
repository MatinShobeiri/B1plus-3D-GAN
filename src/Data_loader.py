import os
from os import listdir
from os.path import isfile, join
import glob
#import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
#from torchvision import transforms
import time
import scipy
from scipy import ndimage
import nibabel as nib # added by -ps- for nii support

# -ps- images are already brain masked, reduces the number of images to transfer, so commenting out all brain mask loading
# -ps- nii support based on https://github.com/keras-team/keras-io/blob/master/examples/vision/3D_image_classification.py

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    img = scan.get_fdata()
    aff = scan.affine
    hdr = scan.header
    return img, aff, hdr

def normalize(volume):
    """Normalize the volume"""
    #print(np.dtype(volume))
    #print(' Normalizing {}'.format(volume.any()))
    if (volume.any() > 1):
       min = 0
       max = 4096
       volume = volume.astype("float32")
       volume[volume < min] = min
       volume[volume > max] = max
       volume = (volume - min) / (max - min)
    return volume

def resize_volume(img, image_size):
    if image_size:
        # Desired sizes
        desired_width  = image_size[0]  # X dimension
        desired_height = image_size[1]  # Y dimension
        desired_depth  = image_size[2]  # Z dimension

        # Current sizes
        current_width  = img.shape[0]
        current_height = img.shape[1]
        current_depth  = img.shape[2]

        # Compute required padding for each dimension
        pad_w = max(desired_width  - current_width,  0)
        pad_h = max(desired_height - current_height, 0)
        pad_d = max(desired_depth  - current_depth,  0)

        # Split padding evenly between both sides
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_front  = pad_d // 2
        pad_back   = pad_d - pad_front

        # Apply 3D padding
        img = np.pad(
            img,
            ((pad_left, pad_right), (pad_top, pad_bottom), (pad_front, pad_back)),
            mode='constant',
            constant_values=0
        )

    return img


def process_scan(path, image_size):
    """Read and resize volume"""
    # Read scan
    volume, aff, hdr = read_nifti_file(path)
    # Normalize
    #volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume, image_size)
    return volume, aff, hdr



class Pair_T1B1_Dataset(data.Dataset):
    def __init__(self, dir_dataset, n_train, image_size=None, med_kernel=(1,1), gaussian=False, in_channels = 0):
        super(Pair_T1B1_Dataset, self).__init__()

        self.gaussian = gaussian
        self.image_size = image_size
        self.med_kernel = med_kernel
        self.path_base = dir_dataset
        self.in_channels = in_channels 
        self.dir_T1_nifti = []
        self.folders = []
        
        
        for n in n_train:

            name = 'source_000_{:04d}.nii'.format(n)
            if os.path.isfile(join(dir_dataset,name)):

                dir_T1_nifti_tmp = glob.glob(dir_dataset + '/target_{:04d}.nii'.format(n))
                selected = [f for f in dir_T1_nifti_tmp]
                
                self.dir_T1_nifti += selected
                self.folders.append(n)
        print('Init done. Selected length = ' + format(len(self.dir_T1_nifti)))
            

    def __getitem__(self, index):
        p_h = np.random.randint(2)
        p_v = np.random.randint(2)

        ind=self.folders[index]


        path_to_images = self.path_base

        t1 = time.time()
        source = np.zeros(self.image_size)
        source = np.expand_dims(source,0)
        source = np.repeat(source, self.in_channels,0)

        for ic in range(0,self.in_channels):
 
            source[ic,:,:,:], _, _ = process_scan(path_to_images + '/source_{:03d}_{:04d}.nii'.format(ic,ind), self.image_size)

        target, _, _ = process_scan(path_to_images + '/target_{:04d}.nii'.format(ind), self.image_size)

        T1 = torch.from_numpy(source)
        B1 = torch.from_numpy(target)
        t2 = time.time()


        # Median Filter on B1 images
        #if np.sum(self.med_kernel) > 2:
            #B1 = cv2.medianBlur(B1,5)
        
        if self.gaussian:
            # cv2.imwrite('/lustre04/scratch/javadhe/Result_Test/Pix2pix/sample.png', B1)
            kernel = np.ones((5, 5), np.float32) / 25
            # B1 = cv2.fastNlMeansDenoising(B1, None, 16, 16, 12)
            #B1 = cv2.filter2D(B1, -1, kernel)
            # cv2.imwrite('/lustre04/scratch/javadhe/Result_Test/Pix2pix/sample_guassian.png', B1)


        inputs_T1 = T1
        outputs_B1 = B1

        outputs_B1 = torch.unsqueeze(outputs_B1,0)
        t3 = time.time()

        return inputs_T1.float(), outputs_B1.float()

    def __len__(self):
        return len(self.dir_T1_nifti)


def preview_dataloader(dir_folder, n_train, image_size=None, in_channels=0, med_kernel=(5,5)):
    
    path_to_images = dir_folder
    for n in n_train:
        name = 'source_000_{:04d}.nii'.format(n)
        if os.path.isfile(join(path_to_images,name)):
            index = n
            break

    source = np.zeros(image_size)
    source = np.expand_dims(source,0)
    source = np.repeat(source, in_channels,0)    
    for ic in range(0,in_channels):
        source[ic,:,:,:], _, _ = process_scan(path_to_images + '/source_{:03d}_{:04d}.nii'.format(ic,index), (224, 288, 224))
    target, aff, hdr = process_scan(path_to_images + '/target_{:04d}.nii'.format(index), (224, 288, 224))

    source_t = torch.from_numpy(source)
    target_t = torch.from_numpy(target)

    source_t = torch.unsqueeze(source_t,0)
    target_t = torch.unsqueeze(target_t,0)


    return source_t.float(), target_t.float(), aff, hdr



def Inference_dataloader(path_to_image, in_channels=0, med_kernel=(5,5)):

    nii_file=os.path.basename(path_to_image)
    path_to_images=os.path.dirname(path_to_image)
    img_pre, img_sep, img_post = nii_file.partition('_000_')
    source, aff, hdr = process_scan(path_to_images + '/' + img_pre +'_000_' + img_post, (224, 288, 224))
    image_size=source.shape
    source = np.expand_dims(source,0)
    source = np.repeat(source, in_channels,0) 
       
    for ic in range(0,in_channels):
        source[ic,:,:,:], aff, hdr = process_scan(path_to_images + '/' + img_pre +'_{:03d}_'.format(ic) + img_post, (224, 288, 224))

    source_t = torch.from_numpy(source)
    source_t = torch.unsqueeze(source_t,0)
    
    return source_t.float(), aff, hdr 

