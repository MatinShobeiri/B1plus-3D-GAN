from genericpath import isfile
import torch
import numpy as np
import os
from torch.autograd import Variable
from Data_loader import preview_dataloader
from utils import save_input
import nibabel as nib # added by -ps- for nii support


def preview_results(test_data_dir, model, e, device, path_test_save, n_train, image_size, in_channels=0):
    with torch.no_grad():

        T1, B1, aff, hdr = preview_dataloader(test_data_dir, n_train, image_size, in_channels)
        enc = model[0]
        dec = model[1]
        enc = enc.to(device)
        dec = dec.to(device)

        src = T1
        T1 = T1.to(device)
        if e==0:
            B1 = torch.squeeze(B1)
            img = nib.Nifti1Image(B1, aff, hdr)
            nib.save(img, path_test_save+'/target_pre_{}.nii'.format(e))
            ii = 0
            src = torch.squeeze(src)
            for srci in src:
                srci = torch.squeeze(srci)
                img = nib.Nifti1Image(srci, aff, hdr)
                nib.save(img, path_test_save+'/source_{:03d}_pre_{}.nii'.format(ii,e))
                ii = ii + 1


        latent = enc(T1)
        output = dec(latent)

        output = torch.squeeze(output)

        #print('output shape: {}'.format(output.shape))
        output = output.cpu()
        #empty_header = nib.Nifti1Header()
        #img = nib.Nifti1Image(output, np.eye(4), empty_header)
        img = nib.Nifti1Image(output, aff, hdr)

        nib.save(img, path_test_save+'/output_pre_{}.nii'.format(e))
        print(' Preview output saved as ' + path_test_save+'/output_pre_{}.nii'.format(e))

    return


def train(dataloaders, model_enc, model_dec, discriminator, optimizer, optimizer_disc, 
            criterion, batch, epoch, device, phase=['train', 'test'], disc=False, flag_cycle=True, 
            writers=None, model_path_save=None, comment=None, checkpoint_interval=25,
            lambda_normal=0, lambda_cyc=10, logger=None):

    def writer_saver(writer):
        writer.add_scalar('Loss', running_loss, epoch)
        writer.add_scalar('Loss input', running_loss_T1, epoch)
        writer.add_scalar('Loss output', running_loss_B1, epoch)
        writer.add_scalar('Loss gen', running_loss_gen, epoch)
        writer.add_scalar('Loss disc', running_loss_disc, epoch)
        writer.add_scalar('Loss input cyc', running_loss_T1_cyc, epoch)
        writer.add_scalar('Loss output cyc', running_loss_B1_cyc, epoch)
        return

    def model_saver():
        ####  .module
        torch.save(enc_T1.state_dict(), model_path_save + "/ENC_{}.model".format(comment))
        torch.save(dec_T1B1.state_dict(), model_path_save + "/DEC_{}.model".format(comment))
        torch.save(disc_B1.state_dict(), model_path_save + "/DISC_{}.model".format(comment))
        #np.savetxt(model_path_save + "/latest_epoch.txt", format(epoch))
        with open(model_path_save + "/latest_epoch.txt", 'w') as f:
            f.write('Most recent model saved at epoch %d' % epoch)
        if flag_cycle:
            torch.save(enc_B1.state_dict(), model_path_save + "/ENC_B1_{}_{}.model".format(comment, epoch))
            torch.save(dec_B1T1.state_dict(), model_path_save + "/DEC_B1T1_{}_{}.model".format(comment, epoch))
            torch.save(disc_T1.state_dict(), model_path_save + "/DISC_T1_{}_{}.model".format(comment, epoch))
        return

    print('Epoch {}'.format(epoch))

    target_real = Variable(torch.Tensor(batch).fill_(1.0), requires_grad=False).to(device)
    target_fake = Variable(torch.Tensor(batch).fill_(0.0), requires_grad=False).to(device)
    
    enc_T1 = model_enc[0].to(device)
    enc_B1 = model_enc[1].to(device) if flag_cycle else None
    dec_T1B1 = model_dec[0].to(device)
    dec_B1T1 = model_dec[1].to(device) if flag_cycle else None
    disc_T1 = discriminator[1].to(device) if flag_cycle else None
    disc_B1 = discriminator[0].to(device)
      
    for ph in phase:
        if ph=='train':
            enc_T1.train()
            dec_T1B1.train() 
            disc_B1.train()
            if flag_cycle:
                disc_T1.train()
                enc_B1.train()
                dec_B1T1.train()

            dataloader = dataloaders[0]
            writer = writers[0]
            
        
        elif ph=='test':
            enc_T1.eval()
            dec_T1B1.eval()
            disc_B1.eval()
            if flag_cycle:
                disc_T1.eval()
                enc_B1.eval()
                dec_B1T1.eval()

            dataloader = dataloaders[1]
            writer = writers[0]
            
        running_loss = 0
        running_loss_T1 = 0
        running_loss_B1 = 0
        running_loss_gen = 0
        running_loss_disc = 0
        running_loss_T1_cyc = 0
        running_loss_B1_cyc = 0

        with torch.set_grad_enabled(ph=='train'):
        
        
            #print('1')
            
            try:
                if logger is not None:
                    logger.debug('Dataloader {} started'.format(ph))

                #for i, (T1, B1, brain) in enumerate(dataloader):
                #print('2')
                
                for i, (T1, B1) in enumerate(dataloader):
                    
                    #print('3')
                    
                    optimizer.zero_grad()
                    optimizer_disc.zero_grad()
                    
                    #print('4')

                    if epoch==0 and i==0 and ph=='train':
                        print('T1 shape: {}'.format(T1.shape))
                        print('B1 shape: {}'.format(B1.shape))
                        
                        #print('5')
                        
                        #save_input(T1, path=model_path_save+'/T1', axis=0)
                        #save_input(B1, path=model_path_save+'/B1', axis=0)

                    T1 = T1.to(device)
                    B1 = B1.to(device)
                    #brain = brain.to(device)
                    
                    #print('+')
                    # Normal path
                    latent_T1 = enc_T1(T1)
  
                    #output_B1_fake = dec_T1B1(latent_T1) * brain
                    output_B1_fake = dec_T1B1(latent_T1)
                    latent_B1 = enc_B1(B1) if flag_cycle else None
                    #output_T1_fake = dec_B1T1(latent_B1) * brain if flag_cycle else None
                    output_T1_fake = dec_B1T1(latent_B1) if flag_cycle else None
                    # Cyc path
                    if flag_cycle:
                        output_B1_fake_in = output_B1_fake
                        output_T1_fake_in = output_T1_fake

                    latentT1B1T1 = enc_B1(output_B1_fake_in)  if flag_cycle else None
                    #output_T1B1T1 = dec_B1T1(latentT1B1T1) * brain  if flag_cycle else None
                    output_T1B1T1 = dec_B1T1(latentT1B1T1) if flag_cycle else None
                    latentB1T1B1 = enc_T1(output_T1_fake_in)  if flag_cycle else None
                    #output_B1T1B1 = dec_T1B1(latentB1T1B1) * brain  if flag_cycle else None
                    output_B1T1B1 = dec_T1B1(latentB1T1B1) if flag_cycle else None
                    #save_input(output_B1_fake_in, 'output_B1_fake_in')
                    # GAN path
                    #print('++')
                    #print('output_B1_fake shape: {}'.format(output_B1_fake.shape))
                    input_T1_fake = torch.cat((output_T1_fake, B1[:,0:1,:,:,:]), 1) if flag_cycle else None
                    input_B1_fake = torch.cat((output_B1_fake, T1[:,0:1,:,:,:]), 1)
                    #print('+++')
                    input_T1_real = torch.cat((T1[:,0:1,:,:,:], B1[:,0:1,:,:,:]), 1) if flag_cycle else None
                    input_B1_real = torch.cat((B1[:,0:1,:,:,:], T1[:,0:1,:,:,:]), 1)
                    p_T1_fake = disc_T1(input_T1_fake) if flag_cycle else None
                    p_B1_fake = disc_B1(input_B1_fake)
                    p_T1_real = disc_T1(input_T1_real) if flag_cycle else None
                    p_B1_real = disc_B1(input_B1_real)

                    print('Loss calc {}'.format(i))
                    # Loss Calculations
                    if flag_cycle:
                        loss_disc = criterion[1](p_T1_fake, target_fake) + criterion[1](p_B1_fake, target_fake) + \
                                        criterion[1](p_T1_real, target_real) + criterion[1](p_B1_real, target_real)
                    else:
                        loss_disc = criterion[1](p_B1_fake, target_fake) + criterion[1](p_B1_real, target_real)
                    loss_disc = loss_disc * 0.5

                    if flag_cycle:
                        loss_gen = criterion[1](p_T1_fake, target_real) + criterion[1](p_B1_fake, target_real)
                    else: 
                        loss_gen = criterion[1](p_B1_fake, target_real)

                    loss_T1 = criterion[0](T1[:,0:1,:,:,:], output_T1_fake) if flag_cycle else 0
                    loss_B1 = criterion[0](B1[:,0:1,:,:,:], output_B1_fake)
                    loss_T1_cyc = criterion[0](T1[:,0:1,:,:,:], output_T1B1T1) if flag_cycle else 0
                    loss_B1_cyc = criterion[0](B1[:,0:1,:,:,:], output_B1T1B1) if flag_cycle else 0

                    if flag_cycle:
                        loss = loss_gen + loss_T1_cyc * lambda_cyc + loss_B1_cyc * lambda_cyc
                    else:
                        loss = loss_gen + loss_B1 * lambda_cyc
                    loss = loss / len(T1)

                    if ph=='train' and not disc:
                        loss.backward()
                        optimizer.step()
                    elif ph=='train' and disc:
                        loss_disc.backward()
                        optimizer_disc.step()

                    running_loss += loss.item()
                    running_loss_B1 += loss_B1.item()
                    running_loss_gen += loss_gen.item()
                    running_loss_disc += loss_disc.item()
                    if flag_cycle:
                        running_loss_B1_cyc += loss_B1_cyc.item()
                        running_loss_T1_cyc += loss_T1_cyc.item()
                        running_loss_T1 += loss_T1.item()

                    if ph == "train": 
                        print("minibatch {} of {}:"" runnning_loss : {:.4f}, ".format(i, len(dataloader), running_loss / (i + 1)))

                    torch.cuda.empty_cache()
                    
                    
            #Updating the batch
            
            except Exception as e:
                print('Error: {}'.format(e))
                #print([dataloader.folder_ind, dataloader.ind])
                

        # Save and print checkpoint
        if not disc and ph=='test':
            writer_saver(writer)
            print('Loss {} (epoch {}): {:.4f}'.format(ph, epoch, running_loss))

            if logger is not None:
                logger.debug('Loss {} (epoch {}): {:.4f}'.format(ph, epoch, running_loss))

        elif not disc and ph=='train':
            if logger is not None:
                logger.debug('Loss {} (epoch {}): {:.4f}'.format(ph, epoch, running_loss))

    #Phase Updating
    
    if (epoch + 0) % checkpoint_interval==0 and not disc:
        model_saver()
        if logger is not None:
            logger.debug('Models are saved')

 
    return


