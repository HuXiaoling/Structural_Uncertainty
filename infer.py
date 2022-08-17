import torch
import numpy as np

import argparse, json
import os, glob, sys
from time import time

from dataloader import CREMI, ISBI2013, DRIVE
from unet.unet_model import UNet
import torch
torch.cuda.empty_cache()
from torchvision.utils import save_image
from probabilistic_unet import ProbabilisticUnet
import torch.nn.functional as F

def parse_func(args):
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    activity = params['common']['activity']
    mydict = {}
    mydict['files'] = [params['common']['img_file'], params['common']['gt_file']]
    mydict['checkpoint_restore'] = params['common']['checkpoint_restore']

    mydict['validation_datalist'] = params['validation']['validation_datalist']
    mydict['output_folder'] = params['validation']['output_folder']
    mydict['validation_batch_size'] = int(params['validation']['validation_batch_size'])
    
    return activity, mydict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    parser.add_argument('--dataset', type= str, default = "CREMI")
    parser.add_argument('--folder', type= str, default = "version1")   

    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        activity, mydict = parse_func(args)

    with open(args.params, 'r') as f:
        params = json.load(f)

    mydict['output_folder'] = 'experiments/' + args.dataset + '/' + args.folder
    # import pdb; pdb.set_trace()
    
    # call train
    print("Inference!")
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    # Test Data
    if args.dataset == 'CREMI':
        validation_set = CREMI(mydict['validation_datalist'], mydict['files'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],shuffle=False,num_workers=1, drop_last=False)
        net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=1)

    elif args.dataset == 'ISBI2013':
        validation_set = ISBI2013(mydict['validation_datalist'], mydict['files'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],shuffle=False,num_workers=1, drop_last=False)
        net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=1)

    elif args.dataset == 'DRIVE':
        validation_set = DRIVE(mydict['validation_datalist'], mydict['files'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],shuffle=False,num_workers=1, drop_last=False)
        net = ProbabilisticUnet(input_channels=3, num_classes=1, num_filters=[32,64,128,192], latent_dim=1)

    else:
        print ('Wrong dataloader!')

    if mydict['checkpoint_restore'] != "":
        checkpoint = torch.load(mydict['checkpoint_restore'] )
        net.load_state_dict(checkpoint['state_dict'])
    else:
        print("No model found!")
        sys.exit()

    validation_start_time = time()
    with torch.no_grad():
        net.eval()
        validation_iterator = iter(validation_generator)
        for i in range(len(validation_generator)):
            x, y_gt = next(validation_iterator)
            x = x.to(device, non_blocking=True)
            y_gt = y_gt.to(device, non_blocking=True)

            net.forward(x, segm=None, training=False)

            # num_preds = 4
            # predictions = []
            # for i in range(num_preds):
            #     mask_pred = net.sample(testing=True, use_prior_mean = False)
            #     mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()
            #     mask_pred = torch.squeeze(mask_pred, 0)
            #     predictions.append(mask_pred)
            # predictions = torch.cat(predictions, 0)

            if args.dataset == 'CREMI' or args.dataset == 'ISBI2013':
                prior, mask_pred = net.sample(testing=True)
            elif args.dataset == 'DRIVE':
                prior, mask_pred = net.sample_vessel(testing=True)
                
            # mask_pred = torch.sigmoid(mask_pred) 
            # mask_pred_binary = (mask_pred > 0.5).float()
            # mask_pred_binary = torch.squeeze(mask_pred_binary, 0)
            mask_pred_binary = np.squeeze(mask_pred, 0)
            for j in range(x.shape[0]):
                save_image(x[j,:,:,:], os.path.join(mydict['output_folder'], 'img' + str(mydict['validation_batch_size'] * i + j) + '.png'))
                save_image(y_gt[j,:,:,:], os.path.join(mydict['output_folder'], 'img' + str(mydict['validation_batch_size'] * i + j) + '_gt.png'))
                # save_image(mask_pred[j,:,:].float(), os.path.join(mydict['output_folder'], 'img' + str(mydict['validation_batch_size'] * i + j) + '_pred.png'))
                save_image(torch.from_numpy(mask_pred_binary[j,:,:]), os.path.join(mydict['output_folder'], 'img' + str(mydict['validation_batch_size'] * i + j) + '_pred_binary.png'))