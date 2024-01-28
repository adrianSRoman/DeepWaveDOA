from __future__ import print_function
import argparse
from math import log10
import matplotlib.pyplot as plt

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from cdbpn import Net as CDBPN
from data import get_training_set
import pdb
import socket
import time

import numpy as np
from dataset import DatasetFromFolderEval, DatasetFromFolder


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=32, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_train_path', type=str, default='data/metu_train.hdf', help='Train HDF file path')
parser.add_argument('--data_eval_path', type=str, default='data/metu_eval.hdf', help='Eval HDF file path')
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--hr_train_dataset', type=str, default='DIV2K_train_HR')
parser.add_argument('--model_type', type=str, default='CDBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='cdbpn', help='Location to save checkpoint models')

use_cuda = 1
device = torch.device("cuda" if use_cuda else "cpu")

opt = parser.parse_args()
gpus_list = range(opt.gpus)

hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        inp, target, bicubic = torch.tensor(batch[0]).to(device).cfloat(), torch.tensor(batch[1]).to(device).cfloat(), torch.tensor(batch[2]).to(device).float() 
        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(inp.real.double(), inp.imag.double())

        if opt.residual:
            prediction = prediction

        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        # print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))
    
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)), " lr:", optimizer.param_groups[0]['lr'])
    return epoch_loss / len(training_data_loader) # return average loss


def test():
    avg_psnr = 0
    avg_mse = 0
    pred, target = None, None
    for batch in testing_data_loader:
        inp, target = torch.tensor(batch[0]).to(device).cfloat(), torch.tensor(batch[1]).to(device).cfloat()
        prediction = model(inp.real.double(), inp.imag.double())
        mse = criterion(prediction, target)
        avg_mse += mse.item()
        psnr = 20 * log10(1 / mse.item())
        avg_psnr += psnr
    label = np.mean(np.abs(target[0].cpu().detach().numpy()), axis=0)
    pred = np.mean(np.abs(prediction[0].cpu().detach().numpy()), axis=0)
    plt.figure()
    pixel_plot1 = plt.imshow(label, interpolation='nearest')
    plt.savefig("figures/pixel_label.png")
    plt.figure()
    pixel_plot2 = plt.imshow(pred, interpolation='nearest')
    plt.savefig("figures/pixel_pred.png")
    print("===> Avg. PSNR: {:.4f} dB - Avg. MSE: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_mse/len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader), avg_mse/len(testing_data_loader) # return avg PSNR, avg MSE

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    model_out_path = opt.save_folder+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
use_cuda = torch.cuda.is_available()
print(use_cuda)
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')

dataset_tr_file = opt.data_train_path
dataset_train = DatasetFromFolder(dataset_tr_file)

dataset_val_file = opt.data_eval_path
dataset_val = DatasetFromFolder(dataset_val_file)

training_data_loader = DataLoader(dataset=dataset_train, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=dataset_val, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True) # no shuffle, batch=1

#combined_dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_val])
#dataset_size = len(combined_dataset)
#test_split = 0.1
#test_size = int(test_split * dataset_size)
#train_size = dataset_size - test_size
#train_dataset, test_dataset = random_split(combined_dataset,[train_size, test_size])

training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_dataset, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True) # no shuffle, batch=1


print('===> Building model ', opt.model_type)
model = CDBPN(num_channels=9, base_filter=32,  feat = 128, num_stages=10, scale_factor=opt.upscale_factor) 
model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

best_metric = float('inf') 
epoch_tr_loss = []
epoch_avg_test_mse = []
epoch_avg_test_psnr = []
for epoch in range(opt.start_iter, opt.nEpochs + 1):
    avg_tr_loss = train(epoch)

    # learning rate is decayed by a factor of 5 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 5.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
    epoch_tr_loss.append(avg_tr_loss.cpu().detach().numpy())

    avg_test_psnr, avg_test_mse = test()
    epoch_avg_test_mse.append(avg_test_mse)
    epoch_avg_test_psnr.append(avg_test_psnr)
#     if avg_test_mse < best_metric:  # Replace "<" with ">" if using PSNR and looking for the highest value
#         best_metric = avg_test_mse  # Update best metric
#         best_epoch = epoch  # Record the epoch where the best metric is achieved
# if best_epoch > 0:
#     checkpoint(best_epoch)


## Uncomment for plotting loss 
#plt.figure()
#plt.plot(epoch_tr_loss, label='train_loss')
#plt.plot(epoch_avg_test_mse,label='val_loss')
#plt.legend()
#plt.savefig("./figures/avg_loss_curve.png")

#plt.figure()
#plt.plot(epoch_avg_test_psnr, label='Avg. val PSNR/epoch')
## plt.plot(epoch_avg_test_mse,label='Avg. val MSE/epoch')
#plt.legend()
#plt.savefig("./figures/avg_psnr_mse_curve.png")
