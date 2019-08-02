"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: train.py
about: main entrance for training the GridDehazeNet
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data import TrainData
from val_data import ValData
from model import GateHazeNet
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
plt.switch_backend('agg')


# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-3, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[240, 240], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=18, type=int)
parser.add_argument('-network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('-network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('-num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('-growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-category', help='Set image category (indoor or outdoor?)', default='indoor', type=str)
args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
category = args.category

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\n'
      'num_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
if category == 'indoor':
    num_epochs = 100
    train_data_dir = './data/train/indoor/'
    val_data_dir = './data/test/SOTS/indoor/'
elif category == 'outdoor':
    num_epochs = 10
    train_data_dir = './data/train/outdoor/'
    val_data_dir = './data/test/SOTS/outdoor/'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')


# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = GateHazeNet(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False

loss_network = LossNetwork(vgg_model)
loss_network.eval()


# --- Load the network weight --- #
try:
    net.load_state_dict(torch.load('{}_haze_best_{}_{}'.format(category, network_height, network_width)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))


# --- Load training data and validation/test data --- #
train_data_loader = DataLoader(TrainData(crop_size, train_data_dir), batch_size=train_batch_size, shuffle=True, num_workers=24)
val_data_loader = DataLoader(ValData(val_data_dir), batch_size=val_batch_size, shuffle=False, num_workers=24)


# --- Previous PSNR and SSIM in testing --- #
old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, category)
print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr, old_val_ssim))

for epoch in range(num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, category=category)

    for batch_id, train_data in enumerate(train_data_loader):

        haze, gt = train_data
        haze = haze.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        dehaze = net(haze)

        smooth_loss = F.smooth_l1_loss(dehaze, gt)
        perceptual_loss = loss_network(dehaze, gt)
        loss = smooth_loss + lambda_loss*perceptual_loss

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))

        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), '{}_haze_{}_{}'.format(category, network_height, network_width))

    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr, val_ssim = validation(net, val_data_loader, device, category)
    one_epoch_time = time.time() - start_time
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, category)

    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(net.state_dict(), '{}_haze_best_{}_{}'.format(category, network_height, network_width))
        old_val_psnr = val_psnr
