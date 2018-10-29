#''''''''''''''''''''''''''''''''''''#
# Version 1.0
# Author: Charlie Dai
# Only for ResNet50 and output_stride = 8 
# To-Do:
# Add Validation Function
#''''''''''''''''''''''''''''''''''''#
# Issue: Need to set batch size to be 1 if we apply random crop here
# Pytorch DataLoader enforce this feature

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader

import argparse
import time
import pandas as pd
import os
from tqdm import tqdm
import psutil
import gc
import math

from Model import Generator, Discriminator
from utils_1 import TrainDatasetFromFolder

INPUT_CHANNEL = 4

alpha_path = './Training_set/alpha'
trimap_path = './Training_set/trimap'
gt_path = './Training_set/fg'

val_alpha_path = './Test_set/alpha'
val_trimap_path = './Test_set/trimap'
val_gt_path = './Test_set/fg'

class AD_Loss(nn.Module):

	def forward(self, out_labels):
		adversarial_loss = torch.mean(1 - out_labels)
		return adversarial_loss #alpha_diff + comp_diff#, overall_loss



parser = argparse.ArgumentParser(description='Train AlphaGAN model')

parser.add_argument('-l', dest='learning_rate', default=1e-4, type=float, help='Define learning rate')
parser.add_argument('-b', dest='batch_size', default=1, type=int, help='Batch size')
parser.add_argument('-e', dest='num_epoch', default=100, type=int, help='number of epoches')
#parser.add_argument('-resume', dest='resume', default=False, type=bool, help='indicate wheter assume training')


opt = parser.parse_args()
############################
# Environment Stats
#
############################
#num_gpus = torch.cuda.device_count()
#print("GPU counts: ", num_gpus)

def memReport():
	for obj in gc.get_objects():
		if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
			print(type(obj), obj.size())

def cpuStats():
	print(sys.version)
	print(psutil.cpu_percent())
	print(psutil.virtual_memory())
	pid = os.getpid()
	py = psutil.Process(pid)
	memoryUse = py.memory_info()[0] / 2. ** 30
	print('memory GB: ', memmoryUse)


'''
def comp_function(image):

	pass
'''
def main(alpha_path, trimap_path, gt_path, val_alpha_path, val_trimap_path, val_gt_path):

	LEARNING_RATE = opt.learning_rate
	BATCH_SIZE = opt.batch_size
	NUM_EPOCHS = opt.num_epoch
	# The reason why we use batch size equal to 1 is here
	#https://medium.com/@yvanscher/pytorch-tip-yielding-image-sizes-6a776eb4115b
	train_set = TrainDatasetFromFolder(alpha_path, trimap_path, gt_path)
	train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)
	#val_set = ValDatasetFromFolder(val_alpha_path, val_trimap_path, val_gt_path)
	#val_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=1, shuffle=True)
	

	in_channel = 4
	os = 8
	netG = Generator(in_channel, os)
	netD = Discriminator()

	ad_loss = AD_Loss()

	if torch.cuda.is_available:
		print('Cuda!!!!')
		netG.cuda()
		netD.cuda()
		#generator_criterion.cuda()

	optimizerG = optim.Adam(netG.parameters(), lr=opt.learning_rate)
	optimizerD = optim.Adam(netD.parameters(), lr=opt.learning_rate)

	# Need Evaluating data
	results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': []}

	for epoch in range(1, NUM_EPOCHS + 1):
		train_bar = tqdm(train_loader) # batch_RGBsT, batch_trimapsT, batch_alphasT, batch_BGsT, batch_FGsT, RGBs_with_meanT

		running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
		# one train_bar is one batch
		a = 0.
		b=0.
		for g_input, g_lable, g_alpha,  g_trimap, g_image, in train_bar:

			#print('g_input', g_input.size())
			#print('g_lable', g_lable.size())
			#print('g_alpha', g_alpha.size())
			#print('g_image', g_image.size())
			#print('g_trimap', g_trimap.size())

			g_input = g_input.permute(0, 3, 1, 2)
			g_lable = g_lable.permute(0, 3, 1, 2)
			#g_alpha = g_alpha.permute(0, 3, 1, 2)
			g_image = g_image.permute(0, 3, 1, 2)
			#g_trimap = g_trimap.permute(0, 3, 1, 2)

			batch_size = opt.batch_size
			running_results['batch_sizes'] += batch_size

			g_alpha = Variable(g_alpha) 
			g_image = Variable(g_image) 
			g_trimap = Variable(g_trimap)
			g_lable = Variable(g_lable) # image label
			g_input = Variable(g_input) # image data

			############################
			# (1) Update D network: maximize D(x)-1-D(G(z))
			###########################

			real_image = Variable(g_alpha)
			real_image = real_image.float()
			real_image = real_image.unsqueeze_(-1)
			real_image = real_image.permute(0, 3, 1, 2)

			
			if torch.cuda.is_available():
				g_alpha = g_alpha.cuda()
				g_image = g_image.cuda()
				g_trimap = g_trimap.cuda()
				g_input = g_input.cuda()
				g_lable = g_lable.cuda()
				real_image = real_image.cuda()

			combine_loss, _pred_rgb, fake_image = netG(g_input, g_alpha, g_trimap, g_image)
			
			netD.zero_grad()
			real_out = netD(real_image).mean()
			fake_out = netD(fake_image).mean()
			d_loss = 1 - real_out + fake_out
			d_loss.backward(retain_graph=True)
			optimizerD.step()

			#############################
			#(2) Update G network: minimize Compsite Loss and alpha loss
			# 	 Need to double check the loss update
			#############################

			netG.zero_grad()
			combined_loss, p_rgb, _ = netG(g_input, g_alpha, g_trimap, g_image) # generator generate an imperfect alpha
			#optimizerG.step()
			#combined_loss, alpha_loss, p_rgb, _ = netG(g_input, g_alpha, g_trimap, g_image)
			combine_loss.backward()
			optimizerG.step()
			combine_loss, _pred_rgb, fake_image = netG(g_input, g_alpha, g_trimap, g_image)
			fake_out = netD(fake_image).mean()

			d_loss = 1 - real_out + fake_out

			#############################
			# Update result here
			# 
			#############################

			running_results['d_loss'] += d_loss.item() * batch_size

			if math.isnan(combined_loss.item()) is not True:
				#print('#'*10, alpha_loss.item())
				a = combined_loss.item() * batch_size
				#print('#'*10, a)
				running_results['g_loss'] += a
				b = b + a
				#print('#'*10, b)

			running_results['d_score'] += real_out.item() * batch_size
			running_results['g_score'] += fake_out.item() * batch_size

			train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
									epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
									running_results['g_loss'] / running_results['batch_sizes'],
									running_results['d_score'] / running_results['batch_sizes'],
									running_results['g_score'] / running_results['batch_sizes']))

		epoch_path = 'epochs_den_rc/'


if __name__ == '__main__':

	pretrained_model = False
	main(alpha_path, trimap_path, gt_path, val_alpha_path, val_trimap_path, val_gt_path)
