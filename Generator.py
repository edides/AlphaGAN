#''''''''''''''''''''''''''''''''''''#
# Initial Build
# Version 1.0
# Author: Charlie Dai
# 
# Shout out to:
# https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/networks/deeplab_resnet.py
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# Only for ResNet50 and output_stride = 8 
#''''''''''''''''''''''''''''''''''''#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


affine_par = True

class ResNet(nn.Module):
	'''
	Need to add os = 8 model!!!!!!!
	'''
	def __init__(self, input_channel=4, os=8, num_classes=1, ceil_mode=True, initial_mode=True, pretrained=False):
		
		if os == 16:
			strides = [1, 2, 2, 1]
			padding = [1, 1, 1, 2]
			blocks = [1, 2, 4]
		elif os == 8:
			strides = [1, 2, 1, 1]
			padding = [1, 1, 1, 2]
			blocks = [1, 2, 4]
		
		super(Discriminator, self).__init__()
		#input_size = 320
		self.inplanes = 64
		self.conv_1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3)#stride use to be 1

		self.relu_1 =  nn.LeakyReLU(0.2)

		# ceil_mode=False Odd image size will result lossing the last column and row
		# ceil_mode=True this will fill the margin for right siz4
		self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=ceil_mode) 
		self.conv_2 = self.reslayer(3, 64, stride=strides[0])
		self.conv_3 = self.reslayer(3, 128, stride=strides[1])
		self.conv_4 = self.reslayer(6, 256, stride=strides[2], dilation=2)
		self.conv_5 = self.reslayer(3, 512, stride=strides[3], dilation=4)

		if initial_mode:
			self._init_weight()

		if pretrained:
			self._load_pretrained_model()


		'''
		# layer 6 and 7 need to be changed
		# After the Resnet block 4, 
		# we add the atrous spatial pyramid pooling (ASPP) module
		'''
		#self.avgpool = nn.AvgPool2d(7, stride=1)
		#self.fc = nn.Linear(512 * 4, num_classes)


	def reslayer(self, num_of_blocks, channel, stride=1, dilation=1, expension=4):
		layers = []

		layers.append(Res50_BottleNeck(self.inplanes, channel, stride=stride, dilation, ds=True))
		self.inplanes = channel * expension

		for i in range(1, num_of_blocks):
			layers.append(Res50_BottleNeck(self.inplanes, channel, stride=1, dilation, ds=False))

		return nn.Sequential(*layers)

	'''
	def _make_MG_unit(self, expension=4, block, channel, blocks=[1,2,4], stride=1, rate=1):
		downsample = None
		if stride != 1 or self.inplanes != channel * expension:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, channel * block.expansion,
							kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(channel * block.expansion),
			)

		layers = []
		layers.append(Res50_BottleNeck(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, len(blocks)):
			layers.append(Res50_BottleNeck(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

		return nn.Sequential(*layers)
	'''
	def _init_weight(self):
		'''
		Weight initialization
		'''
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				# Modify the parameters by writing to weight.data
				# Also for bias
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _load_pretrained_model(self):
		pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
		model_dict = {}
		# Returns a dictionary containing a whole state of the module
		state_dict = self.state_dict()
		for k, v in pretrain_dict.items():
			if k in state_dict:
				model_dict[k] = v
		state_dict.update(model_dict)
		# Copies parameters and buffers from state_dict into this module and its descendants
		self.load_state_dict(state_dict)

	def forward(self, x):
		skip_connection1 = x
		out = self.conv_1(x)
		out = self.bn_1(out)
		out = self.relu_1(out)
		skip_connection2 = out
		out = self.pool_1(out)
		skip_connection3 = out
		out = self.conv_2(out)
		
		out = self.conv_3(out)
		out = self.conv_4(out)
		out = self.conv_5(out)

		#out = self.avgpool(out)
		#out = out.view(out.size(0), -1)

		return out, skip_connection1, skip_connection2, skip_connection3 #F.sigmoid(self.fc(out))



'''
# BottleNeck block
'''
class Res50_BottleNeck(nn.Module):
	#Variable that can access outside the class
	inplane_expansion = 4

	def __init__(self, in_channel, num_channels, stride, dilation, ds=False):
		super(Res50_BottleNeck, self).__init__()
		self.channel_count = num_channels
		self.downsamples = ds 
		

		self.conv1 = nn.Conv2d(in_channel, num_channels, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(num_channels, affine=affine_par)
		for i in self.bn1.parameters():
			i.requires_grad = False
		self.relu1 = nn.LeakyReLU(0.2)
		

		padding = dilation

		self.conv2 = nn.Conv2d(num_channels,num_channels, kernel_size=3,  stride=stride,,
			padding=padding, dilation=dilation, bias=False)
		self.bn2 = nn.BatchNorm2d(num_channels, affine=affine_par)
		for i in self.bn2.parameters():
			i.requires_grad = False
		self.relu2 = nn.LeakyReLU(0.2)
		

		self.conv3 = nn.Conv2d(num_channels, num_channels*4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(num_channels*4, affine=affine_par)
		for i in self.bn3.parameters():
			i.requires_grad = False
		self.relu3 = nn.LeakyReLU(0.2)
		self.relu4 = nn.ReLU(inplace=True)
		
		#downsample
		self.downsample = nn.Conv2d(in_channel, num_channels*4, kernel_size=1, stride=stride)#stride=2)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)

		out = self.conv3(out)
		out = self.bn3(out)
		out = self.relu3(out)

		if self.downsamples == True:
		    residual = self.downsample(x)
		    #print("downsample", self.channel_count*4)
		#else: residual = self.nonedownsample(x)
		out += residual

		out = self.relu4(out)

		return out



class ASPP_module(nn.Module):
	def __init__(self, in_planes, out_planes, rate, initial_mode=True):
		super(ASPP_module, self).__init__()
		if rate == 1:
			kernel_size = 1
			padding = 0
		else:
			kernel_size = 3
			padding = rate
		self.atrous_convolution = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
			sride=1, padding=padding, dilation=rate, bias=False)
		self.bn1 = nn.BatchNorm2d(out_planes)
		self.relu1 = nn.ReLU(inplanes=True)

		if initial_mode:
			self._init_weight()

	def forward(self, x):
		x = self.atrous_convolution(x)
		x = self.bn1(x)
		x = self.relu1(x)

		return x

	def _init__weight(self):
		for m in nn.module:
			if isinstance(m, nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
	'''
	DeepLab v3+ structure without decoder
	'''
	def __init__(self, input_channel=4, n_classes, initial_mode=True, pretrained=False, _print=True):
		if _print:
			print('Constructing DeepLab v3+ model in Encoder...')
			print('Number of classes: %d' % n_classes)
			print('Output stride: %d' % os)
			print('Number of Input Channels: %d' % input_channel)
		super(DeepLabv3_plus, self).__init__()

		# Atrous 
		# Need to add os options!!!!!!!!!!!!!!!!
		self.resnet_features = ResNet50(input_channel=input_channel, ceil_mode=True, initial_mode=initial_mode, pretrained=pretrained)

		#ASPP
		
		rates = [1, 6, 12, 18]

		'''
		elif os == 16:
			rates = [1, 12, 24, 36]
		'''
		else:
			raise NotImplementedError

		self.aspp1 = ASPP_model(2048, 256, rates[0], initial_mode=True)
		self.aspp2 = ASPP_model(2048, 256, rates[1], initial_mode=True)
		self.aspp3 = ASPP_model(2048, 256, rates[2], initial_mode=True)
		self.aspp4 = ASPP_model(2048, 256, rates[3], initial_mode=True)

		self.relu = nn.ReLU()

		self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
											nn.Conv2d(2048, 256, 1, stride=1, bias=False),
											nn.BatchNorm2d(256),
											nn.ReLU())

	def forward(self, x):
		# low level feature is the output from first resnet block
		x, skip_connection1, skip_connection2, skip_connection3 = self.resnet_features(x)
		x1 = self.aspp1(x) #output 256 channels
		x2 = self.aspp2(x) #output 256 channels
		x3 = self.aspp3(x) #output 256 channels
		x4 = self.aspp4(x) #output 256 channels
		x5 = self.global_avg_pool(x) # input 2048, output 256
		# in total would be 5*256=1280 channels
		# not sure if I need this upsample, assume I do 
		x5 = nn.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corner=True)

		x = torch.cat((x1, x2, x3, x4, x5), dim=1) # we will have 1280 channels here
		return x, skip_connection1, skip_connection2, skip_connection3


class Generator(nn.Module):
	'''
	# Add Decoder feature
	# in_channel should be the number of output channels from encoder
	'''
	def __init__(self, in_channel, out_channel, os, initial_mode, pretrained):
		super(Decoder, self).__init__()
		self.Encoder = Encoder(input_channel=4, n_classes, os, initial_mode, pretrained, _print=True)

		self.conv1 = nn.Conv2d(input_channel, 256, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(256)
		self.relu1 = nn.ReLU(inplanes=True)

		#use kernel=[1,1] and 48 channels for channel reduction
		self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
		self.bn2 = nn.BatchNorm2d(48)
		self.relu2 = nn.ReLU(inplanes=True)

		# conv 3x3 block - input channel should be 48 + 256 = 304
		self.conv_3x3_block1= nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
											nn.BatchNorm2d(256),
											nn.ReLU(),
											nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
											nn.BatchNorm2d(128),
											nn.ReLU(),
											nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False))

		self.conv3 = nn.Conv2d(64, 32, 1, bias=False)
		self.bn3 = nn.BatchNorm2d(32)
		self.relu3 = nn.ReLU(inplanes=True)	



		# Input should be 32 + 64 = 96
 		self.conv_3x3_block2 = nn.Sequential(Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
 								nn.BatchNorm2d(64),
 								nn.ReLU(inplanes=True),
 								nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
 								nn.BatchNorm2d(64),
 								nn.ReLU(inplanes=True),
 								nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False))

 		self.conv_3x3_block3 = nn.Sequential(Conv2d(36, 32, kernel_size=3, padding=1, bias=False),
 								nn.BatchNorm2d(32),
 								nn.ReLU(inplanes=True),
 								nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))



		self.last_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)
 								

	def forward(self, in_data, batch_alphasT, batch_trimapsT):
		x, skip_connection1, skip_connection2, skip_connection3 = self.Encoder(in_data)
	
		x = nn.upsample(x, size=(int(math.ceil(input.size()[-2]/4)), 
			int(math.ceil(input.size()[-1]/4))), mode='bilinear', align_corner=True)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu1(x)

		skip_connection3 = self.conv2(skip_connection3)
		skip_connection3 = self.bn2(skip_connection3)
		skip_connection3 = self.relu2(skip_connection3)

		x = torch.cat((x, skip_connection3), dim=1)

		x = self.conv_3x3_block1(x)
		# Input stride is 2, channel is 64
		skip_connection2 = self.conv3(skip_connection2)
		skip_connection2 = self.bn3(skip_connection2)
		skip_connection2 = self.relu3(skip_connection2)
		# Skip_connection2 32 channels and stide = 2
		# Input 64 channel stride = 4
		x = nn.upsample(x, size=(int(math.ceil(input.size()[-2]/2)), 
			int(math.ceil(input.size()[-1]/2))), mode='bilinear', align_corner=True)

		# Input x stride is 2, channel is 64. 
		# Skip_connection2 32 channels and stide = 2
		x = torch.cat((x, skip_connection2), dim=1)
		# Input x stride is 2, channel is 96
		x = self.conv_3x3_block2(x)
		# Input x stride is 2, channel is 32

		x = nn.upsample(x, size=(int(math.ceil(input.size()[-2])), 
			int(math.ceil(input.size()[-1]))), mode='bilinear', align_corner=True)
		# Input stride is 1, channel is 32
		# skip_connection1 stride is 1, channel is 4
		x = torch.cat((x, skip_connection1), dim=1)
		# Input x stride is 1, channel is 32
		x = self.conv_3x3_block3(x)
		# Input size is 32, output size is 1 
		x = F.sigmoid(self.last_conv(x))

class Discriminator(nn.Module):
	def __init__(self， input_nc=4, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
		super(Discriminator, self).__init__()
		if type(norm_layer) == functools.partial:
			use_bias = norm_layer.func == nn.InstanceNorm2d

		kw = 4
		padw = 1
		sequence = [
			nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
			nn.LeakyReLU(0.2, True)
			]

		nf_mult = 1
		nf_mult_prev = 1
		for n in range(1, n_layers):
			nf_mult_prev = nf_mult
			nf_mult = min(2**n, 8)
			sequence += [
				nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
						kernel_size=kw, stride=2, padding=pawd, bias=use_bias),
						norm_layer(ndf * nf_mult),
						nn.LeakyReLU(0.2, True)]

		nf_mult_prev = nf_mult
		nf_mult = min(2**n_layers, 8)
		sequence += [
			nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
				kernel_size=kw, stride=1, padding=pawd, bias=use_bias),
			norm_layer(ndf * nf_mult),
			nn.LeakyReLU(0.2, True)
			]
		sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride-1, padding=pawd)]

		if use_sigmoid:
			sequence += [nn.Sigmoid()]

		self.model = nn.Sequential(*sequence)

	def forward(self, input):
		return self.model(input)


#--------------------------------------------Functions------------------------------------------------------#

def ResNet50(input_channel=4, ceil_mode=True, initial_mode=True, pretrained=False):
	model = ResNet(input_channel, ceil_mode=ceil_mode, initial_mode=initial_mode, pretrained=pretrained)
	return model

def Encoder(input_channel=4, n_classes, os, initial_mode, pretrained, _print=True):
	model = DeepLabv3_plus(nput_channel=4, n_classes, os=os, pretrained=pretrained, _print=True)
	return model

if __name__ == '__main__':
	print('Hello')