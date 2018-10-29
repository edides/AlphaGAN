#''''''''''''''''''''''''''''''''''''#
# Version 1.0
# Author: Charlie Dai
#''''''''''''''''''''''''''''''''''''#

import os
from os import listdir
import cv2
from PIL import Image


#from PIL import image
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

import numpy as np
import random
from scipy import misc, ndimage
import copy
import gc
from sys import getrefcount
from Model import img_rows, img_cols

import math
import random
from random import shuffle
#np.set_printoptions(threshold='nan')

########################
#change alpha to (1,1,320,320)
# now it is (1,320,320)

unknown_area = 128
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))



def safe_crop(mat, x, y, crop_size=(img_rows, img_cols)):
	crop_height, crop_width = crop_size
	if len(mat.shape) == 2:
		ret = np.zeros((crop_height, crop_width), np.float32)
	else:
		ret = np.zeros((crop_height, crop_width, 3), np.float32)
	crop = mat[y:y+crop_height, x:x+crop_width] #This will prevent crop size out of bound
	#cv2.imshow('image', crop)
	h, w = crop.shape[:2]
	ret[0:h, 0:w] = crop

	if crop_size != (img_rows, img_cols):
		ret = misc.imresize(ret, [img_rows, img_cols], interp = 'nearest')
		#ret = cv2.resize(ret, dsize=(img_cols, img_rows), interpolation=cv2.INTER_NEAREST)
	return ret

def random_choice(trimap, crop_size=(320, 320)):
	crop_height, crop_width = crop_size

	#print(np.where(trimap == unknown_area))
	y_indices, x_indices, _ = np.where(trimap == unknown_area)
	#print(y_indices, x_indices)
	num_unknowns = len(y_indices)
	x, y  = 0, 0
	#print(y_indices, x_indices)
	if num_unknowns > 0:
		ix = np.random.choice(range(num_unknowns))
		center_x = x_indices[ix]
		center_y = y_indices[ix]
		x = max(0, center_x - int(crop_width/2))
		#print(x)
		#print(y)
		y = max(0, center_y - int(crop_height/2))
	return x, y

def generate_trimap(alpha, save_path):
	fg = np.array(np.equal(alpha, 255).astype(np.float32))
	# fg = cv.erode(fg, kernel, iterations=np.random.randint(1, 3))
	unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))
	unknown = cv2.dilate(unknown, kernel, iterations=np.random.randint(1, 20))
	trimap = fg * 255 + (unknown - fg) * 128
	trimap = trimap.astype(np.uint8)
	cv2.imwrite(save_path, trimap)
	return trimap


def load_alpha(root, save_path):
	dirs = os.walk(root)
	for roots, dirs, filenames in dirs:
		for filename in filenames:
			if filename.endswith('.jpg'):
				full_name = os.path.join(root, filename)
				print(full_name)
				alpha = cv2.imread(full_name)
				cv2.imshow('img', alpha)
				cv2.waitKey(2000)
				path = os.path.join(save_path, filename)
				generate_trimap(alpha, path)


def is_image_file(filename):
	'''
	Allow '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'
	'''
	return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class TrainDatasetFromFolder(Dataset):
	def __init__(self, alpha_path, trimap_path, gt_path):
		super(TrainDatasetFromFolder, self).__init__()
		self.alpha_filenames = [os.path.join(alpha_path, x) for x in listdir(alpha_path) if is_image_file(x)]
		self.trimap_path = [os.path.join(trimap_path, x) for x in listdir(trimap_path) if is_image_file(x)]
		self.gt_path = [os.path.join(gt_path, x) for x in listdir(gt_path) if is_image_file(x)]

	def __getitem__(self, index):

		#length = min(batch_size, (len(alpha_filenames) - index))
		different_sizes = [(320, 320), (480, 480), (640, 640), (720, 720)]
		crop_size = random.choice(different_sizes)


		'''
		alpha = Image.open(self.alpha_filenames[index])
		alpha = np.asarray(alpha)
		trimap = Image.open(self.trimap_path[index])
		trimap = np.asarray(trimap)
		gt = Image.open(self.gt_path[index])
		gt = np.asarray(gt)
		'''
		
		alpha = cv2.imread(self.alpha_filenames[index])
		trimap = cv2.imread(self.trimap_path[index])
		gt = cv2.imread(self.gt_path[index])
		
		x, y = random_choice(trimap, crop_size)
		trimap = safe_crop(gt, x, y, crop_size)
		alpha = safe_crop(alpha, x, y, crop_size)
		gt = safe_crop(gt, x, y, crop_size)

		alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY) / 255.
		trimap = cv2.cvtColor(trimap, cv2.COLOR_BGR2GRAY)

		input_image = np.empty((img_rows, img_cols, 4), dtype=np.float32)
		gt_lable = np.empty((img_rows, img_cols, 2), dtype=np.float32)

		input_image[:, :, 0:3] = gt
		input_image[:, :, 3] = alpha / 255.

		mask = np.equal(trimap, 128).astype(np.float32)
		gt_lable[:, :, 0] = alpha / 255.
		gt_lable[:, :, 1] = mask / 255.

		input_image = torch.from_numpy(input_image)
		gt_lable = torch.from_numpy(gt_lable)
		alpha = torch.from_numpy(alpha)
		trimap = torch.from_numpy(trimap)
		gt = torch.from_numpy(gt)
		#print('******************x5', trimap.size())


		return input_image, gt_lable, alpha, trimap, gt

	def __len__(self):
		return len(self.alpha_filenames)

if __name__ == '__main__':
	#path = 'D:\\Desktop\\projects\\AlphaGAN\\AdobeOriginalImages\\Training_set\\Adobe_licensed_images\\alpha'
	#save_path = 'D:\\Desktop\\projects\\AlphaGAN\\AdobeOriginalImages\\Training_set\\Adobe_licensed_images\\trimap'
	#alpha = cv2.imread(path)
	#load_alpha(path, save_path)
	#cv2.imshow('img', img)
	#cv2.waitKey(2000)
	path = '.\\Training_set\\fg\\1-1252426161dfXY.jpg'
	path1 = '.\\Training_set\\trimap\\1-1252426161dfXY.jpg'
	path2 = '.\\Training_set\\alpha\\1-1252426161dfXY.jpg'
	input_image = np.empty((1, 320, 320, 4), dtype=np.float32)
	#a = np.array(3)
	#print(input_image)
	alpha = cv2.imread(path2)
	mat = cv2.imread(path)
	trimap = cv2.imread(path1)
	
	x, y = random_choice(trimap, crop_size=(320, 320))

	alpha = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
	trimap = cv2.cvtColor(trimap, cv2.COLOR_BGR2GRAY)

	
	mat = safe_crop(mat, x, y, crop_size=(320, 320))
	alpha = safe_crop(alpha, x, y, crop_size=(320, 320))
	trimap = safe_crop(trimap, x, y, crop_size=(320, 320))
	#print(mat)
	input_image[0, :, :, 0:3] = mat
	input_image[0, :, :, 3] = alpha
	print(input_image)
	#cv2.imshow('img1', trimap)
	#trimap = cv2.cvtColor(trimap, cv2.COLOR_BGR2GRAY)
	#print(trimap)
	#print('alpha', alpha.shape)
	#random_choice(trimap, crop_size=(320, 320))
	#img = save_crop(mat, 205, 307, crop_size=(1280, 1280))
	cv2.imshow('img', input_image)
	cv2.waitKey(2000)

