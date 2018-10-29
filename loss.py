import torch
import torch.nn
import numpy as np
from Model import Generator, Discriminator


class AD_Loss(nn.Module):
	'''
	def __init__(self, overall_loss_weight=0.5):
		super(GeneratorLoss, self).__init__():
		#self.alpha_loss = AlphaLoss()
		#self.comp_loss = CompositionalLoss()
		self.l_weight = overall_loss_weight
	'''
	def forward(self, out_labels):
		#alpha_diff = self.alpha_loss(p_alpha, g_alpha)
		#comp_diff = self.comp_loss(p_image, g_image)
		#overall_loss = self.l_weight * alpha_diff + (1 - self.l_weight) * comp_diff
		adversarial_loss = torch.mean(1 - out_labels)
		return adversarial_loss #alpha_diff + comp_diff#, overall_loss

'''

class AlphaLoss(nn.Module):
	def __init__(self, alpha_small_value = 1e-6):
		super(AlphaLoss, self).__init__():
		self.small_value = alpha_small_value

	def forward(self, predict_alpha, ground_truth):
		alpha_diff = torch.sqrt((predict_alpha - ground_truth) ** 2 + self.small_value)
		return alpha_diff


class CompositionalLoss(nn.Module):
	def __init__(self, comp_small_value):
		super(CompositionalLoss, self).__init__():
		self.small_value = comp_small_value

	def forward(self, predict_image, ground_truth):
		comp_diff = torch.sqrt((predict_image - ground_truth) ** 2 + self.small_value)
		return comp_diff

'''


		