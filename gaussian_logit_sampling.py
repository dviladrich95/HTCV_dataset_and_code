import torch
import torch.nn.functional as F
import torch.nn as nn


class GaussianLogitSampler(nn.Module):

	def __init__(self,**kwargs):
		super(GaussianLogitSampler, self).__init__(**kwargs)

	def forward(self, inputs):
# 		indix=K.shape(inputs)[-1]/2
# 		indix=K.cast(indix,dtype='int32')#some change here to get Integer index
		pred_m = inputs[:,:24,:,:]
		pred_v = F.relu(inputs[:,24:,:,:])
		pred_v = torch.exp(pred_v) - 1.0;
		gauss_pred_v=torch.randn(pred_v.shape).to(pred_v.device)
		pred_m = pred_m + torch.multiply(pred_v, gauss_pred_v)
		return pred_m