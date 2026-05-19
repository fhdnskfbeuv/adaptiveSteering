import functools
import os
import pickle

import numpy
import torch
import numpy as np
from peft import PeftModel

adaPath = {
	"llama": './AdaSteer/vectors/llama31-8b-instruct',
	"gemma": './AdaSteer/vectors/gemma2-9b-it',
	"qwen": './AdaSteer/vectors/qwen25-7b-instruct',
}


def load_from_pickle(file_path):
	if not os.path.exists(file_path):
		return None
	# logging.info(f"Loading hidden states from {file_path}")
	with open(file_path, 'rb') as f:
		return pickle.load(f)


def register_ada_hooks(model, modelName):
	vectorPath = adaPath[modelName]
	retHooks = []
	baseModel = model.model if not isinstance(model, PeftModel) else model.base_model.model.model
	if hasattr(baseModel, 'language_model'):
		baseModel = baseModel.language_model
	aaa = adaHook(modelName, vectorPath)
	for i in range(len(baseModel.layers)):
		retHooks.append(baseModel.layers[i].register_forward_hook(
			functools.partial(aaa.forward, layerIdx=i)
		))
	return retHooks, aaa


class adaHook:
	def __init__(self, modelName, vectorPath):
		self.isRecord = True
		self.alpha = None
		self.beta = None
		
		if 'llama' in modelName:
			self.steer_vector = torch.from_numpy(load_from_pickle(os.path.join(vectorPath, 'RD', 'mean_diff.pkl'))).to(torch.float16).to("cuda")
			
			self.steer_vector_2 = torch.from_numpy(load_from_pickle(os.path.join(vectorPath, "HD", "proj.pkl"))).to(torch.float16).to("cuda")
			self.all_activations = []
			
			self.harmful_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "RD", "class_a.pkl")), axis=1)
			self.harmless_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "RD", "class_b.pkl")), axis=1)
			
			self.acceptance_direction = self.harmless_anchors - self.harmful_anchors
			self.acceptance_direction[13] /= np.linalg.norm(self.acceptance_direction[13])
			self.acceptance_direction[8] /= np.linalg.norm(self.acceptance_direction[8])
			
			self.pseudo_harmful_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "HD", "class_a.pkl")), axis=1)
			self.pseudo_harmless_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "HD", "class_b.pkl")), axis=1)
			
			self.pseudo_acceptance_direction = self.pseudo_harmless_anchors - self.pseudo_harmful_anchors
			self.pseudo_acceptance_direction[13] /= np.linalg.norm(self.pseudo_acceptance_direction[13])
			self.pseudo_acceptance_direction[8] /= np.linalg.norm(self.pseudo_acceptance_direction[8])
			
			self.standard = 100
			self.layerAlpha = 9
			self.layerBeta = 14
			# self.magicAlpha = [0.02, 60, 0.0, -0.22, -0.08]
			# self.magicBeta = [-0.05, 5, -0.5, -0.75, float('inf')]
			self.magicAlpha = [-0.02, 60, 0.0, 0.22, 0.08]
			self.magicBeta = [0.017, 14.70588, 0.0, -0.5, 0.25]
		elif 'gemma' in modelName:
			self.steer_vector = torch.from_numpy(load_from_pickle(os.path.join(vectorPath, 'RD', 'mean_diff.pkl'))).to(torch.float16).to("cuda")
			
			self.steer_vector_2 = torch.from_numpy(load_from_pickle(os.path.join(vectorPath, "HD", "proj.pkl"))).to(torch.float16).to("cuda")
			
			self.all_activations = []
			
			self.harmful_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "RD", "class_a.pkl")), axis=1)
			self.harmless_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "RD", "class_b.pkl")), axis=1)
			
			self.acceptance_direction = self.harmless_anchors - self.harmful_anchors
			self.acceptance_direction[19] /= np.linalg.norm(self.acceptance_direction[19])
			
			self.acceptance_direction[12] /= 563
			
			self.pseudo_harmful_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "HD", "class_a.pkl")), axis=1)
			self.pseudo_harmless_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "HD", "class_b.pkl")), axis=1)
			
			self.pseudo_acceptance_direction = self.pseudo_harmless_anchors - self.pseudo_harmful_anchors
			self.pseudo_acceptance_direction[19] /= np.linalg.norm(self.pseudo_acceptance_direction[19])
			self.pseudo_acceptance_direction[12] /= np.linalg.norm(self.pseudo_acceptance_direction[12])
			
			self.standard = 100
			
			self.layerAlpha = 13
			self.layerBeta = 20
			# self.magicAlpha = [0.004, -35, 0.0, -0.02, 0.06]
			# self.magicBeta = [0.01, -50, 0.0, -0.06, 0.02]
			self.magicAlpha = [-0.004, -35, 0.0, -0.2, 0.2]
			self.magicBeta = [0.01, -50, 0.0, -0.06, 0.02]
		elif 'qwen' in modelName:
			self.steer_vector = torch.from_numpy(load_from_pickle(os.path.join(vectorPath, 'RD', 'mean_diff.pkl'))).to(torch.float16).to("cuda")
			self.all_activations = []
			
			self.steer_vector_2 = torch.from_numpy(load_from_pickle(os.path.join(vectorPath, "HD", "proj.pkl"))).to(torch.float16).to("cuda")
			
			self.harmful_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "RD", "class_a.pkl")), axis=1)
			self.harmless_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "RD", "class_b.pkl")), axis=1)
			
			self.acceptance_direction = self.harmless_anchors - self.harmful_anchors
			self.acceptance_direction[13] /= np.linalg.norm(self.acceptance_direction[13])
			self.acceptance_direction[5] /= np.linalg.norm(self.acceptance_direction[5])
			
			self.pseudo_harmful_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "HD", "class_a.pkl")), axis=1)
			self.pseudo_harmless_anchors = np.mean(load_from_pickle(os.path.join(vectorPath, "HD", "class_b.pkl")), axis=1)
			
			self.pseudo_acceptance_direction = self.pseudo_harmless_anchors - self.pseudo_harmful_anchors
			self.pseudo_acceptance_direction[13] /= np.linalg.norm(self.pseudo_acceptance_direction[13])
			self.pseudo_acceptance_direction[5] /= np.linalg.norm(self.pseudo_acceptance_direction[5])
			
			self.standard = 100
			
			self.layerAlpha = 6
			self.layerBeta = 14
			# self.magicAlpha = [0.1, -140, 0.0, -0.2, 0.0]
			# self.magicBeta = [-0.06, -50, 0.0, -0.6, 0.4]
			self.magicAlpha = [-0.01, -140, 0.0, -0.2, 0.0]
			self.magicBeta = [-0.06, -50, 0.0, -0.6, 0.4]
		else:
			print(f'{modelName} not implemented!!!!!!!!!!!!!!')
			exit(1)
		
		
	
	def calAlpha(self, hd):
		layer_hs_last_cpu = hd.clone().detach().to(torch.float).cpu().numpy()
		dis_harmful = layer_hs_last_cpu - self.harmful_anchors[self.layerAlpha - 1]
		dis_harmless = layer_hs_last_cpu - self.harmless_anchors[self.layerAlpha - 1]
		
		harmful_dis_multi = np.dot(dis_harmful, self.acceptance_direction[self.layerAlpha - 1])
		harmless_dis_multi = np.dot(dis_harmless, self.acceptance_direction[self.layerAlpha - 1])
		
		scale = harmful_dis_multi - harmless_dis_multi
		scale = np.mean(scale)
		
		harmful_dis_multi = harmful_dis_multi / scale * self.standard
		# harmless_dis_multi = harmless_dis_multi / scale * self.standard
		# print(harmful_dis_multi)
		# print(scale)
		self.alpha = torch.from_numpy(numpy.array([self.magicAlpha[0] * (harmful_dis_multi + self.magicAlpha[1]) + self.magicAlpha[2]])).to(hd)
		self.alpha = torch.clamp(self.alpha, min=self.magicAlpha[3])
		self.alpha = torch.clamp(self.alpha, max=self.magicAlpha[4])
	
	def calBeta(self, hd):
		layer_hs_last_cpu = hd.clone().detach().to(torch.float).cpu().numpy()
		
		pseudo_dis_harmful = layer_hs_last_cpu - self.pseudo_harmful_anchors[self.layerBeta - 1]
		pseudo_dis_harmless = layer_hs_last_cpu - self.pseudo_harmless_anchors[self.layerBeta - 1]
		pseudo_harmful_dis_multi = np.dot(pseudo_dis_harmful, self.pseudo_acceptance_direction[self.layerBeta - 1])
		pseudo_harmless_dis_multi = np.dot(pseudo_dis_harmless, self.pseudo_acceptance_direction[self.layerBeta - 1])
		
		scale = pseudo_harmful_dis_multi - pseudo_harmless_dis_multi
		scale = np.mean(scale)
		
		pseudo_harmful_dis_multi = pseudo_harmful_dis_multi / scale * self.standard
		# print(pseudo_harmful_dis_multi)
		# print(scale)
		# pseudo_harmless_dis_multi = pseudo_harmless_dis_multi / scale * self.standard
		
		self.beta = torch.from_numpy(numpy.array([self.magicBeta[0] * (pseudo_harmful_dis_multi + self.magicBeta[1]) + self.magicBeta[2]])).to(hd)
		self.beta = torch.clamp(self.beta, min=self.magicBeta[3])
		self.beta = torch.clamp(self.beta, max=self.magicBeta[4])
	
	def reset(self):
		self.alpha = None
		self.beta = None
		self.isRecord = True
	
	def forward(self, module, inputs, outputs, layerIdx):
		if len(outputs[0].shape) == 2:
			_B = 1
			if outputs[0].shape[1] > 1:  # only prefilling
				if self.isRecord and layerIdx == self.layerAlpha - 1:
					self.calAlpha(outputs[0][-1, :])
				if self.isRecord and layerIdx == self.layerBeta - 1:
					self.calBeta(outputs[0][-1, :])
				if not self.isRecord:
					outputs[0][:, :] += (self.alpha[0:_B].unsqueeze(-1) * self.steer_vector[layerIdx].repeat(_B, 1))
					outputs[0][:, :] += (self.beta[0:_B].unsqueeze(-1) * self.steer_vector_2[layerIdx].repeat(_B, 1))
		else:
			_B = outputs[0].shape[0]
			if outputs[0].shape[1] > 1:  # only prefilling
				if self.isRecord and layerIdx == self.layerAlpha - 1:
					self.calAlpha(outputs[0][:, -1, :])
				if self.isRecord and layerIdx == self.layerBeta - 1:
					self.calBeta(outputs[0][:, -1, :])
				if not self.isRecord:
					outputs[0][:, :, :] += (self.alpha[0:_B].unsqueeze(-1).unsqueeze(-1) * self.steer_vector[layerIdx].unsqueeze(0).repeat(_B, 1).unsqueeze(1))
					outputs[0][:, :, :] += (self.beta[0:_B].unsqueeze(-1).unsqueeze(-1) * self.steer_vector_2[layerIdx].unsqueeze(0).repeat(_B, 1).unsqueeze(1))
		return outputs
