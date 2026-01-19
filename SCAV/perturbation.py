from .classifier_manager import ClassifierManager
import torch


class Perturbation:
	def __init__(self, classifier_manager: ClassifierManager, target_probability):
		self.classifier_manager = classifier_manager
		self.target_probability = target_probability if isinstance(target_probability, list) else [target_probability] * len(self.classifier_manager.classifiers)

	def setPT(self, pt):
		self.target_probability = pt if isinstance(pt, list) else [float(pt)] * len(self.classifier_manager.classifiers)

	def get_perturbation(self, output: torch.Tensor, layer: int, posi='response'):
		assert posi in ['response', 'prompt', 'last', 'all']
		if len(output[0].shape) == 2:  # Strange! Qwen3!
			if posi in ['prompt', 'last'] and output[0].shape[0] == 1:
				return output
			if posi in ['prompt', 'all']:
				output[0][:, :] = self.classifier_manager.cal_perturbation(
					embds_tensor=output[0][:, :],
					layer=layer,
					target_prob=self.target_probability[layer]
				)
			else:
				output[0][-1, :] = self.classifier_manager.cal_perturbation(
					embds_tensor=output[0][-1, :],
					layer=layer,
					target_prob=self.target_probability[layer]
				)
		else:  # Regular
			if posi in ['prompt', 'last'] and output[0].shape[1] == 1:
				return output
			if posi in ['prompt', 'all']:
				output[0][:, :, :] = self.classifier_manager.cal_perturbation(
					embds_tensor=output[0][:, :, :],
					layer=layer,
					target_prob=self.target_probability[layer]
				)
			else:
				output[0][:, -1, :] = self.classifier_manager.cal_perturbation(
					embds_tensor=output[0][:, -1, :],
					layer=layer,
					target_prob=self.target_probability[layer]
				)
		return output
