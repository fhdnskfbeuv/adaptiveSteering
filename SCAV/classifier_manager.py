import torch

from .embedding_manager import EmbeddingManager
from .layer_classifier import LayerClassifier


class ClassifierManager:
	def __init__(self, classifier_type: str, useGPU):
		self.trainacc = []
		self.type = classifier_type
		self.classifiers = []
		self.testacc = []
		self.useGPU = useGPU
		self.completion = []
		self.seen = None

	def setDirection(self, direct: bool):
		for i in range(len(self.classifiers)):
			self.classifiers[i].setDirection(direct)

	def _train_classifiers(
		self,
		pos_embds: EmbeddingManager,
		neg_embds: EmbeddingManager,
		penalty='l2',
		sampleWeight=None,
	):
		for i in range(len(pos_embds.layers)):
			if i >= len(self.classifiers):
				layer_classifier = LayerClassifier(penalty=penalty, useGPU=self.useGPU)
				layer_classifier.train(
					pos_tensor=pos_embds.layers[i],
					neg_tensor=neg_embds.layers[i],
					sampleWeight=sampleWeight[i] if sampleWeight is not None else None,
				)
				self.classifiers.append(layer_classifier)
			else:
				layer_classifier = self.classifiers[i]
				layer_classifier.setDirection(True)
				layer_classifier.train(
					pos_tensor=pos_embds.layers[i],
					neg_tensor=neg_embds.layers[i],
					sampleWeight=sampleWeight[i] if sampleWeight is not None else None,
				)

	def evaluate(self, pos_embds: EmbeddingManager, neg_embds: EmbeddingManager):
		allAcc = []
		for i in range(len(self.classifiers)):
			self.classifiers[i].setDirection(True)
			allAcc.append(
				self.classifiers[i].evaluate_testacc(
					pos_tensor=pos_embds.layers[i],
					neg_tensor=neg_embds.layers[i],
				)
				# if not big else self.bigClfr.evaluate_testacc(
				# 	pos_tensor=pos_embds.layers[i],
				# 	neg_tensor=neg_embds.layers[i],
				# )
			)
		return allAcc

	def fit(
		self,
		pos_embds_train: EmbeddingManager,
		neg_embds_train: EmbeddingManager,
		pos_embds_test: EmbeddingManager,
		neg_embds_test: EmbeddingManager,
		sampleWeight=None,
		penalty='l2',
	):
		pos_embds_train.trainMode()
		neg_embds_train.trainMode()
		pos_embds_test.trainMode()
		neg_embds_test.trainMode()
		self._train_classifiers(
			pos_embds_train,
			neg_embds_train,
			penalty,
			sampleWeight,
		)

		self.testacc = self.evaluate(
			pos_embds_test,
			neg_embds_test,
		)
		self.trainacc = self.evaluate(
			pos_embds_train,
			neg_embds_train,
		)

		for i in range(len(self.classifiers)):
			self.classifiers[i].w, self.classifiers[i].b = None, None

		return self

	def predictProba(self, layer, embd, testortrain):
		# if 'big' not in testortrain.lower():
		return self.classifiers[layer].predict_proba(embd)

	# return self.bigClfr.predict_proba(embd)

	def save(self, path: str):
		torch.save(self, path)

	def cal_perturbation(
		self,
		embds_tensor: torch.tensor,  # [..., D]
		layer: int,
		target_prob: float,
	):
		w, b = self.classifiers[layer].get_weights_bias()  # if 'big' not in testortrain.lower() else self.bigClfr.get_weights_bias()
		w = w.to(embds_tensor)
		b = b.to(embds_tensor)
		w_norm = self.classifiers[layer].wNorm.to(embds_tensor)
		logit_target = torch.log(torch.tensor(target_prob / (1 - target_prob)))
		logit = b + torch.sum(embds_tensor * w, dim=-1)
		epsilon = (logit_target - logit) / w_norm
		zl = torch.zeros_like(epsilon)
		epsilon = torch.clip(epsilon, max=zl, min=zl - float('inf')).unsqueeze(-1)
		perturbation = epsilon * w / w_norm

		return embds_tensor + perturbation


def load_classifier_manager(file_path: str):
	return torch.load(file_path, weights_only=False, map_location='cpu')
