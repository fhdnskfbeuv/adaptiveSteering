from .llm_config import cfg

from sklearn.linear_model import LogisticRegression as skLR

import torch.nn as nn
import torch
# from cuml.common.device_selection import using_device_type
from cuml.linear_model import LogisticRegression

class LayerClassifier:
	def __init__(self, penalty='l2', max_iter: int = 10000, useGPU=False):
		# self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.wNorm = None
		pArgs = penalty.split(' ')
		C = 1.0
		if len(pArgs) == 2:
			C = 1 / float(pArgs[1])
		pType = pArgs[0]
		if useGPU:
			self.linear = LogisticRegression(solver="qn", max_iter=max_iter,
											 # class_weight='balanced',
											 output_type='numpy', penalty=pType, C=C, fit_intercept=True)
			self.linear.solver_model.warm_start = True
			self.linear.verbose = 0
		else:
			self.linear = skLR(solver="saga", max_iter=max_iter,
							   # class_weight='balanced',
							   warm_start=True,
							   n_jobs=16, penalty=pType, C=C, fit_intercept=True)
			self.linear.verbose = 0
		self.direction = True
		self.w, self.b = None, None

	def setDirection(self, direct: bool):
		if self.direction == direct:
			return self.direction
		# print(f'Switch direction to {direct}')
		self.direction = direct
		return self.direction

	def train(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor, sampleWeight=None) -> list[float]:
		self.setDirection(True)
		X = torch.vstack([pos_tensor, neg_tensor])
		y = torch.cat((torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0))))
		self.linear.fit(X.cpu().numpy(), y.cpu().numpy(), sample_weight=sampleWeight)
		self.w, self.b = None, None
		return []

	def predict(self, tensor: torch.tensor) -> torch.tensor:
		classes = self.linear.predict(tensor.cpu().numpy())
		if self.direction == False:
			classes = 1 - classes
		return torch.tensor(classes, dtype=torch.int64)

	def predict_proba(self, tensor: torch.tensor) -> torch.tensor:
		w, b = self.get_weights_bias()
		score = tensor @ w.T.to(tensor) + b.to(tensor)
		return torch.sigmoid(score)

	def getLogit(self, tensor: torch.tensor) -> torch.tensor:
		w, b = self.get_weights_bias()
		score = tensor @ w.T.to(tensor) + b.to(tensor)
		return score

	def evaluate_testacc(self, pos_tensor: torch.tensor, neg_tensor: torch.tensor):
		test_data = torch.vstack([pos_tensor, neg_tensor])
		predictions = self.predict(test_data)
		true_labels = torch.cat((torch.ones(pos_tensor.size(0)), torch.zeros(neg_tensor.size(0))))
		pred = predictions > 0.5
		correct_count = torch.sum(pred == true_labels).item()

		TP = ((predictions == 1) & (true_labels == 1)).sum().item()
		TN = ((predictions == 0) & (true_labels == 0)).sum().item()
		FP = ((predictions == 1) & (true_labels == 0)).sum().item()
		FN = ((predictions == 0) & (true_labels == 1)).sum().item()

		accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
		recall = TP / (TP + FN) if (TP + FN) != 0 else 0
		precision = TP / (TP + FP) if (TP + FP) != 0 else 0
		f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

		# self.data["test"]["pos"] = pos_tensor.cpu()
		# self.data["test"]["neg"] = neg_tensor.cpu()

		return {
			"Accuracy": accuracy,
			"Recall": recall,
			"Precision": precision,
			"F1 Score": f1_score
		}

	def get_weights_bias(self) -> tuple[torch.tensor, torch.tensor]:
		if self.w is None:
			self.w, self.b = torch.tensor(self.linear.coef_).cuda(), torch.tensor(self.linear.intercept_).cuda()
		if self.wNorm is None:
			self.wNorm = torch.norm(self.w, dim=-1).cuda()
			self.wNorm[self.wNorm == 0] = 1e-6
		if self.direction == False:
			return -self.w, -self.b
		return self.w, self.b

