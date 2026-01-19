import torch
import os


class EmbeddingManager:
	def __init__(self):
		self.layers = []
		self.variedLenLayers = []
		# self.allEmbds = None
	
	def isEmpty(self):
		if len(self.variedLenLayers) == 0:
			return True
		for i in range(len(self.variedLenLayers)):
			if len(self.variedLenLayers[i]) > 0:
				return False
		return True

	def getNorm(self):
		allLayersNorm = []
		self.trainMode()
		for i in range(len(self.layers)):
			norm = torch.norm(self.layers[i], dim=-1)
			allLayersNorm.append((torch.mean(norm).item(), torch.min(norm).item(), torch.max(norm).item()))
		return allLayersNorm

	def trainMode(self):  # List[List[Depth, D]]  ->  List[B * Depth, D]
		for i in range(len(self.variedLenLayers)):
			if i >= len(self.layers):
				self.layers.append(None)
			if len(self.variedLenLayers[i]) > 0:
				self.layers[i] = torch.concat(self.variedLenLayers[i], dim=0)
		# self.allEmbds = torch.concat(self.layers if len(goodLayer) <= 0 else [self.layers[i] for i in goodLayer], dim=0)


def load_embds_manager(file_path: str) -> EmbeddingManager:
	return torch.load(file_path, weights_only=False)


def merge_tensor(tensors: list) -> torch.tensor:
	return torch.cat(tensors, dim=0)
