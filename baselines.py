import json
import os
from functools import partial

import torch
import tqdm

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.utils.hook_utils import add_hooks
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks
from vanillaSCAV.classifier_manager import ClassifierManager
from vanillaSCAV.embedding_manager import EmbeddingManager
from vanillaSCAV.perturbation import Perturbation, AllPerturbation
from vanillaSCAV.llm_config import cfg as CFG


class refusalDirection:
	def __init__(self, modelN, processor, model=None):
		self.ablation_fwd_pre_hooks = None
		self.actadd_fwd_pre_hooks = None
		self.actadd_fwd_hooks = None
		self.ablation_fwd_hooks = None
		model_alias = os.path.basename(modelN)
		self.cfg = Config(model_alias=model_alias, model_path=modelN)
		self.pos = None
		self.direction = None
		self.layer = None
		self.model = construct_model_base(self.cfg.model_path, model=model, tokenizer=processor)
		self.modelN = modelN
		self.processor = processor

	def generate_and_save_candidate_directions(self, cfg, model_base, harmful_train, harmless_train):
		"""Generate and save candidate directions."""
		if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
			os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

		mean_diffs = generate_directions(
			model_base,
			harmful_train,
			harmless_train,
			artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

		torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

		return mean_diffs

	def select_and_save_direction(self, cfg, model_base, harmful_val, harmless_val, candidate_directions):
		"""Select and save the direction."""
		if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
			os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))

		pos, layer, direction = select_direction(
			model_base,
			harmful_val,
			harmless_val,
			candidate_directions,
			artifact_dir=os.path.join(cfg.artifact_path(), "select_direction"), batch_size=1
		)

		with open(f'{cfg.artifact_path()}/direction_metadata.json', "w") as f:
			json.dump({"pos": pos, "layer": layer}, f, indent=4)

		torch.save(direction, f'{cfg.artifact_path()}/direction.pt')

		return pos, layer, direction

	def filter_data(self, cfg, model_base, harmful_train, harmless_train, harmful_val, harmless_val):
		"""
		Filter datasets based on refusal scores.

		Returns:
			Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
		"""

		def filter_examples(dataset, scores, threshold, comparison):
			return [inst for inst, score in zip(dataset, scores.tolist()) if comparison(score, threshold)]

		if cfg.filter_train:
			harmful_train_scores = get_refusal_scores(model_base.model, harmful_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, batch_size=1)
			harmless_train_scores = get_refusal_scores(model_base.model, harmless_train, model_base.tokenize_instructions_fn, model_base.refusal_toks, batch_size=1)
			harmful_train = filter_examples(harmful_train, harmful_train_scores, 0, lambda x, y: x > y)
			harmless_train = filter_examples(harmless_train, harmless_train_scores, 0, lambda x, y: x < y)

		if cfg.filter_val:
			harmful_val_scores = get_refusal_scores(model_base.model, harmful_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, batch_size=1)
			harmless_val_scores = get_refusal_scores(model_base.model, harmless_val, model_base.tokenize_instructions_fn, model_base.refusal_toks, batch_size=1)
			harmful_val = filter_examples(harmful_val, harmful_val_scores, 0, lambda x, y: x > y)
			harmless_val = filter_examples(harmless_val, harmless_val_scores, 0, lambda x, y: x < y)

		return harmful_train, harmless_train, harmful_val, harmless_val

	def getDirection(self, data):
		if isinstance(data, str):
			with open(os.path.join(data, 'direction_metadata.json'), "w") as f:
				self.pos, self.layer = json.load(f)['pos'], json.load(f)['layer']
			self.direction = torch.load(os.path.join(data, 'direction.pt'))
		else:
			# Load and sample datasets
			harmful_train, harmless_train, harmful_val, harmless_val = data[0], data[1], data[2], data[3]

			# Filter datasets based on refusal scores
			harmful_train, harmless_train, harmful_val, harmless_val = self.filter_data(self.cfg, self.model, harmful_train, harmless_train, harmful_val, harmless_val)

			# 1. Generate candidate refusal directions
			candidate_directions = self.generate_and_save_candidate_directions(self.cfg, self.model, harmful_train, harmless_train)
			try:
				# 2. Select the most effective refusal direction
				self.pos, self.layer, self.direction = self.select_and_save_direction(self.cfg, self.model, harmful_val, harmless_val, candidate_directions)
			except Exception as e:
				error_message = str(e)
				print(e)
				if 'All scores have been filtered out!' in error_message:
					return 'NOT MY FAULT!'

		return 'GOOD'

	def getHooks(self):
		self.ablation_fwd_pre_hooks, self.ablation_fwd_hooks = get_all_direction_ablation_hooks(self.model, self.direction)
		self.actadd_fwd_pre_hooks, self.actadd_fwd_hooks = [(self.model.model_block_modules[self.layer], get_activation_addition_input_pre_hook(vector=self.direction, coeff=-1.0))], []

	def ablationGen(self, prompts, maxL, doSample, rp=1.0):
		fullStrs = []
		completions = []
		with add_hooks(module_forward_pre_hooks=self.ablation_fwd_pre_hooks, module_forward_hooks=self.ablation_fwd_hooks):
			for prompt in tqdm.tqdm(prompts, total=len(prompts)):
				query = [
					{
						"role": "user",
						"content": prompt
					}
				]
				templatedQuery = self.processor.apply_chat_template(query,
																	tokenize=False,
																	add_generation_prompt=True)  # Prepare texts for processing
				# if prefix is not None:
				# 	templatedQuery += prefix
				inputs = self.processor(
					text=[templatedQuery], return_tensors="pt"
				).to(self.model.model.device)  # Encode texts and images into tensors
				# if prefix is not None:  # [1, L]
				# 	inputs['input_ids'] = torch.concat([inputs['input_ids'], prefix.repeat(inputs['input_ids'].shape[0], 1).to(inputs['input_ids'])], dim=1)
				# 	inputs['attention_mask'] = torch.concat([inputs['attention_mask'], torch.ones((inputs['attention_mask'].shape[0], prefix.shape[1])).to(inputs['attention_mask'])], dim=1)
				# inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in batchNoTarget.items()}
				# Inference: Generation of the output
				generated_ids = self.model.model.generate(**inputs, max_new_tokens=maxL, do_sample=doSample, repetition_penalty=rp)
				trimmedIDs = []
				for i in range(len(generated_ids)):
					trimmedIDs.append(generated_ids[i][inputs['input_ids'][i].shape[0]:])
				completeion = self.processor.batch_decode(
					trimmedIDs, skip_special_tokens=True, clean_up_tokenization_spaces=False
				)[0]
				fullStr = self.processor.batch_decode(
					generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
				)[0]
				fullStrs.append(fullStr)
				completions.append(completeion)
		return fullStrs, completions

	def actaddGen(self, prompts, maxL, doSample, rp=1.0):
		fullStrs = []
		completions = []
		with add_hooks(module_forward_pre_hooks=self.actadd_fwd_pre_hooks, module_forward_hooks=self.actadd_fwd_hooks):
			for prompt in tqdm.tqdm(prompts, total=len(prompts)):
				query = [
					{
						"role": "user",
						"content": prompt
					}
				]
				templatedQuery = self.processor.apply_chat_template(query,
																	tokenize=False,
																	add_generation_prompt=True)  # Prepare texts for processing
				# if prefix is not None:
				# 	templatedQuery += prefix
				inputs = self.processor(
					text=[templatedQuery], return_tensors="pt"
				).to(self.model.model.device)  # Encode texts and images into tensors
				# if prefix is not None:  # [1, L]
				# 	inputs['input_ids'] = torch.concat([inputs['input_ids'], prefix.repeat(inputs['input_ids'].shape[0], 1).to(inputs['input_ids'])], dim=1)
				# 	inputs['attention_mask'] = torch.concat([inputs['attention_mask'], torch.ones((inputs['attention_mask'].shape[0], prefix.shape[1])).to(inputs['attention_mask'])], dim=1)
				# inputs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in batchNoTarget.items()}
				# Inference: Generation of the output
				generated_ids = self.model.model.generate(**inputs, max_new_tokens=maxL, do_sample=doSample, repetition_penalty=rp)
				trimmedIDs = []
				for i in range(len(generated_ids)):
					trimmedIDs.append(generated_ids[i][inputs['input_ids'][i].shape[0]:])
				completeion = self.processor.batch_decode(
					trimmedIDs, skip_special_tokens=True, clean_up_tokenization_spaces=False
				)[0]
				fullStr = self.processor.batch_decode(
					generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
				)[0]
				fullStrs.append(fullStr)
				completions.append(completeion)
		return fullStrs, completions


def getMessages(texts, systemPrompt=None):
	messages = []
	for text in texts:
		if systemPrompt is not None:
			message = [{"role": "system", "content": systemPrompt}, {"role": "user", "content": text}]
		else:
			message = [{"role": "user", "content": text}]

		messages.append(message)
	return messages


class vanillaSCAV:
	def __init__(self, modelN, model, tokenizer, config):
		self.hooks = None
		self.pert = None
		self.model = model
		self.tokenizer = tokenizer
		self.config = config
		self.llm_cfg = CFG({
			'model_nickname': modelN,
			'model_name': modelN,
			'n_layer': config.num_hidden_layers,
			'n_dimension': config.hidden_size
		})

	def apply_sft_template(self, instruction, system_message=None):
		if system_message is not None:
			messages = [
				{
					"role": "system",
					"content": system_message
				},
				{
					"role": "user",
					"content": instruction
				}
			]
		else:
			messages = [
				{
					"role": "user",
					"content": instruction
				}
			]

		return messages

	def extract_embds(self, inputs: list[str], system_message: str = None, message: str = None) -> EmbeddingManager:
		embds_manager = EmbeddingManager(self.llm_cfg, message)
		embds_manager.layers = [
			torch.zeros(len(inputs), self.llm_cfg.n_dimension) for _ in range(self.llm_cfg.n_layer)
		]

		for i, txt in tqdm.tqdm(enumerate(inputs), desc="Extracting embeddings"):
			txt = self.apply_sft_template(instruction=txt, system_message=system_message)

			input_ids = self.tokenizer.apply_chat_template(txt, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

			with torch.no_grad():
				outputs = self.model(input_ids, output_hidden_states=True)

			hidden_states = outputs.hidden_states

			for j in range(self.llm_cfg.n_layer):
				embds_manager.layers[j][i, :] = hidden_states[j + 1][:, -1, :].detach().cpu()  # j + 1 because the first is input embedding

		return embds_manager

	def _register_hooks(self, perturbation):
		def _hook_fn(module, input, output, layer_idx, perturbation):
			output = perturbation.get_perturbation(output, layer_idx)

			return output

		retHook = []
		for i in range(self.llm_cfg.n_layer):
			layer = self.model.model.layers[i]
			hook = layer.register_forward_hook(partial(_hook_fn, layer_idx=i, perturbation=perturbation))
			retHook.append(hook)
		return retHook

	def prepare(self, posTrainPrompts, negTrainPrompts, posValPrompts, negValPrompts, pt):
		pos_train_embds = self.extract_embds(posTrainPrompts)
		neg_train_embds = self.extract_embds(negTrainPrompts)
		pos_test_embds = self.extract_embds(posValPrompts)
		neg_test_embds = self.extract_embds(negValPrompts)
		clfr = ClassifierManager('')
		clfr.fit(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds)
		print('Test Acc:')
		print(clfr.testacc)
		self.pert = Perturbation(clfr, target_probability=pt)
		self.hooks = self._register_hooks(self.pert)

	def Norm(self, prompts):
		eeee = self.extract_embds(prompts)
		allEmbdsNorm = []
		for j in range(self.llm_cfg.n_layer):
			embds = eeee.layers[j]
			allEmbdsNorm.append(torch.norm(embds, dim=-1).cpu().numpy().tolist())
		return allEmbdsNorm

	def Acc(self, posTrainPrompts, negTrainPrompts, posValPrompts, negValPrompts):
		pos_train_embds = self.extract_embds(posTrainPrompts)
		neg_train_embds = self.extract_embds(negTrainPrompts)
		pos_test_embds = self.extract_embds(posValPrompts)
		neg_test_embds = self.extract_embds(negValPrompts)
		clfr = ClassifierManager('')
		clfr.fit(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds)
		return clfr.testacc


class lastAllSCAV:
	def __init__(self, modelN, model, tokenizer, config):
		self.hooks = None
		self.pert = None
		self.model = model
		self.tokenizer = tokenizer
		self.config = config
		self.llm_cfg = CFG({
			'model_nickname': modelN,
			'model_name': modelN,
			'n_layer': config.num_hidden_layers,
			'n_dimension': config.hidden_size
		})

	def apply_sft_template(self, instruction, system_message=None):
		if system_message is not None:
			messages = [
				{
					"role": "system",
					"content": system_message
				},
				{
					"role": "user",
					"content": instruction
				}
			]
		else:
			messages = [
				{
					"role": "user",
					"content": instruction
				}
			]

		return messages

	def extract_embds(self, inputs: list[str], system_message: str = None, message: str = None) -> EmbeddingManager:
		embds_manager = EmbeddingManager(self.llm_cfg, message)
		embds_manager.layers = [
			torch.zeros(len(inputs), self.llm_cfg.n_dimension) for _ in range(self.llm_cfg.n_layer)
		]

		for i, txt in tqdm.tqdm(enumerate(inputs), desc="Extracting embeddings"):
			txt = self.apply_sft_template(instruction=txt, system_message=system_message)

			input_ids = self.tokenizer.apply_chat_template(txt, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

			with torch.no_grad():
				outputs = self.model(input_ids, output_hidden_states=True)

			hidden_states = outputs.hidden_states

			for j in range(self.llm_cfg.n_layer):
				embds_manager.layers[j][i, :] = hidden_states[j + 1][:, -1, :].detach().cpu()  # j + 1 because the first is input embedding

		return embds_manager

	def _register_hooks(self, perturbation):
		def _hook_fn(module, input, output, layer_idx, perturbation):
			output = perturbation.get_perturbation(output, layer_idx)

			return output

		retHook = []
		for i in range(self.llm_cfg.n_layer - 1):
			layer = self.model.model.layers[i]
			hook = layer.register_forward_hook(partial(_hook_fn, layer_idx=i, perturbation=perturbation))
			retHook.append(hook)
		return retHook

	def prepare(self, posTrainPrompts, negTrainPrompts, posValPrompts, negValPrompts, pt):
		pos_train_embds = self.extract_embds(posTrainPrompts)
		neg_train_embds = self.extract_embds(negTrainPrompts)
		pos_test_embds = self.extract_embds(posValPrompts)
		neg_test_embds = self.extract_embds(negValPrompts)
		clfr = ClassifierManager('')
		clfr.fit(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds)
		print('Test Acc:')
		print(clfr.testacc)
		self.pert = AllPerturbation(clfr, target_probability=pt)
		self.hooks = self._register_hooks(self.pert)

	def Norm(self, prompts):
		eeee = self.extract_embds(prompts)
		allEmbdsNorm = []
		for j in range(self.llm_cfg.n_layer):
			embds = eeee.layers[j]
			allEmbdsNorm.append(torch.norm(embds, dim=-1).cpu().numpy().tolist())
		return allEmbdsNorm

	def Acc(self, posTrainPrompts, negTrainPrompts, posValPrompts, negValPrompts):
		pos_train_embds = self.extract_embds(posTrainPrompts)
		neg_train_embds = self.extract_embds(negTrainPrompts)
		pos_test_embds = self.extract_embds(posValPrompts)
		neg_test_embds = self.extract_embds(negValPrompts)
		clfr = ClassifierManager('')
		clfr.fit(pos_train_embds, neg_train_embds, pos_test_embds, neg_test_embds)
		return clfr.testacc
