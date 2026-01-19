import functools
import gc
import textwrap
from typing import List

import numpy as np
import torch
from colorama import Fore
from jaxtyping import Float, Int
from sklearn.decomposition import PCA
from torch import Tensor
from torch.nn.functional import cosine_similarity, normalize
from tqdm import tqdm
from transformer_lens import HookedTransformer, utils, ActivationCache
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES



BATCH_SIZE = 1


def get_rotate_to_target_func(target_degree, basis1, basis2):
	assert len(basis1.shape) == 1
	assert len(basis2.shape) == 1
	assert basis1.shape == basis2.shape

	n = basis1.shape[-1]

	# ensure bases are orthonormal
	u = basis1 / np.linalg.norm(basis1)
	v = basis2 - (basis2 @ u) * u
	v /= np.linalg.norm(v)

	theta = np.deg2rad(target_degree)
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)

	P = np.outer(u, u) + np.outer(v, v)

	# rotate counter-clockwise
	R_theta = [[cos_theta, -sin_theta], [sin_theta, cos_theta]]

	uv = np.column_stack([u, v])

	rotated_component = uv @ R_theta @ np.array([1, 0])

	def __func(x: Tensor):
		Px = x @ torch.tensor(P, device=x.device, dtype=x.dtype)
		scale = Px.norm(dim=-1, keepdim=True)

		result = (
			x
			- Px
			+ scale * torch.tensor(rotated_component, device=x.device, dtype=x.dtype)
		)

		return result

	return __func


def activation_rotation_hook(
	activation: Float[Tensor, "... d_act"],
	hook: HookPoint,
	transformation_func,
):
	return transformation_func(activation)


def instructions_to_chat_tokens(
	tokenizer: AutoTokenizer,
	instructions: List[str],
) -> Int[Tensor, "batch_size seq_len"]:
	if tokenizer.chat_template:
		convos = [
			[{"role": "user", "content": instruction}] for instruction in instructions
		]
		return tokenizer.apply_chat_template(
			convos,
			padding=True,
			truncation=False,
			add_generation_prompt=True,
			return_tensors="pt",
		)
	else:
		return tokenizer(
			instructions, padding=True, truncation=False, return_tensors="pt"
		).input_ids


def _generate_with_hooks(
	model: HookedTransformer,
	toks: Int[Tensor, "batch_size seq_len"],
	max_tokens_generated: int = BATCH_SIZE,
	fwd_hooks=[],
) -> List[str]:
	all_toks = torch.zeros(
		(toks.shape[0], toks.shape[1] + max_tokens_generated),
		dtype=torch.long,
		device=toks.device,
	)
	all_toks[:, : toks.shape[1]] = toks

	for i in range(max_tokens_generated):
		with model.hooks(fwd_hooks=fwd_hooks):
			logits = model(all_toks[:, : -max_tokens_generated + i])
			next_tokens = logits[:, -1, :].argmax(
				dim=-1
			)  # greedy sampling (temperature=0)
			all_toks[:, -max_tokens_generated + i] = next_tokens

	return model.tokenizer.batch_decode(
		all_toks[:, toks.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False
	)


def get_generations(
	model: HookedTransformer,
	instructions: List[str],
	tokenizer: AutoTokenizer,
	fwd_hooks=[],
	max_tokens_generated: int = 64,
	batch_size: int = BATCH_SIZE,
) -> List[str]:
	generations = []

	for i in tqdm(range(0, len(instructions), batch_size)):
		toks = instructions_to_chat_tokens(
			tokenizer=tokenizer, instructions=instructions[i: i + batch_size]
		)

		with torch.no_grad():
			generation = _generate_with_hooks(
				model,
				toks,
				max_tokens_generated=max_tokens_generated,
				fwd_hooks=fwd_hooks,
			)
		generations.extend(generation)

	return generations


def run_single_sample(model, input, tokenizer, fwd_hooks=[], max_tokens_generated=64):
	baseline_generations = get_generations(
		model,
		[input],
		tokenizer,
		fwd_hooks=[],
		max_tokens_generated=max_tokens_generated,
	)
	intervention_generations = get_generations(
		model,
		[input],
		tokenizer,
		fwd_hooks=fwd_hooks,
		max_tokens_generated=max_tokens_generated,
	)

	print(f"INSTRUCTION: {repr(input)}")
	print(Fore.GREEN + f"BASELINE COMPLETION:")
	print(
		textwrap.fill(
			baseline_generations[0],
			width=100,
			initial_indent="\t",
			subsequent_indent="\t",
		)
	)
	print(Fore.RED + f"INTERVENTION COMPLETION:")
	print(
		textwrap.fill(
			intervention_generations[0],
			width=100,
			initial_indent="\t",
			subsequent_indent="\t",
		)
	)


def __run_with_cache(model, data, batch_size):
	cache = {}
	with torch.no_grad():
		for i in range(0, len(data), batch_size):
			_, batch_cache = model.run_with_cache(
				data[i: i + batch_size],
				names_filter=lambda hook_name: "resid" in hook_name,
				return_cache_object=False,
			)
			for k, v in batch_cache.items():
				if k not in cache:
					cache[k] = [v.clone().detach().cpu()]
				else:
					cache[k].append(v.clone().detach().cpu())
		for k, v in cache.items():
			cache[k] = torch.concat(v, dim=0)

	return ActivationCache(cache, model)


def get_template_suffix_toks(tokenizer):
	# Since the padding is on the left side, the suffix of all samples are the same
	# when using the same template.
	# The activations on these suffix tokens are after the prompt has been processed,
	# thus it's interesting to see how the activations differ between contrastive
	# samples

	# get the common suffix between 2 samples
	toks = instructions_to_chat_tokens(tokenizer=tokenizer, instructions=["a", "b"])
	suffix = toks[0]
	for i in range(len(toks[0]) - 1, -1, -1):
		if toks[0][i] != toks[1][i]:
			suffix = toks[0][i + 1:]

	return tokenizer.convert_ids_to_tokens(suffix)


def get_activations(
	model: HookedTransformer,
	instructions: List[str],
	batch_size: int = BATCH_SIZE,
	act_names: List[str] = ["resid_mid", "resid_post"],
	num_last_tokens: int = 1,
):
	# tokenize instructions
	toks = instructions_to_chat_tokens(
		tokenizer=model.tokenizer, instructions=instructions
	)

	# run model on instructions and cache activations
	with torch.no_grad():
		cache = __run_with_cache(model, toks, batch_size=BATCH_SIZE)

	# get activations for the last n tokens
	acts = torch.stack(
		[
			torch.stack(
				[cache[act, layer][:, -num_last_tokens:, :] for act in act_names]
			)
			for layer in range(model.cfg.n_layers)
		]
	)

	# layers x resid_modules x batch x tokens x dim
	return acts, cache


def get_pairwise_cosine_similarity(acts_normed):
	# comput cosine similarity of each pair of vector from a set of normalized vectors
	# acts_normed is ... x batch x toks x dim

	acts_normed = torch.tensor(acts_normed, device="cuda")

	# ... batch1 toks dim, ... batch2 toks dim -> ... toks batch1 batch2
	acts_pairwise_sim = torch.einsum("...ikl,...jkl->...kij", acts_normed, acts_normed)

	batch_size = acts_pairwise_sim.shape[-1]

	# get the indices of the upper triangular part of the batch x batch similarity matrix
	indices = np.arange(batch_size ** 2).reshape(batch_size, batch_size)
	indices = indices[np.triu_indices_from(indices, k=1)]

	# ... x toks x batch x (batch * batch)
	acts_pairwise_sim = acts_pairwise_sim.reshape(*acts_pairwise_sim.shape[:-2], -1)
	# ... x toks x batch x (batch * (batch - 1) // 2)
	acts_pairwise_sim = acts_pairwise_sim[..., indices]
	# ... x (batch * (batch - 1) // 2) x toks
	acts_pairwise_sim = acts_pairwise_sim.swapaxes(-1, -2)

	return acts_pairwise_sim


def get_cosine_with_mean(acts_normed):
	# compute cosine similarity of each vector with the mean vector
	# acts_normed is ... x batch x toks x dim

	acts_normed = torch.tensor(acts_normed, device="cuda")
	mean_act = acts_normed.mean(axis=2)
	mean_act /= mean_act.norm(dim=-1, keepdim=True)

	# ... batch toks dim, ... toks dim -> ... batch toks
	cosine_with_mean = torch.einsum("...ijk,...jk ->...ij", acts_normed, mean_act)

	return cosine_with_mean


def get_rotation_matrix(degree, basis1, basis2):
	assert len(basis1.shape) == 1
	assert len(basis2.shape) == 1
	assert basis1.shape == basis2.shape

	n = basis1.shape[-1]

	if degree % 360 == 0:
		return np.eye(n)

	# ensure bases are orthonormal
	u = basis1 / np.linalg.norm(basis1)
	v = basis2 - (basis2 @ u) * u
	v /= np.linalg.norm(v)

	theta = np.deg2rad(degree)
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)
	# print(cos_theta, sin_theta)

	# rotate counter-clockwise
	R_theta = [[cos_theta, -sin_theta], [sin_theta, cos_theta]]

	uv = np.column_stack([u, v])
	R = np.eye(n) - (np.outer(u, u) + np.outer(v, v)) + uv @ R_theta @ uv.T

	return R


def rotate_to_target(x, target_degree, basis1, basis2):
	assert len(basis1.shape) == 1
	assert len(basis2.shape) == 1
	assert basis1.shape == basis2.shape

	n = basis1.shape[-1]

	# ensure bases are orthonormal
	u = basis1 / np.linalg.norm(basis1)
	v = basis2 - (basis2 @ u) * u
	v /= np.linalg.norm(v)

	theta = np.deg2rad(target_degree)
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)

	P = np.outer(u, u) + np.outer(v, v)

	# rotate counter-clockwise
	R_theta = [[cos_theta, -sin_theta], [sin_theta, cos_theta]]

	uv = np.column_stack([u, v])

	rotated_component = uv @ R_theta @ np.array([1, 0])
	Px = x @ P
	scale = np.linalg.norm(Px, axis=-1, keepdims=True)

	result = x - Px + scale * rotated_component

	return result


def _get_rotation_args(
	first_directions: torch.Tensor,
	second_directions,
	target_degree: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
	"""Compute the rotated component with respect to a 2D subspace and an rotation
	angle."""

	if second_directions is None:
		return None, None

	# first_direction: (batch) x hidden_dim
	# second_directions: (batch) x hidden_dim

	# ensure bases are orthonormal
	b1 = first_directions / first_directions.norm(dim=-1, keepdim=True)
	b2 = (
		second_directions - torch.sum(second_directions * b1, dim=-1, keepdim=True) * b1
	)
	b2 /= b2.norm(dim=-1, keepdim=True)

	theta = np.deg2rad(target_degree)
	cos_theta = np.cos(theta)
	sin_theta = np.sin(theta)

	proj_matrix = torch.einsum("...i, ...j -> ...ij", b1, b1) + torch.einsum(
		"...i, ...j -> ...ij", b2, b2
	)

	uv = torch.stack([b1.expand_as(b2), b2], dim=-1)  # shape (..., 2)

	# rotate counter-clockwise
	R_theta = torch.tensor(
		[[cos_theta, -sin_theta], [sin_theta, cos_theta]],
		device=uv.device,
		dtype=uv.dtype,
	)

	rotated_component = (
		uv @ R_theta @ torch.tensor([1, 0], device=uv.device, dtype=uv.dtype)
	)

	return proj_matrix, rotated_component


def get_angular_steering_output_hook(
	steering_config: dict[str, Tensor],
	target_degree: float,
	adaptive_mode: int = 1,
):
	first_dir = torch.from_numpy(steering_config["first_direction"])
	second_dir = torch.from_numpy(steering_config["second_direction"])
	proj_matrix, rotated_component = _get_rotation_args(
		first_directions=first_dir,
		second_directions=second_dir,
		target_degree=target_degree,
	)

	def hook_fn(module, input, output):
		nonlocal first_dir, second_dir, proj_matrix, rotated_component
		if isinstance(output, tuple):
			activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
		else:
			activation: Float[Tensor, "batch_size seq_len d_model"] = output
		# if adaptive_mode < 4:
		#     proj_matrix, rotated_component = _get_rotation_args(
		#         first_directions=first_dir,
		#         second_directions=second_dir,
		#         target_degree=target_degree,
		#     )
		# else:
		#     proj_matrix, rotated_component = _get_rotation_args(
		#         first_directions=activation,
		#         second_directions=second_dir.to(activation.device),
		#         target_degree=target_degree,
		#     )
		proj_matrix = proj_matrix.to(activation)
		rotated_component = rotated_component.to(activation)
		Px = torch.einsum("...i, ...ij -> ...j", activation, proj_matrix)
		scale = Px.norm(dim=-1, keepdim=True)
		if adaptive_mode in {0, 4}:
			activation += -Px + scale * rotated_component
		else:
			if adaptive_mode == 1:
				feature_direction = first_dir
			elif adaptive_mode == 2:
				feature_direction = second_dir
			elif adaptive_mode == 3:
				feature_direction = first_dir
			else:
				raise ValueError(f"Invalid adaptive mode: {adaptive_mode}")
			feature_direction = feature_direction.to(
				device=activation.device, dtype=activation.dtype
			)
			proj_to_feature_direction = activation @ feature_direction
			mask = proj_to_feature_direction > 0
			# activation: batch x seq_len x hidden_dim
			# mask: batch x seq_len
			# scale: batch x seq_len x 1
			# rotated_component: (batch) x seq_len x hidden_dim
			# Px: batch x seq_len x hidden_dim
			activation += mask.unsqueeze(-1) * (scale * rotated_component - Px)
		if isinstance(output, tuple):
			return (activation, *output[1:])
		else:
			return activation

	return hook_fn


class AngularSteering:
	def __init__(self):
		self.allConfig = None

	def getActivation(self, model, harmfulTrainPrompts, harmlessTrainPrompts):
		# extraction points per decoder block
		act_names = ["resid_mid", "resid_post"]

		# get the template suffix tokens
		template_suffix_toks = get_template_suffix_toks(model.tokenizer)
		if not template_suffix_toks:
			template_suffix_toks = ["<last token>"]

		# only get the activations of the template suffix tokens since these tokens are the same
		# for all samples
		num_last_tokens = len(template_suffix_toks)
		print("template_suffix_toks:", template_suffix_toks)

		# get activations for harmful instructions then save to file
		harmful_acts, cache = get_activations(
			model,
			harmfulTrainPrompts,
			batch_size=1,
			act_names=act_names,
			num_last_tokens=num_last_tokens,
		)
		harmful_acts = harmful_acts.cpu().float()
		# get activations for harmless instructions then save to file
		harmless_acts, cache = get_activations(
			model,
			harmlessTrainPrompts,
			batch_size=1,
			act_names=act_names,
			num_last_tokens=num_last_tokens,
		)
		harmless_acts = harmless_acts.cpu().float()
		return harmful_acts, harmless_acts

	def prepare(self, modelN, harmfulTrainPrompts, harmlessTrainPrompt, tokenizer=None, hfModel=None):
		if modelN not in OFFICIAL_MODEL_NAMES:
			OFFICIAL_MODEL_NAMES.append(modelN)
		model = HookedTransformer.from_pretrained_no_processing(
			modelN,
			device='cuda',
			dtype=torch.bfloat16,
			default_padding_side="left",
			tokenizer=tokenizer,
			hf_model=hfModel
			# bf16=True
		)
		model.tokenizer.padding_side = "left"
		if not model.tokenizer.pad_token:
			if "qwen1" in modelN.lower():
				model.tokenizer.pad_token = "<|endoftext|>"
			elif model.tokenizer.eos_token:
				model.tokenizer.pad_token = model.tokenizer.eos_token
			else:
				raise ValueError("No pad token found in the tokenizer.")

		chosen_token = -1
		harmful_acts, harmless_acts = self.getActivation(model, harmfulTrainPrompts, harmlessTrainPrompt)
		harmful_acts_normed = harmful_acts / harmful_acts.norm(dim=-1, keepdim=True)
		harmless_acts_normed = harmless_acts / harmless_acts.norm(dim=-1, keepdim=True)
		harmful_acts_normed_mean = harmful_acts_normed.mean(dim=2)
		harmless_acts_normed_mean = harmless_acts_normed.mean(dim=2)
		harmful_acts_normed_mean_normed = normalize(
			harmful_acts_normed_mean[:, :, chosen_token], dim=-1
		)
		harmless_acts_normed_mean_normed = normalize(
			harmless_acts_normed_mean[:, :, chosen_token], dim=-1
		)
		refusal_dirs = harmful_acts_normed_mean_normed - harmless_acts_normed_mean_normed

		refusal_dirs /= refusal_dirs.norm(dim=-1, keepdim=True)
		refusal_dirs = refusal_dirs.cpu().float().numpy()
		refusal_dirs_flatten = refusal_dirs.reshape(-1, refusal_dirs.shape[-1])
		pca_model = PCA().fit(refusal_dirs_flatten)
		components = pca_model.components_
		raw_dirs = harmful_acts_normed_mean_normed - harmless_acts_normed_mean_normed
		raw_dirs = raw_dirs.reshape((-1, raw_dirs.shape[-1]))
		criteria = raw_dirs.norm(dim=-1)[:-1]
		flatten_dirs = refusal_dirs.reshape(-1, refusal_dirs.shape[-1])
		pairwise_cosine = flatten_dirs @ flatten_dirs.T
		mean_cosine = np.nanmean(pairwise_cosine, axis=-1)
		argmax = np.nanargmax(criteria)
		max_norm_layer = argmax // 2
		max_norm_act_idx = argmax % 2
		argmax = np.nanargmax(mean_cosine)
		max_mean_cosine_layer = argmax // 2
		max_mean_cosine_act_idx = argmax % 2
		layernorm_modules = ["input_layernorm", "post_attention_layernorm"]
		if "gemma" in modelN:
			layernorm_modules += ["post_attention_layernorm", "post_feedforward_layernorm"]

		mean_d = refusal_dirs_flatten.mean(axis=0)
		mean_d /= np.linalg.norm(mean_d)

		self.allConfig = {}
		# saving various steering configs
		for first_direction, first_dir_name in [
			(
				refusal_dirs[max_norm_layer][max_norm_act_idx].copy(),
				"norm",
			),
			(
				refusal_dirs[max_mean_cosine_layer][max_mean_cosine_act_idx].copy(),
				"cosine",
			),
			(mean_d.copy(), "dir_mean"),
		]:

			second_direction = components[0].copy()

			num_layers = refusal_dirs.shape[0]
			steering_config = {}
			for layer_idx in range(num_layers):
				for module in layernorm_modules:
					if module != "input_layernorm":
						module_name = f"model.layers.{layer_idx}.{module}"
					elif layer_idx < num_layers - 1:
						module_name = f"model.layers.{layer_idx + 1}.{module}"
					else:
						continue

					steering_config[module_name] = {
						"mode": "rotate_to",
						"first_direction": first_direction,
						"second_direction": second_direction,
					}
			self.allConfig[first_dir_name] = steering_config
		if hfModel is not None:
			del hfModel
		del model
		torch.cuda.empty_cache()
		gc.collect()

	def addHook(self, model, degree, steerType='cosine', **kwargs):
		module_dict = dict(model.named_modules())
		hooks = [
			(
				module_dict[module_name],
				get_angular_steering_output_hook(
					steering_config=steering_config,
					target_degree=degree,
					adaptive_mode=0,
				),
			)
			for module_name, steering_config in self.allConfig[steerType].items()
		]
		handles = []
		for module, hook in hooks:
			partial_hook = functools.partial(hook, **kwargs)
			handles.append(module.register_forward_hook(partial_hook))
		return handles
