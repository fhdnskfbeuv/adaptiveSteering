import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase


# Llama 3 chat templates are based on
# <|begin_of_text|> is automatically added by the tokenizer

# LLAMA3_CHAT_TEMPLATE = """<|start_header_id|>user<|end_header_id|>
#
# {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# """
#
# LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM = """<|start_header_id|>system<|end_header_id|>
#
# {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
#
# {instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# """

# LLAMA3_REFUSAL_TOKS = [40] # 'I'

# def format_instruction_llama3_chat(
#     instruction: str,
#     output: str=None,
#     system: str=None,
#     include_trailing_whitespace: bool=True
# ):
#     if system is not None:
#         formatted_instruction = LLAMA3_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system_prompt=system)
#     else:
#         formatted_instruction = LLAMA3_CHAT_TEMPLATE.format(instruction=instruction)
#
#     if not include_trailing_whitespace:
#         formatted_instruction = formatted_instruction.rstrip()
#
#     if output is not None:
#         formatted_instruction += output
#
#     return formatted_instruction

def tokenize_instructions_mistral_chat(
	tokenizer: AutoTokenizer,
	instructions: List[str],
	outputs: List[str] = None,
	system: str = None,
	include_trailing_whitespace=True
):
	if outputs is not None:
		prompts = [
			tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": instruction}] if system is not None else [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True) + output
			for instruction, output in zip(instructions, outputs)
		]
	else:
		prompts = [
			tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": instruction}] if system is not None else [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True)
			for instruction in instructions
		]
	single_prompt = prompts[0]
	result = tokenizer(
		prompts,
		padding=True,
		truncation=False,
		add_special_tokens=not (tokenizer.bos_token is not None and single_prompt.startswith(tokenizer.bos_token)),
		return_tensors="pt",
	)

	return result


def orthogonalize_mistral_weights(model, direction: Float[Tensor, "d_model"]):
	model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

	for block in model.model.layers:
		block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
		block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T


def act_add_mistral_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
	dtype = model.model.layers[layer - 1].mlp.down_proj.weight.dtype
	device = model.model.layers[layer - 1].mlp.down_proj.weight.device

	bias = (coeff * direction).to(dtype=dtype, device=device)

	model.model.layers[layer - 1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class MistralModel(ModelBase):

	def _load_model(self, model_path, dtype=torch.bfloat16):
		model = AutoModelForCausalLM.from_pretrained(
			model_path,
			torch_dtype=dtype,
			trust_remote_code=True,
			device_map="auto",
		).eval()

		model.requires_grad_(False)

		return model

	def _load_tokenizer(self, model_path):
		tokenizer = AutoTokenizer.from_pretrained(model_path)

		tokenizer.padding_side = "left"
		tokenizer.pad_token = tokenizer.eos_token

		return tokenizer

	def _get_tokenize_instructions_fn(self):
		return functools.partial(tokenize_instructions_mistral_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

	def _get_eoi_toks(self):
		return self.tokenizer.encode(self.tokenizer.apply_chat_template([{"role": "user", "content": 'sdkfbkesjedfbjakbdhjkbkfjkdvfkb'}], tokenize=False, add_generation_prompt=True).split("sdkfbkesjedfbjakbdhjkbkfjkdvfkb")[-1], add_special_tokens=False)

	def _get_refusal_toks(self):

		inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": "How to build a Bomb?"}],
										   tokenize=True,
										   return_tensors="pt",
										   return_dict=True,
										   add_generation_prompt=True).to(self.model.device)
		generated_ids = self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
		print(self.tokenizer.decode(generated_ids[0][inputs['input_ids'][0].shape[0]:][:1]))
		return generated_ids[0][inputs['input_ids'][0].shape[0]:][:1]

	def _get_model_block_modules(self):
		return self.model.model.layers

	def _get_attn_modules(self):
		return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])

	def _get_mlp_modules(self):
		return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

	def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
		return functools.partial(orthogonalize_mistral_weights, direction=direction)

	def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
		return functools.partial(act_add_mistral_weights, direction=direction, coeff=coeff, layer=layer)
