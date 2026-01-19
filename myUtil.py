import csv
import gc
import json
import os.path
import time

import numpy as np
import torch
import tqdm
from colorama import Fore, Style
from peft import PeftModel
from strong_reject import evaluate, load_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from transformers import pipeline

import myJudge
from repe import repe_pipeline_registry, WrappedReadingVecModel

customizedChatTemplate = {  # We hope authors of models below provide official jinja-format chat template :(
	'Youliang/llama3-8b-instruct-lora-derta-100step': """{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% else %}{% endif %}{% endfor %}""",
	# see https://huggingface.co/Youliang/llama3-8b-instruct-lora-derta-100step. Yet, we must say that this is, in fact, similar to Llama2 and is quite different from Llama3's fromat:( Oh my god! They use Llama3's bos token but use Llama2's format????
	"vicuna-7b-v1.5": """{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{{ bos_token + system_message }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}""",
	# see https://huggingface.co/lmsys/vicuna-7b-v1.5/commit/3321f76e3f527bd14065daf69dad9344000a201d
	# "vicuna-7b-v1.5": """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ 'USER: ' + message['content'] | trim + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] | trim + eos_token + '\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"""  # see https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/vicuna.jinja. The difference lie in "\n\n" after USER. Yet, see also https://github.com/thu-coai/SafeUnlearning/blob/47892d0bd7258820ec5a28791534906f5dec48b3/data/ft_full_data/vicuna_format/dev.json#L3. Thus, we adopt the version above
}

lora2base = {
	'Youliang/llama3-8b-instruct-lora-derta-100step': "PawanKrd/Meta-Llama-3-8B-Instruct",
	"thkim0305/RepBend_Llama3_8B_LoRA": "PawanKrd/Meta-Llama-3-8B-Instruct",
	"thkim0305/RepBend_Mistral_7B_LoRA": "mistralai/Mistral-7B-Instruct-v0.2"
}

nameMap = {
	'thu-coai/Mistral-7B-Instruct-v0.2-safeunlearning': 'Mistral-SU',
	'thu-coai/vicuna-7b-v1.5-safeunlearning': 'Vicuna-SU',
	'lapisrocks/Llama-3-8B-Instruct-TAR-Refusal': 'Llama3-TAR',
	'GraySwanAI/Llama-3-8B-Instruct-RR': 'Llama3-CB',
	'cais/zephyr_7b_r2d2': 'R2D2',
	'Unispac/Llama2-7B-Chat-Augmented': 'Llama2-DA',
	'LLM-LAT/robust-llama3-8b-instruct': 'Llama3-LAT',
	'thkim0305/RepBend_Mistral_7B': 'Mistral-RB',
	'thkim0305/RepBend_Llama3_8B': 'Llama3-RB',
	"GraySwanAI/Mistral-7B-Instruct-RR": 'Mistral-CB',
	'Unispac/Gemma-2-9B-IT-With-Deeper-Safety-Alignment': 'Gemma-DA',
	'Youliang/llama3-8b-instruct-lora-derta-100step': 'Llama3-DeRTA',
	"PawanKrd/Meta-Llama-3-8B-Instruct": 'Llama-3-8B-Instruct',
	'meta-llama/Llama-2-7b-chat-hf': 'Llama-2-7b-chat',
	"Qwen/Qwen3-4B-Instruct-2507": 'Qwen3-4B-Instruct',
	"Qwen/Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
	"lmsys/vicuna-7b-v1.5": "Vicuna-7b-v1.5",
	"mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
	"google/gemma-2-9b-it": "Gemma-2-9b-it"
}


def extractFromJson(file_path, key):
	values = []
	with open(file_path, 'r', encoding='utf-8') as file:
		data = json.load(file)
		if not isinstance(data, list):
			data = [data]
		for d in data:
			values.append(d[key])
	return values


def loadDataset(dirP, harmTrain, benignTrain, harmVal, benignVal, full=False, trainSize=None):
	harmfulTrain = extractFromJson(os.path.join(dirP, 'harmful_train.json'), "instruction")
	harmfulVal = extractFromJson(os.path.join(dirP, 'harmful_val.json'), "instruction")
	harmfulTest = extractFromJson(os.path.join(dirP, 'harmful_test.json'), "instruction")
	harmlessTrain = extractFromJson(os.path.join(dirP, 'harmless_train.json'), "instruction")
	harmlessVal = extractFromJson(os.path.join(dirP, 'harmless_val.json'), "instruction")
	harmlessTest = extractFromJson(os.path.join(dirP, 'harmless_test.json'), "instruction")
	harmfulTVLen = len(harmfulTrain + harmfulVal)
	harmlessTVLen = len(harmlessTrain + harmlessVal)
	seenHarmful = harmfulTrain + harmfulVal
	seenHarmless = harmlessTrain + harmlessVal
	if full:
		if trainSize is not None:
			totalNum = len(seenHarmful + harmfulTest)
			tNum = int(totalNum * trainSize)
			return {
				'train': [(seenHarmful + harmfulTest)[:tNum], (seenHarmless + harmlessTest)[:tNum]],
				'val': [(seenHarmful + harmfulTest)[tNum:], (seenHarmless + harmlessTest)[tNum:totalNum]],
				'test': [harmfulTest, harmlessTest],
			}
		else:
			return {
				'train': [seenHarmful + harmfulTest, (seenHarmless + harmlessTest)[:len(seenHarmful + harmfulTest)]],
				'val': [seenHarmful + harmfulTest, (seenHarmless + harmlessTest)[:len(seenHarmful + harmfulTest)]],
				'test': [harmfulTest, harmlessTest],
			}
	return {
		'train': [seenHarmful[:harmTrain], seenHarmless[:benignTrain]],
		'val': [seenHarmful[harmTrain:min(harmfulTVLen, harmTrain + harmVal)], seenHarmless[benignTrain:min(harmlessTVLen, benignTrain + benignVal)]],
		'test': [harmfulTest, harmlessTest],
	}


def init_rep_control(
	model,
	tokenizer,
	harmful,
	benign,
	layer_id: list = list(range(-11, -21, -1)),
	repe_coeff: float = 1.0,
):
	repe_pipeline_registry()

	rep_token = -1
	hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
	n_difference = 1
	direction_method = 'pca'
	rep_reading_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
	# rep_reading_pipeline.tokenizer.pad_token_id = rep_reading_pipeline.model.config.eos_token_id
	direction_finder_kwargs = {"n_components": 5}
	component_index = 0

	# print("Loading dataset...")
	# dataset = load_dataset("justinphan3110/harmful_harmless_instructions")
	# train_dataset = dataset['train']

	train_data = []
	train_labels = []
	for h, b in zip(harmful, benign):
		train_data.append([h, b])
		train_labels.append([False, True])
	train_data = np.concatenate(train_data).tolist()
	train_data = [[{"role": "user", "content": c}] for c in train_data]
	train_data = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in train_data]

	print("Getting directions...")
	rep_reader = rep_reading_pipeline.get_directions(
		train_data,
		rep_token=rep_token,
		hidden_layers=hidden_layers,
		n_difference=n_difference,
		train_labels=train_labels,
		direction_method=direction_method,
		direction_finder_kwargs=direction_finder_kwargs
	)

	activations = {}
	for layer in layer_id:
		activations[layer] = torch.tensor(repe_coeff * rep_reader.directions[layer][component_index] * rep_reader.direction_signs[layer][component_index]).to(model.device).to(model.dtype)

	print("Wrapping model...")
	wrapped_model = WrappedReadingVecModel(model, tokenizer)
	wrapped_model.unwrap()
	wrapped_model.wrap_block(layer_id, block_name="decoder_block")

	### Controlled model hidden_states:
	wrapped_model.set_controller(layer_id, activations, masks=1, operator='linear_comb')

	return wrapped_model


def loadModel(modelN, tokenizerN):
	tryNum = 10
	while tryNum > 0:
		try:
			tryNum -= 1
			if modelN in lora2base.keys():
				print(f'{modelN} has adapter! Loading now.')
				model = AutoModelForCausalLM.from_pretrained(
					lora2base[modelN], torch_dtype=torch.bfloat16, token=os.getenv('HF_TOKEN', default=None), attn_implementation="sdpa"
				)
				model = PeftModel.from_pretrained(model, modelN, adapter_name="default")
				config = AutoConfig.from_pretrained(lora2base[modelN], token=os.getenv('HF_TOKEN', default=None))
			else:
				model = AutoModelForCausalLM.from_pretrained(modelN, dtype=torch.bfloat16, token=os.getenv('HF_TOKEN', default=None),
															 attn_implementation="sdpa")
				config = AutoConfig.from_pretrained(modelN, token=os.getenv('HF_TOKEN', default=None))
			processor = AutoProcessor.from_pretrained(tokenizerN, token=os.getenv('HF_TOKEN', default=None))
			for k in customizedChatTemplate.keys():
				if k in modelN:
					processor.chat_template = customizedChatTemplate[k]
			example = processor.apply_chat_template([{"role": "user", "content": '{Instruct}'}],
													tokenize=True,
													return_tensors="pt",
													return_dict=True,
													add_generation_prompt=True)['input_ids']
			print(f'{Fore.RED} {modelN}\'s chat template: {processor.batch_decode(example, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]} {Style.RESET_ALL}')
			print(f'{Fore.RED} {modelN}\'s chat template: {processor.convert_ids_to_tokens(example[0])} {Style.RESET_ALL}')
			model.generation_config.use_cache = True
			if model.generation_config.pad_token_id is None:
				model.generation_config.pad_token_id = model.generation_config.eos_token_id[0] if isinstance(model.generation_config.eos_token_id, list) else model.generation_config.eos_token_id
				print(f"Setting `pad_token_id` to `eos_token_id`:{model.generation_config.pad_token_id} for open-end generation.")
			successDL = True
			time.sleep(1)
			model.cuda()
			if processor.pad_token_id is None:
				processor.pad_token_id = model.config.eos_token_id[0] if isinstance(model.config.eos_token_id, list) else model.config.eos_token_id
			tryNum = 0
		except Exception as e:
			print(e)
			print('What can I say?')
			time.sleep(1)
	return model, processor, config


def loadJudge(judgeN):
	tryNum = 10
	judgeM = None
	judgeF = None
	while tryNum > 0:
		try:
			tryNum -= 1
			if judgeN[0] == 'llama':
				judgeM = myJudge.LlamaGuard3(r'meta-llama/Llama-Guard-3-8B')
				judgeF = judgeM.judge
			elif len(judgeN) == 1 and 'qwen' in judgeN[0].lower():
				judgeM = myJudge.QwenGuard(judgeN[0])
				judgeF = judgeM.judge
			elif judgeN[0] == 'hb':
				judgeF = myJudge.HarmBenchJudge
				myJudge.HarmBenchJudge('a', 'b')
				judgeM = evaluate.cached_models["harmbench"][0]
			elif judgeN[0] == 'sjf':
				judgeF = myJudge.StrongRejectJudge
				myJudge.StrongRejectJudge('a', 'b')
				judgeM = evaluate.cached_models["strongreject_finetuned"][0]
			elif len(judgeN) == 1:
				judgeM = AutoModelForCausalLM.from_pretrained(judgeN[0], torch_dtype="auto", device_map="auto",
															  token=os.getenv('HF_TOKEN', default=None), attn_implementation="sdpa")
				judgeT = AutoTokenizer.from_pretrained(judgeN[0], token=os.getenv('HF_TOKEN', default=None))
				judgeF = myJudge.SJRubricHF(judgeM, judgeT).judge
			elif len(judgeN) == 3:
				judgeF = myJudge.SJRubricAPI(judgeN[0], judgeN[1], judgeN[2]).judge
			else:
				print(f'{judgeN} not implemented')
				exit(1)
			tryNum = 0
		except Exception as e:
			print(e)
	return judgeM, judgeF


def easyGen(model, processor, text: str, maxL=128, prefix=None, doSample=False):
	query = [
		{
			"role": "user",
			"content": text
		}
	]
	inputs = processor.apply_chat_template(query,
										   tokenize=True,
										   return_tensors="pt",
										   return_dict=True,
										   add_generation_prompt=True).to(model.device)  # Prepare texts for processing
	if prefix is not None:  # [1, L]
		inputs['input_ids'] = torch.concat([inputs['input_ids'], prefix.repeat(inputs['input_ids'].shape[0], 1).to(inputs['input_ids'])], dim=1)
		inputs['attention_mask'] = torch.concat([inputs['attention_mask'], torch.ones((inputs['attention_mask'].shape[0], prefix.shape[1])).to(inputs['attention_mask'])], dim=1)
	generated_ids = model.generate(**inputs, max_new_tokens=maxL, do_sample=doSample)
	trimmedIDs = []
	for i in range(len(generated_ids)):
		trimmedIDs.append(generated_ids[i][inputs['input_ids'][i].shape[0]:])
	completeion = processor.batch_decode(
		trimmedIDs, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)[0]
	fullStr = processor.batch_decode(
		generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
	)[0]
	return fullStr, completeion


def GenAndEval(model, processor, judge, prompts, maxL, prefixs=None, doSample=False):
	allCompletion = []
	allScores = []
	meanScore = 0
	with tqdm.tqdm(prompts, total=len(prompts), desc=f"{Fore.RED}Average Score: {meanScore}{Style.RESET_ALL}", dynamic_ncols=True) as pbar:
		for i, prompt in enumerate(prompts):
			fullStr, completion = easyGen(model, processor, prompt, maxL, prefixs[i] if prefixs is not None else None, doSample)
			score = judge(prompt, completion)
			allCompletion.append(completion)
			allScores.append(score)
			meanScore = torch.tensor(allScores).float().mean().item()
			pbar.set_description(f"{Fore.RED}Average Score: {meanScore}; Current Score: {allScores[-1]};){Style.RESET_ALL}")
			pbar.update()
	return allCompletion, allScores


def gen(model, processor, prompts, maxL, prefixs=None, doSample=False):
	allCompletion = []
	with tqdm.tqdm(prompts, total=len(prompts), dynamic_ncols=True) as pbar:
		for i, prompt in enumerate(prompts):
			fullStr, completion = easyGen(model, processor, prompt, maxL, prefixs[i] if prefixs is not None else None, doSample)
			# print(f'{completion}')
			allCompletion.append(completion)
			pbar.update()
	return allCompletion


def eval(prPair, judges):
	allScores = {}
	for judgeN in judges:
		judgeM, judgeF = loadJudge(judgeN.split(' '))
		print(judgeN)
		meanScore = 0
		scores = []
		with tqdm.tqdm(prPair, total=len(prPair), desc=f"{Fore.RED}Average Score: {meanScore}{Style.RESET_ALL}", dynamic_ncols=True) as pbar:
			for prompt, response in prPair:
				score = judgeF(prompt, response)
				scores.append(score)
				meanScore = torch.tensor(scores).float().mean().item()
				pbar.set_description(f"{Fore.RED}Average Score: {meanScore}; Current Score: {scores[-1]};){Style.RESET_ALL}")
				pbar.update()
		allScores[judgeN.split(' ')[0]] = scores
		if judgeM is not None:
			del judgeM
			torch.cuda.empty_cache()
			gc.collect()
	return allScores


def loadEvalData(evalData):
	prompts = []
	if evalData == 'sj':
		prompts = [p['forbidden_prompt'] for p in load_datasets.load_strongreject()]
		prompts = prompts[:100]
	elif evalData == 'harmbench':
		csvr = csv.reader(open(r'./instructions/harmbench_behaviors_text_test.csv', 'r+'))
		for row in csvr:
			if row[1] == 'standard':
				prompts.append(row[0])
		prompts = prompts[:100]
	elif evalData == 'harm':
		csvr = csv.reader(open(r'./instructions/harmbench_behaviors_text_test.csv', 'r+'))
		for row in csvr:
			if row[1] == 'standard':
				prompts.append(row[0])
		prompts = prompts[:100] + [p['forbidden_prompt'] for p in load_datasets.load_strongreject()][:100]
	elif evalData == 'benign':
		insts = loadDataset(r'./instructions/', 100, 100, 100, 100)
		prompts = insts['train'][1]
	else:
		print(f'{evalData} not supported')
		exit(1)
	return prompts
