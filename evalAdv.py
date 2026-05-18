import argparse
import copy
import csv
import os
import time

import PIL.Image
import torch
import tqdm
from colorama import Fore, Style
from datasets import disable_caching
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoConfig, AutoProcessor
import torchvision.transforms.functional as ttf

import myUtil

model2thinkend = {
	'Qwen/Qwen3-4B-Thinking-2507': "</think>",
	'zai-org/GLM-4.6V-Flash': "</think>",
}


def easyLVLMGen(model, processor, text: str, img=None, maxL=128, doSample=False, rawCompletion=False, endThink=None):
	query = [
		{
			"role": "user",
			"content": [{'type': 'image', "image": img}, {'type': 'text', "text": text}] if img is not None else [{'type': 'text', "text": text}]
		}
	]
	inputs = processor.apply_chat_template(query,
										   tokenize=True,
										   return_tensors="pt",
										   return_dict=True,
										   add_generation_prompt=True).to(model.device)  # Prepare texts for processing
	generated_ids = model.generate(**inputs, max_new_tokens=maxL, do_sample=doSample)
	trimmedIDs = []
	for i in range(len(generated_ids)):
		trimmedIDs.append(generated_ids[i][inputs['input_ids'][i].shape[0]:])
	completion = processor.batch_decode(
		trimmedIDs, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)[0]
	fullStr = processor.batch_decode(
		generated_ids, skip_special_tokens=not rawCompletion, clean_up_tokenization_spaces=False
	)[0]
	if endThink is not None:
		completion = completion.split(endThink)[-1]
	return fullStr, completion


def genLVLM(model, processor, prompts, imgs, maxL, doSample=False, endThink=None):
	allCompletion = []
	inputs = zip(prompts, imgs) if imgs is not None else zip(prompts, [None] * len(prompts))
	with tqdm.tqdm(inputs, total=len(prompts), dynamic_ncols=True) as pbar:
		for i, (prompt, img) in enumerate(inputs):
			fullStr, completion = easyLVLMGen(model, processor, prompt, img, maxL, doSample, False, endThink)
			# print(f'{completion}')
			allCompletion.append(completion)
			pbar.update()
	return allCompletion


def loadVisualModel(modelN, tokenizerN):
	tryNum = 10
	while tryNum > 0:
		try:
			tryNum -= 1
			if modelN in myUtil.lora2base.keys():
				print(f'{modelN} has adapter! Loading now.')
				model = AutoModelForImageTextToText.from_pretrained(
					myUtil.lora2base[modelN], torch_dtype=torch.bfloat16, token=os.getenv('HF_TOKEN', default=None), attn_implementation="sdpa"
				)
				model = PeftModel.from_pretrained(model, modelN, adapter_name="default")
				config = AutoConfig.from_pretrained(myUtil.lora2base[modelN], token=os.getenv('HF_TOKEN', default=None))
			else:
				model = AutoModelForImageTextToText.from_pretrained(modelN, dtype=torch.bfloat16, token=os.getenv('HF_TOKEN', default=None),
																	attn_implementation="sdpa")
				config = AutoConfig.from_pretrained(modelN, token=os.getenv('HF_TOKEN', default=None))
			processor = AutoProcessor.from_pretrained(tokenizerN, token=os.getenv('HF_TOKEN', default=None), use_fast_image_processor=True)
			for k in myUtil.customizedChatTemplate.keys():
				if k in modelN:
					processor.chat_template = myUtil.customizedChatTemplate[k]
			example = processor.apply_chat_template([{"role": "user", "content": [{'type': 'image'}, {'type': 'text', "text": '{Instruct}'}]}],
													tokenize=True,
													return_tensors="pt",
													return_dict=True,
													add_generation_prompt=True)['input_ids']
			print(f'{Fore.RED} {modelN}\'s chat template: {processor.tokenizer.batch_decode(example, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]} {Style.RESET_ALL}')
			print(f'{Fore.RED} {modelN}\'s chat template: {processor.tokenizer.convert_ids_to_tokens(example[0])} {Style.RESET_ALL}')
			model.generation_config.use_cache = True
			if model.generation_config.pad_token_id is None:
				model.generation_config.pad_token_id = model.generation_config.eos_token_id[0] if isinstance(model.generation_config.eos_token_id, list) else model.generation_config.eos_token_id
				print(f"Setting `pad_token_id` to `eos_token_id`:{model.generation_config.pad_token_id} for open-end generation.")
			time.sleep(1)
			model.cuda()
			if processor.tokenizer.pad_token_id is None:
				processor.tokenizer.pad_token_id = model.config.eos_token_id[0] if isinstance(model.config.eos_token_id, list) else model.config.eos_token_id
			tryNum = 0
		except Exception as e:
			print(e)
			print('What can I say?')
			time.sleep(1)
	return model, processor, config


if __name__ == '__main__':
	disable_caching()
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str)
	parser.add_argument('--tokenizer', type=str)
	parser.add_argument('--csvP', type=str)
	parser.add_argument('--doSample', action='store_true')
	parser.add_argument('--answerOnly', action='store_true')
	parser.add_argument('--imgP', type=str)
	parser.add_argument('--evalJudge', type=str, nargs='+')
	parser.add_argument('--maxL', type=int, required=True)
	parser.add_argument('--evalData', type=str, choices=['sj', 'harmbench', 'harm'])
	args = parser.parse_args()
	if args.tokenizer is None:
		args.tokenizer = args.model
	modelN = args.model
	tokenizerN = args.tokenizer
	evalData = args.evalData
	imgP = args.imgP
	doSample = args.doSample
	answerOnly = args.answerOnly
	maxL = args.maxL
	evalJudge = args.evalJudge
	prompts = myUtil.loadEvalData(evalData)
	model, processor, config = loadVisualModel(modelN, tokenizerN)
	pilImg = PIL.Image.open(imgP) if imgP is not None else ttf.to_pil_image(torch.randint(0, 255, (3, 256, 256), dtype=torch.float32) / 255)
	with torch.no_grad():
		allComp = genLVLM(model, processor, prompts, [copy.deepcopy(pilImg) for _ in range(len(prompts))], maxL, doSample, model2thinkend.get(modelN, None) if answerOnly else None)
	del model
	torch.cuda.empty_cache()
	allScores = myUtil.eval([(p, r) for p, r in zip(prompts, allComp)], evalJudge)
	headerLine = ['Model', 'imgP', 'Data', 'doSample', 'maxL', 'answerOnly']
	valueLine = [modelN, imgP, evalData, doSample, maxL, answerOnly]
	for k, v in allScores.items():
		headerLine.append(k)
		valueLine.append(torch.tensor(v).float().mean().item())
	with open(args.csvP, 'a+', newline='') as f:
		csv.writer(f).writerows([headerLine, valueLine])
