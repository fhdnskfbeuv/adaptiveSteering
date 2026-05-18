import argparse
import csv
import os

import torch
from datasets import disable_caching

import myUtil
import tqdm


def easyGen(model, processor, text: str, maxL=128, doSample=False, endThink=None):
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
	if hasattr(model, 'adaHooks'):
		model.adaHooks[1].reset()
		model.generate(**inputs, max_new_tokens=2, do_sample=doSample)
		model.adaHooks[1].isRecord = False
	generated_ids = model.generate(**inputs, max_new_tokens=maxL, do_sample=doSample)
	trimmedIDs = []
	for i in range(len(generated_ids)):
		trimmedIDs.append(generated_ids[i][inputs['input_ids'][i].shape[0]:])
	completion = processor.batch_decode(
		trimmedIDs, skip_special_tokens=True, clean_up_tokenization_spaces=False
	)[0]
	fullStr = processor.batch_decode(
		generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
	)[0]
	if endThink is not None:
		completion = completion.split(endThink)[-1]
	return fullStr, completion


def gen(model, processor, prompts, maxL, doSample=False, endThink=None):
	allCompletion = []
	with tqdm.tqdm(prompts, total=len(prompts), dynamic_ncols=True) as pbar:
		for i, prompt in enumerate(prompts):
			fullStr, completion = easyGen(model, processor, prompt, maxL, doSample, endThink)
			# print(f'{completion}')
			allCompletion.append(completion)
			pbar.update()
	return allCompletion


model2thinkend = {
	'Qwen/Qwen3-4B-Thinking-2507': "</think>",
	'zai-org/GLM-4.6V-Flash': "</think>",
}

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str)
	parser.add_argument('--tokenizer', type=str)
	parser.add_argument('--evalPT', type=str)
	parser.add_argument('--csvP', type=str)
	parser.add_argument('--clfP', type=str, default='None')
	parser.add_argument('--evalClfr', type=str, choices=['all', 'first', 'last', 'best'])
	parser.add_argument('--layer', nargs='+', type=int, default=[])
	parser.add_argument('--maxL', type=int)
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--doSample', action='store_true')
	parser.add_argument('--answerOnly', action='store_true')
	parser.add_argument('--posi', type=str)
	parser.add_argument('--evalJudge', type=str, nargs='+')
	parser.add_argument('--evalData', type=str, choices=['sj', 'harmbench', 'harm'])
	args = parser.parse_args()
	print(args)
	if args.tokenizer is None:
		args.tokenizer = args.model
	# load model & processor
	disable_caching()
	modelN = args.model
	clfP = args.clfP
	layerIdxs = args.layer
	verbose = args.verbose
	evalPT = args.evalPT
	evalJudge = args.evalJudge
	posi = args.posi
	evalData = args.evalData
	evalClfr = args.evalClfr
	maxL = args.maxL
	answerOnly = args.answerOnly
	hooks = []
	prompts = myUtil.loadEvalData(evalData)
	headerLine = ['Dataset', 'Model', 'Sample', 'evalPT', 'ClfP', 'evalClfr', 'maxL', 'answerOnly']
	valueLine = [evalData, modelN, args.doSample, evalPT, os.path.split(clfP)[-1], args.evalClfr, maxL, answerOnly]
	with torch.no_grad():
		model, processor, config = myUtil.loadModel(modelN, args.tokenizer)
		clfrN = 'N/A'
		if clfP != 'None':
			if len(layerIdxs) == 1:
				layerIdxs.insert(0, -config.num_hidden_layers)
			layers = list(range(config.num_hidden_layers))[config.num_hidden_layers + layerIdxs[0]:config.num_hidden_layers + 1 + layerIdxs[1]]
			hooks, clfrN = myUtil.scavHook(model, clfP, evalClfr, layers, evalPT, posi)
			print(clfrN)
		allComp = gen(model, processor, prompts, maxL, args.doSample, model2thinkend.get(modelN, None) if answerOnly else None)
		for hook in hooks:
			hook.remove()
		del model
		torch.cuda.empty_cache()
		allScores = myUtil.eval([(p, r) for p, r in zip(prompts, allComp)], evalJudge)
		for k, v in allScores.items():
			headerLine.append(clfrN + ';' + k)
			valueLine.append(torch.tensor(v).float().mean().item())
		with open(args.csvP.replace('.csv', f'_{evalPT}.csv'), 'a+', newline='') as f:
			csv.writer(f).writerows([headerLine, valueLine])
