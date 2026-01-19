import argparse
import csv

import torch
from datasets import disable_caching

import myUtil

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str)
	parser.add_argument('--tokenizer', type=str)
	parser.add_argument('--csvP', type=str)
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--doSample', action='store_true')
	parser.add_argument('--evalJudge', type=str, nargs='+')
	args = parser.parse_args()
	print(args)
	if args.tokenizer is None:
		args.tokenizer = args.model
	# load model & processor
	disable_caching()
	modelN = args.model
	verbose = args.verbose
	evalJudge = args.evalJudge

	prompts = myUtil.loadEvalData('benign')
	headerLine = ['Dataset', 'Model', 'Sample']
	valueLine = ['benign', modelN, args.doSample]
	with torch.no_grad():
		model, processor, config = myUtil.loadModel(modelN, args.tokenizer)
		allComp = myUtil.gen(model, processor, prompts, 512, None, args.doSample)
		del model
		torch.cuda.empty_cache()
		allScores = myUtil.eval([(p, r) for p, r in zip(prompts, allComp)], evalJudge)
		for k, v in allScores.items():
			headerLine.append(k)
			valueLine.append(torch.tensor(v).float().mean().item())
	with open(args.csvP, 'a+', newline='') as f:
		csv.writer(f).writerows([headerLine, valueLine])
