import argparse
import copy
import csv
import functools
import os

import torch
from datasets import disable_caching
from peft import PeftModel

import myUtil
from SCAV import classifier_manager, perturbation


def register_hooks(model, perturbations: list, layerIdxs):
	retHooks = []

	def _hook_fn(module, inputs, outputs, layer_idx, perturbs):
		for ppp in perturbs:
			outputs = ppp.get_perturbation(outputs, layer_idx, posi=posi)
		return outputs

	for i in layerIdxs:
		baseModel = model.model if not isinstance(model, PeftModel) else model.base_model.model.model
		retHooks.append(baseModel.layers[i].register_forward_hook(
			functools.partial(_hook_fn, layer_idx=i, perturbs=perturbations)
		))
	return retHooks


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str)
	parser.add_argument('--tokenizer', type=str)
	parser.add_argument('--evalPT', type=str)
	parser.add_argument('--csvP', type=str)
	parser.add_argument('--clfP', type=str)
	parser.add_argument('--evalClfr', type=str, choices=['all', 'first', 'last', 'best'])
	parser.add_argument('--layer', nargs='+', type=int, default=[])
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--doSample', action='store_true')
	parser.add_argument('--posi', type=str, required=True)
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
	hooks = []

	allClfr = classifier_manager.load_classifier_manager(clfP)
	prompts = myUtil.loadEvalData(evalData)
	headerLine = ['Dataset', 'Model', 'Sample', 'evalPT', 'ClfP', 'evalClfr']
	valueLine = [evalData, modelN, args.doSample, evalPT, os.path.split(clfP)[-1], args.evalClfr]
	attrs = clfP.split('_')
	with torch.no_grad():
		if args.evalClfr == 'all':
			clfr2test = {k: allClfr[k] for k in sorted(allClfr)}
		elif args.evalClfr == 'first':
			clfr2test = {k: allClfr[k] for k in sorted(allClfr)[:1]}
		elif args.evalClfr == 'best':
			clfr2test = {k: allClfr[k] for k in sorted(allClfr, key=lambda k: allClfr[k][0], reverse=True)[:1]}
		elif args.evalClfr == 'last':
			clfr2test = {k: allClfr[k] for k in sorted(allClfr, reverse=True)[:1]}
		else:
			clfr2test = None
		for iterNum, (score, clfr, negEachLayerProb, posEachLayerProb, usefulEachLayerProb) in list(clfr2test.items()):
			model, processor, config = myUtil.loadModel(modelN, args.tokenizer)
			if len(layerIdxs) == 1:
				layerIdxs.insert(0, -config.num_hidden_layers)
			layers = list(range(config.num_hidden_layers))[config.num_hidden_layers + layerIdxs[0]:config.num_hidden_layers + 1 + layerIdxs[1]]
			probType = negEachLayerProb
			eType = evalPT.split(' ')
			eType = eType[0]
			if eType in probType.keys():
				ept = copy.deepcopy(probType[eType])
			else:
				if float(eType) <= 0.5:
					ept = [probType['min'][i] + float(eType) / 0.5 * (0.5 - probType['min'][i]) for i in range(len(clfr.classifiers))]  # [min(max(float(eType), probType['min'][i]), posEachLayerProb['max'][i]) for i in range(len(clfr.classifiers))]
				else:
					ept = [0.5 + (float(eType) - 0.5) / 0.5 * (posEachLayerProb['max'][i] - 0.5) for i in range(len(clfr.classifiers))]  # [min(max(float(eType), probType['min'][i]), posEachLayerProb['max'][i]) for i in range(len(clfr.classifiers))]
			clfrN = f'clfr{iterNum}, {score}'
			print(clfrN)
			pert = perturbation.Perturbation(clfr, target_probability=ept)

			hooks = register_hooks(model, [pert], layers)
			allComp = myUtil.gen(model, processor, prompts, 512, None, args.doSample)
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
