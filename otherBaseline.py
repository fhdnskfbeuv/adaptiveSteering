import argparse
import csv
import gc

import torch
from datasets import disable_caching
from peft import PeftModel
from strong_reject import load_datasets
from transformers import AutoProcessor

import angular
import baselines
import myUtil
from baselines import refusalDirection
from pipeline.utils.hook_utils import add_hooks

if __name__ == '__main__':
	disable_caching()
	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type=str, choices=['rep', 'scav', 'rd', 'angular', 'lascav'])
	parser.add_argument('--model', type=str)
	parser.add_argument('--tokenizer', type=str)
	parser.add_argument('--evalData', type=str, choices=['sj', 'harmbench', 'harm'])
	parser.add_argument('--trainsize', type=float, default=1.0)
	parser.add_argument('--strength', type=float)
	parser.add_argument('--evalJudge', type=str, nargs='+')
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--doSample', action='store_true')
	parser.add_argument('--full', action='store_true')
	parser.add_argument('--csvP', type=str)
	args = parser.parse_args()
	print(args)
	method = args.method
	modelN = args.model
	evalJudge = args.evalJudge
	verbose = args.verbose
	strength = args.strength
	trainsize = args.trainsize
	evalData = args.evalData
	full = args.full
	if args.tokenizer is None:
		args.tokenizer = args.model
	prompts = myUtil.loadEvalData(evalData)
	insts = myUtil.loadDataset(r'./instructions/', int(100 * trainsize), int(100 * trainsize), 100 - int(100 * trainsize), 100 - int(100 * trainsize), full, trainsize)
	hooks = []
	with torch.no_grad():
		if method == 'rep':
			model, processor, config = myUtil.loadModel(modelN, args.tokenizer)

			myUtil.init_rep_control(model if not isinstance(model, PeftModel) else model.base_model.model, processor, insts['train'][0] + insts['val'][0], insts['train'][1] + insts['val'][1], layer_id=list(range(-11, -21, -1)), repe_coeff=strength)
		elif method == 'rd':
			model, processor, config = myUtil.loadModel(modelN, args.tokenizer)
			model = refusalDirection(modelN, processor, model if not isinstance(model, PeftModel) else model.base_model.model)
			data = (insts['train'][0], insts['train'][1], insts['val'][0], insts['val'][1])
			haha = model.getDirection(data)
			if haha == 'NOT MY FAULT!':
				headerLine = ['Dataset', 'Model', "doSample", 'method', 'Strength', 'TrainSize', 'Crashed!!!', 'Crashed!!!', 'Crashed!!!', 'Crashed!!!']
				valueLine = [evalData, modelN, args.doSample, method, strength, trainsize]
				with open(args.csvP, 'a+', newline='') as f:
					csv.writer(f).writerows([headerLine, valueLine])
				exit(1)
			model.getHooks()
		elif method == 'scav':
			model, processor, config = myUtil.loadModel(modelN, args.tokenizer)
			scav = baselines.vanillaSCAV(modelN, model if not isinstance(model, PeftModel) else model.base_model.model, processor, config)
			scav.prepare(insts['train'][0], insts['train'][1], insts['val'][0], insts['val'][1], strength)
		elif method == 'lascav':
			model, processor, config = myUtil.loadModel(modelN, args.tokenizer)
			scav = baselines.lastAllSCAV(modelN, model if not isinstance(model, PeftModel) else model.base_model.model, processor, config)
			scav.prepare(insts['train'][0], insts['train'][1], insts['val'][0], insts['val'][1], strength)
		elif method == 'angular':
			angularManager = angular.AngularSteering()
			if modelN in myUtil.lora2base.keys():
				model, processor, config = myUtil.loadModel(modelN, args.tokenizer)
				angularManager.prepare(
					modelN if modelN not in myUtil.lora2base.keys() else myUtil.lora2base[modelN],
					insts['train'][0], insts['train'][1],
					processor,
					model.base_model.model
				)
			else:
				angularManager.prepare(
					modelN,
					insts['train'][0], insts['train'][1],
					AutoProcessor.from_pretrained(args.tokenizer, token='os.getenv('HF_TOKEN', default=None)')
				)
			model, processor, config = myUtil.loadModel(modelN, args.tokenizer)
			hooks = angularManager.addHook(model if not isinstance(model, PeftModel) else model.base_model.model, 180)
		else:
			print(f'{method} not implemented yet')
			exit(1)

	judgeM = None
	with torch.no_grad():
		headerLine = ['Dataset', 'Model', "doSample", 'method', 'Strength', 'TrainSize']
		valueLine = [evalData, modelN, args.doSample, method, strength, trainsize]
		if method in ['rep', 'scav', 'angular', 'lascav']:
			allComp = myUtil.gen(model, processor, prompts, 512, None, args.doSample)
			del model
			torch.cuda.empty_cache()
			allScores = myUtil.eval([(p, r) for p, r in zip(prompts, allComp)], evalJudge)
			for k, v in allScores.items():
				headerLine.append(k)
				valueLine.append(torch.tensor(v).float().mean().item())
		elif method == 'rd':
			with add_hooks(module_forward_pre_hooks=model.ablation_fwd_pre_hooks, module_forward_hooks=model.ablation_fwd_hooks):
				allAblaComp = myUtil.gen(model.model.model, processor, prompts, 512, None, args.doSample)
			with add_hooks(module_forward_pre_hooks=model.actadd_fwd_pre_hooks, module_forward_hooks=model.actadd_fwd_hooks):
				allAddComp = myUtil.gen(model.model.model, processor, prompts, 512, None, args.doSample)
			del model.model.model
			del model
			torch.cuda.empty_cache()
			gc.collect()
			allScores = myUtil.eval([(p, r) for p, r in zip(prompts + prompts, allAblaComp + allAddComp)], evalJudge)
			for k, v in allScores.items():
				headerLine.append(k + ' (Ablation)')
				valueLine.append(torch.tensor(v[:len(allAblaComp)]).float().mean().item())
			for k, v in allScores.items():
				headerLine.append(k + ' (Additive)')
				valueLine.append(torch.tensor(v[len(allAblaComp):]).float().mean().item())
	with open(args.csvP, 'a+', newline='') as f:
		csv.writer(f).writerows([headerLine, valueLine])
