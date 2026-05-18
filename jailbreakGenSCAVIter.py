import argparse
import copy
import functools
import json
import os

import numpy as np
import torch
import tqdm
from colorama import Fore, Style
from datasets import disable_caching
from peft import PeftModel
from strong_reject import evaluate

import myJudge
import myUtil
from SCAV import classifier_manager, embedding_manager, perturbation


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


def getEmb(model, messages: list[list[dict]], processor, maxL, prefix=None, embType='last'):
	lastEmbdsManager = embedding_manager.EmbeddingManager()
	completions = []
	for i, message in tqdm.tqdm(enumerate(messages), total=len(messages), dynamic_ncols=True):
		inputs = processor.apply_chat_template(message,
											   tokenize=True,
											   return_tensors="pt",
											   return_dict=True,
											   add_generation_prompt=True).to(model.device)  # Prepare texts for processing
		if prefix is not None:  # [1, L]
			inputs['input_ids'] = torch.concat([inputs['input_ids'], prefix.repeat(inputs['input_ids'].shape[0], 1).to(inputs['input_ids'])], dim=1)
			inputs['attention_mask'] = torch.concat([inputs['attention_mask'], torch.ones((inputs['attention_mask'].shape[0], prefix.shape[1])).to(inputs['attention_mask'])], dim=1)
		# Inference: Generation of the output
		output = model.generate(**inputs, max_new_tokens=maxL,
								output_hidden_states=True,
								return_dict_in_generate=True, do_sample=False)
		hiddenStates = output.hidden_states  # [maxL, layers + 1, 1, L, D]
		generated_ids = output.sequences

		trimmedIDs = []
		for i in range(len(generated_ids)):
			trimmedIDs.append(generated_ids[i][inputs['input_ids'][i].shape[0]:])
		completeion = processor.batch_decode(
			trimmedIDs, skip_special_tokens=True, clean_up_tokenization_spaces=False
		)[0]
		if verbose:
			print(completeion)
		completions.append(completeion)
		for j in range(len(hiddenStates[0]) - 1):  # loop for each layer
			if j >= len(lastEmbdsManager.variedLenLayers):
				lastEmbdsManager.variedLenLayers.append([])
			if embType == 'last':
				lastEmbdsManager.variedLenLayers[j].append(hiddenStates[0][j + 1][:, -1, :].float().clone().cpu())
			elif embType == 'prompt':
				lastEmbdsManager.variedLenLayers[j].append(torch.mean(hiddenStates[0][j + 1][0, :, :].float(), dim=0, keepdim=True).clone().cpu())
			elif embType == 'response':
				lastEmbdsManager.variedLenLayers[j].append(torch.mean(torch.concat([hiddenStates[t][j + 1][:, -1, :].float() for t in range(len(hiddenStates))], dim=0), dim=0, keepdim=True).clone().cpu())
			elif embType == 'all':
				lastEmbdsManager.variedLenLayers[j].append(torch.mean(torch.concat([hiddenStates[t][j + 1][0, :, :].float() for t in range(len(hiddenStates))], dim=0), dim=0, keepdim=True).clone().cpu())
			else:
				print(f'{embType} not implemented')
				exit(1)
	lastEmbdsManager.trainMode()
	return lastEmbdsManager, completions  #dict[str, dict[float, embdsManager]]


def getMessages(texts, systemPrompt=None):
	messages = []
	for text in texts:
		if systemPrompt is not None:
			message = [{"role": "system", "content": systemPrompt}, {"role": "user", "content": text}]
		else:
			message = [{"role": "user", "content": text}]

		messages.append(message)
	return messages


def calculateProba(embds: embedding_manager.EmbeddingManager, clfrs: classifier_manager.ClassifierManager, sampleWeight):
	probs = []
	embds.trainMode()
	if sampleWeight is not None:
		print(f'CalculateProba: Available Samples {np.sum(sampleWeight[-1] != 0)}')
	for i, clfr in enumerate(clfrs.classifiers):
		prob = clfr.predict_proba(embds.layers[i]).cpu().numpy()[sampleWeight[i] != 0] if sampleWeight is not None else clfr.predict_proba(embds.layers[i]).cpu().numpy()
		nonZeroMin = np.min(prob[prob != 0]).item()  # zero will induce numerical issue
		q1 = np.percentile(prob, 25)
		mid = np.percentile(prob, 50)
		q3 = np.percentile(prob, 75)
		mean = prob.mean()
		std = prob.std()
		low3Sigma = np.clip(mean - 3 * std, 0, 1)
		high3Sigma = np.clip(mean + 3 * std, 0, 1)
		probs.append((max(nonZeroMin, prob.min().item()),
					  max(nonZeroMin, q1.item()),
					  max(nonZeroMin, mid.item()),
					  max(nonZeroMin, q3.item()),
					  max(nonZeroMin, prob.max().item()),
					  max(nonZeroMin, low3Sigma.item()),
					  max(nonZeroMin, mean.item()),
					  max(nonZeroMin, high3Sigma.item())))
	return {
		'min': [_[0] for _ in probs],
		'q1': [_[1] for _ in probs],
		'mid': [_[2] for _ in probs],
		'q3': [_[3] for _ in probs],
		'max': [_[4] for _ in probs],
		'low3sigma': [_[5] for _ in probs],
		'mean': [_[6] for _ in probs],
		'high3sigma': [_[7] for _ in probs],
	}


def filterEmbds(steeredEmbds, prompts, completions, judgeF, thres):
	maliciousScore = 0
	oriJudgeRes = []
	assert len(steeredEmbds.variedLenLayers[0]) == len(prompts)
	with tqdm.tqdm(enumerate(zip(prompts, completions)), total=len(completions), desc=f"Filtering. Thres: {thres}; Average Score: {maliciousScore}", dynamic_ncols=True) as pbar:
		for i, (prompt, completion) in enumerate(zip(prompts, completions)):
			score = judgeF(prompt, completion)
			oriJudgeRes.append(score)
			maliciousScore = torch.tensor(oriJudgeRes).float().mean().item()
			pbar.set_description(f"Filtering. Thres: {thres}; Average Score: {maliciousScore}; Current Score: {oriJudgeRes[-1]}")
			pbar.update()
			assert 0 <= oriJudgeRes[-1] <= 1
	return steeredEmbds, oriJudgeRes, maliciousScore


def augmentEmbds(embdsPool, weightPool, steeredEmbds, judgeRes, oriJudgeRes, reweight, thres, selectedEmbds=None):
	for i in range(len(judgeRes)):
		if judgeRes[i] == 0:
			if reweight and thres is not None:
				weight = (thres - oriJudgeRes[i]) / thres
				if weight == 0:
					weight += 1e-6
			else:
				weight = 1.0
			for j in range(len(embdsPool.variedLenLayers)):  # each layer
				embdsPool.variedLenLayers[j].append(steeredEmbds.variedLenLayers[j][i].clone())
				if selectedEmbds is not None:
					selectedEmbds.variedLenLayers[j].append(steeredEmbds.variedLenLayers[j][i].clone())
				weightPool[j] = np.concatenate([weightPool[j], np.zeros((steeredEmbds.variedLenLayers[j][i].shape[0],)) + weight], axis=0)
	embdsPool.trainMode()  # [B, D]
	return embdsPool, weightPool


def getFilteredDataset(modelN, dataN, model, processor, refusalTrain, nonRefusalTrain, refusalVal, nonRefusalVal):
	if os.path.exists(r'./instructions/filteredSJF.json'):
		with open(r'./instructions/filteredSJF.json', 'r+') as f:
			data = json.load(f)
	else:
		data = {}
		with open(r'./instructions/filteredSJF.json', 'w+') as f:
			json.dump(data, f)
	if dataN in data.keys() and modelN in data[dataN].keys():
		return data[dataN][modelN]['refusalTrain'], data[dataN][modelN]['nonRefusalTrain'], data[dataN][modelN]['refusalVal'], data[dataN][modelN]['nonRefusalVal']
	fJudge = myJudge.StrongRejectJudge
	if dataN not in data.keys():
		data[dataN] = {}
	data[dataN][modelN] = {
		'refusalTrain': [],
		'nonRefusalTrain': [],
		'refusalVal': [],
		'nonRefusalVal': []
	}
	for prompt in tqdm.tqdm(refusalTrain, total=len(refusalTrain), dynamic_ncols=True):
		data[dataN][modelN]['refusalTrain'].append(prompt)
	for prompt in tqdm.tqdm(nonRefusalTrain, total=len(nonRefusalTrain), dynamic_ncols=True):
		_, completion = myUtil.easyGen(model, processor, prompt, 256)
		if fJudge(prompt, completion) > 0.5:
			data[dataN][modelN]['nonRefusalTrain'].append(prompt)
	for prompt in tqdm.tqdm(refusalVal, total=len(refusalVal), dynamic_ncols=True):
		data[dataN][modelN]['refusalVal'].append(prompt)
	for prompt in tqdm.tqdm(nonRefusalVal, total=len(nonRefusalVal), dynamic_ncols=True):
		_, completion = myUtil.easyGen(model, processor, prompt, 256)
		if fJudge(prompt, completion) > 0.5:
			data[dataN][modelN]['nonRefusalVal'].append(prompt)
	with open(r'./instructions/filteredSJF.json', 'r+') as f:
		newData = json.load(f)
		if dataN not in newData.keys():
			newData[dataN] = {}
		newData[dataN][modelN] = data[dataN][modelN]
	with open(r'./instructions/filteredSJF.json', 'w+') as f:
		json.dump(newData, f)
	jjj = evaluate.cached_models["strongreject_finetuned"][0]
	del jjj
	torch.cuda.empty_cache()
	return data[dataN][modelN]['refusalTrain'], data[dataN][modelN]['nonRefusalTrain'], data[dataN][modelN]['refusalVal'], data[dataN][modelN]['nonRefusalVal']


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str)
	parser.add_argument('--tokenizer', type=str)
	parser.add_argument('--pt', type=str)
	parser.add_argument('--evalPT', type=str, nargs='+')
	parser.add_argument('--softThres', type=float, nargs='+')
	parser.add_argument('--reweight', action='store_true')
	parser.add_argument('--maxIter', type=int)
	parser.add_argument('--trainL', type=int)
	parser.add_argument('--embType', type=str, choices=['last', 'prompt', 'response', 'all', 'responseLast'])
	parser.add_argument('--penalty', type=str, default='l2')
	parser.add_argument('--saveDir', type=str)
	parser.add_argument('--judge', type=str, nargs='+')
	parser.add_argument('--layer', nargs='+', type=int, default=[])
	parser.add_argument('--tvNum', nargs='*', type=int, default=[50, 50, 50, 50])
	parser.add_argument('--verbose', action='store_true')
	parser.add_argument('--gpuLR', action='store_true')
	parser.add_argument('--val', action='store_true')
	parser.add_argument('--full', action='store_true')
	parser.add_argument('--filterData', action='store_true')
	parser.add_argument('--posi', type=str)
	args = parser.parse_args()
	print(args)
	if args.tokenizer is None:
		args.tokenizer = args.model
	# load model & processor
	disable_caching()
	modelN = args.model
	pt = args.pt
	layerIdxs = args.layer
	softThres = args.softThres
	maxIter = args.maxIter
	trainL = args.trainL
	judgeN = args.judge
	reweight = args.reweight
	verbose = args.verbose
	evalPTs = args.evalPT
	gpuLR = args.gpuLR
	embType = args.embType
	tvNum = args.tvNum
	full = args.full
	penalty = None if args.penalty == 'None' else args.penalty
	filterData = args.filterData
	posi = args.posi
	assert len(tvNum) == 4

	successDL = False
	hooks = []
	model, processor, config = myUtil.loadModel(modelN, args.tokenizer)
	insts = myUtil.loadDataset(r'./instructions/', tvNum[0], tvNum[1], tvNum[2], tvNum[3], full=full)
	if filterData:
		insts['train'][0], insts['train'][1], insts['val'][0], insts['val'][1] = getFilteredDataset(modelN, f'{tvNum[0]}_{tvNum[1]}_{tvNum[2]}_{tvNum[3]}',
																									model, processor,
																									insts['train'][0],
																									insts['train'][1], insts['val'][0], insts['val'][1])
	saveDir = os.path.join(args.saveDir, modelN.replace('./', '').replace('/', '_'))
	saveDir = os.path.join(saveDir, f'harm{[tvNum[0], tvNum[2]]}_benign{[tvNum[1], tvNum[3]]}') if not full else os.path.join(saveDir, 'full')
	os.makedirs(saveDir, exist_ok=True)
	judgeM = None
	if len(layerIdxs) == 1:
		layerIdxs.insert(0, -config.num_hidden_layers)
	with torch.no_grad():
		posTrainMessages = getMessages(insts['train'][0])
		posValMessages = getMessages(insts['val'][0])
		negTrainMessages = getMessages(insts['train'][1])
		negValMessages = getMessages(insts['val'][1])
		posTrainEmbds, posTrainCompletions = getEmb(model, posTrainMessages, processor, trainL if embType in ['all', 'response', 'responseLast'] else 1, embType=embType)
		# posValEmbds, posValCompletions = getEmb(model, posValMessages, processor, depth + 1, depth)
		negTrainEmbds, negTrainCompletions = getEmb(model, negTrainMessages, processor, trainL if embType in ['all', 'response', 'responseLast'] else 1, embType=embType)
		# negValEmbds, negValCompletions = getEmb(model, negValMessages, processor, trainL if embType in ['all', 'response', 'responseLast'] else 1, embType=embType)
	usefulEmbds = embedding_manager.EmbeddingManager()
	usefulEmbds.variedLenLayers = [[] for _ in range(len(posTrainEmbds.variedLenLayers))]
	if len(layerIdxs) == 1:
		layerIdxs.insert(0, -config.num_hidden_layers)

	print(f'Layer: {layerIdxs}')

	clfP = os.path.join(saveDir,
						f'judge{judgeN[0]}_embType{embType}_posi{posi}_filterData{filterData}_layer{layerIdxs}_penalty{penalty}_gpuLR{gpuLR}_reweight{reweight}_maxIter{maxIter}_trainL{trainL}_pt{pt}_softThres{softThres}_clfrs.pt'.replace(
							'/',
							'-'))
	posSampleWeight = [np.ones((posTrainEmbds.layers[j].shape[0],)) for j in range(len(posTrainEmbds.layers))]
	negSampleWeight = [np.ones((negTrainEmbds.layers[j].shape[0],)) for j in range(len(negTrainEmbds.layers))]
	allClfr = {}
	successDL = False
	judgeM, judgeF = myUtil.loadJudge(judgeN)
	clfr = None
	initialMean = None
	for currentIter in range(maxIter):
		classBalancedWeight = []
		finalPNWeight = {'pos': [], 'neg': []}
		for j in range(len(posSampleWeight)):  # looping layer
			p = copy.deepcopy(posSampleWeight[j])
			n = copy.deepcopy(negSampleWeight[j])
			totalWeight = p.shape[0] + n.shape[0]
			posClassWeight = totalWeight / (2 * p.sum())
			negClassWeight = totalWeight / (2 * n.sum())
			www = np.concatenate([p * posClassWeight, n * negClassWeight], axis=0)
			finalPNWeight['pos'].append(p * posClassWeight)
			finalPNWeight['neg'].append(n * negClassWeight)
			if np.any(np.isnan(www)):
				print(Fore.RED, 'nan?????????????????????????????????????', Style.RESET_ALL)
				www = np.nan_to_num(www, nan=0.0)
			classBalancedWeight.append(www)
		clfr = classifier_manager.ClassifierManager('', gpuLR)
		clfr.fit(posTrainEmbds, negTrainEmbds, posTrainEmbds, negTrainEmbds, sampleWeight=classBalancedWeight, penalty=penalty)

		with torch.no_grad():
			isUsefulEmpty = usefulEmbds.isEmpty()
			negEachLayerProb = calculateProba(negTrainEmbds, clfr, finalPNWeight['neg'])  # [(min, q1, mid, q3, max)] * layerNum
			posEachLayerProb = calculateProba(posTrainEmbds, clfr, finalPNWeight['pos'])  # [(min, q1, mid, q3, max)] * layerNum
			usefulEachLayerProb = calculateProba(usefulEmbds, clfr, None) if not isUsefulEmpty else copy.deepcopy(negEachLayerProb)
			if full:
				allClfr[currentIter] = (1.0, copy.deepcopy(clfr), negEachLayerProb, posEachLayerProb, usefulEachLayerProb)
				allClfr = {k: allClfr[k] for k in sorted(allClfr, key=lambda k: allClfr[k][0], reverse=True)}
				torch.save(allClfr, clfP)
				print(f'save to {clfP}')
				break
			eType = pt.split(' ')
			eType = eType[0]
			probType = negEachLayerProb
			if eType in probType.keys():
				ept = copy.deepcopy(probType[eType])
			else:
				if float(eType) <= 0.5:
					ept = [probType['min'][i] + float(eType) / 0.5 * (0.5 - probType['min'][i]) for i in range(len(clfr.classifiers))]  # [min(max(float(eType), probType['min'][i]), posEachLayerProb['max'][i]) for i in range(len(clfr.classifiers))]
				else:
					ept = [0.5 + (float(eType) - 0.5) / 0.5 * (posEachLayerProb['max'][i] - 0.5) for i in range(len(clfr.classifiers))]  # [min(max(float(eType), probType['min'][i]), posEachLayerProb['max'][i]) for i in range(len(clfr.classifiers))]
			pert = perturbation.Perturbation(clfr, target_probability=ept)
			for hook in hooks:
				hook.remove()
			hooks = register_hooks(model, [pert], list(range(config.num_hidden_layers))[config.num_hidden_layers + layerIdxs[0]:config.num_hidden_layers + 1 + layerIdxs[1]])
			sampledPosTrainPrompts = copy.deepcopy(insts['train'][0])
			sampledPosTrainMessages = getMessages(sampledPosTrainPrompts)
			clfr.setDirection(True)
			print(f'{Fore.RED}Probe PT: {pert.target_probability}{Style.RESET_ALL}')
			steeredPosEmbds, posCompletions = getEmb(model, sampledPosTrainMessages, processor, trainL, embType=embType)
			filteredSteeredPosEmbds, posOriJudgeRes, posScore = filterEmbds(steeredPosEmbds,
																			sampledPosTrainPrompts,
																			posCompletions,
																			judgeF,
																			softThres[0])
			posFailedNum = np.sum(np.array([int(r < softThres[0]) for r in posOriJudgeRes]))
			usefulRes = [int(r <= softThres[1]) for r in posOriJudgeRes]
			usefulNum = np.sum(1 - np.array(usefulRes))
			posTrainEmbds, posSampleWeight = augmentEmbds(posTrainEmbds, posSampleWeight, filteredSteeredPosEmbds,
														  [int(r >= softThres[0]) for r in posOriJudgeRes], posOriJudgeRes, reweight,
														  softThres[0])
			negTrainEmbds, negSampleWeight = augmentEmbds(negTrainEmbds, negSampleWeight, filteredSteeredPosEmbds,
														  usefulRes, [1 - r for r in posOriJudgeRes], reweight, 1 - softThres[1], usefulEmbds)
			clfr.completion = []
			clfr.completion.append([{'Prompt': p, 'Response': r, 'Score': s} for p, r, s in zip(sampledPosTrainPrompts, posCompletions, posOriJudgeRes)])

			if verbose:
				for c in posCompletions:
					print(c[:min(len(c), 256)])
			print(f"{Fore.RED}Pos Total: {len(posCompletions)}; Refused: {posFailedNum}; Accept: {len(posCompletions) - posFailedNum}; Useful: {usefulNum}{Style.RESET_ALL}")
		maxPosScore = posScore
		if args.val:
			maxPosScore = 0
			with torch.no_grad():
				print('Val:')
				for evalPT in evalPTs:
					eType = evalPT.split(' ')
					eType = eType[0]
					if eType in probType.keys():
						ept = copy.deepcopy(probType[eType])
					else:
						if float(eType) <= 0.5:
							ept = [probType['min'][i] + float(eType) / 0.5 * (0.5 - probType['min'][i]) for i in range(len(clfr.classifiers))]  # [min(max(float(eType), probType['min'][i]), posEachLayerProb['max'][i]) for i in range(len(clfr.classifiers))]
						else:
							ept = [0.5 + (float(eType) - 0.5) / 0.5 * (posEachLayerProb['max'][i] - 0.5) for i in range(len(clfr.classifiers))]
					oldTP = copy.deepcopy(pert.target_probability)
					pert.target_probability = ept
					valData = insts['val'][0][:50]
					print(f'{Fore.RED}Val Probe PT: {pert.target_probability}{Style.RESET_ALL}')
					ccc, allRes = myUtil.GenAndEval(model, processor, judgeF, valData, 256, None, False)
					posScore = torch.tensor(allRes).float().mean().item()
					pert.target_probability = oldTP
					clfr.completion.append([{'Prompt': p, 'Response': r, 'Score': s} for p, r, s in zip(valData, ccc, allRes)])
					maxPosScore = max(maxPosScore, posScore)
		allClfr[currentIter] = (maxPosScore, copy.deepcopy(clfr), negEachLayerProb, posEachLayerProb, usefulEachLayerProb)
		allClfr = {k: allClfr[k] for k in sorted(allClfr, key=lambda k: allClfr[k][0], reverse=True)}
		torch.save(allClfr, clfP)
		print(Fore.RED, [allClfr[k][0] for k in allClfr.keys()], Style.RESET_ALL)
		print(f'save to {clfP}')
		if posFailedNum == 0 and usefulNum == 0:
			print(f"{Fore.RED}Early Stop!!!!{Style.RESET_ALL}")
			break

	allClfr = {k: allClfr[k] for k in sorted(allClfr, key=lambda k: allClfr[k][0], reverse=True)}
	for clfr in allClfr.values():
		clfr[1].setDirection(True)
	torch.save(allClfr, clfP)
