import argparse
import json
import os
import random

import torch

import baselines
import myUtil

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str)
	parser.add_argument('--tokenizer', type=str)
	args = parser.parse_args()
	if args.tokenizer is None:
		args.tokenizer = args.model
	modelN = args.model
	model, processor, config = myUtil.loadModel(modelN, args.tokenizer)
	dirP = r'./instructions/'
	harmful = myUtil.extractFromJson(os.path.join(dirP, 'harmful_train.json'), "instruction")
	harmless = myUtil.extractFromJson(os.path.join(dirP, 'harmless_train.json'), "instruction")
	allAcc = []
	scav = baselines.vanillaSCAV(modelN, model, processor, config)
	with torch.no_grad():
		allNorms = scav.Norm((harmful + harmless)[:500])
		for i in range(50):
			sampledHarmful, sampledHarmless = random.sample(harmful, 100), random.sample(harmless, 100)
			posTrainPrompt, posValPrompt, negTrainPrompt, negValPrompt = sampledHarmful[:50], sampledHarmful[50:], sampledHarmless[:50], sampledHarmless[50:]
			acc = scav.Acc(posTrainPrompt, negTrainPrompt, posValPrompt, negValPrompt)
			print(acc)
			allAcc.append(acc)
	if os.path.exists(r'./scavACC.json'):
		with open(r'./scavACC.json', 'r+') as f:
			data = json.load(f)
	else:
		data = {}
		with open(r'./scavACC.json', 'w+') as f:
			json.dump(data, f)
	data[modelN] = {'acc': allAcc, 'norm': allNorms}
	with open(r'./scavACC.json', 'w+') as f:
		json.dump(data, f)


