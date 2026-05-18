import argparse
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import matplotlib.patches as mpatches
import seaborn as sns
from SCAV import classifier_manager
import matplotlib

def plot_line_boxplot(
	xs, ys, labels, x_label, y_label, save_path, nCols,
):
	num_lines = len(xs)
	fig, axes = plt.subplots(
		ceil(num_lines / nCols), nCols,
		figsize=(4096 / 300, 4096 / 300),
	)
	totalSubFig = ceil(num_lines / nCols) * nCols
	axes = axes.flatten().tolist()
	for i in range(num_lines, totalSubFig):
		axes[i].set_visible(False)
	for i, (x, y) in enumerate(zip(xs, ys)):
		box = axes[i].violinplot(y,
						  	positions=x,
						  	# patch_artist=True,
						  	widths=0.75,
							# whis=0.0,
							showmedians=True,
						  	showextrema=True,
						  	# flierprops=dict(marker='o', markersize=2)
						)
		# box = axes[i].boxplot(y,
		# 					  positions=x,
		# 					  patch_artist=True,
		# 					  widths=0.75,
		# 					  whis=0.0,
		# 					  showfliers=True,
		# 					  flierprops=dict(marker='o', markersize=2)
		# 					  )
		# for pos, values in zip(x, y):
		# 	axes[i].text(pos, 1.1, f'{np.median(values).item():.2f}', fontsize=5, color='red')
		box['cbars'].set_color('red')
		box['cbars'].set_linewidth(2)
		box['cmins'].set_color('red')
		box['cmaxes'].set_color('red')
		box['cmedians'].set_color('red')
		box['cmedians'].set_linewidth(2)
		for pc in box['bodies']:
			# pc.set_linewidth(2)
			pc.set_alpha(1.0)
		# axes[i].plot(x, [np.median(yy).item() for yy in y],
		# 		 '-',
		# 		 linewidth=2,
		# 		 # markersize=4,
		# 		 color='red',
		# 		 alpha=0.8)
		axes[i].set_title(labels[i], fontsize=15)
		axes[i].set_ylabel("")
		axes[i].grid(True, alpha=0.8)
		axes[i].set_ylim(-0.05, 1.2)
		if i % nCols == 0:
			axes[i].set_ylabel(y_label, fontsize=15, fontweight='bold')
		if i + nCols >= num_lines:
			axes[i].set_xlabel(x_label, fontsize=15, fontweight='bold')
	plt.tight_layout()
	if save_path:
		if not save_path.endswith('.pdf'):
			save_path = save_path.rsplit('.', 1)[0] + '.pdf'
		plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
		print(f"Main figure saved to: {save_path}")
		print(f"Figure size: {4096}×{4096} inches, DPI: {300}")
		print(f"Output resolution: {int(4096 * 300)}×{int(4096 * 300)} pixels")


if __name__ == '__main__':
	matplotlib.rcParams['pdf.fonttype'] = 42
	parser = argparse.ArgumentParser()
	parser.add_argument('--l', type=str)
	parser.add_argument('--h', type=str)
	args = parser.parse_args()
	l, h = args.l, args.h
	models = [
			f"./iterSCAVWeight/cais_zephyr_7b_r2d2/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/GraySwanAI_Llama-3-8B-Instruct-RR/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/GraySwanAI_Mistral-7B-Instruct-RR/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/lapisrocks_Llama-3-8B-Instruct-TAR-Refusal/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/LLM-LAT_robust-llama3-8b-instruct/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/thkim0305_RepBend_Llama3_8B/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/thkim0305_RepBend_Mistral_7B/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/thu-coai_Mistral-7B-Instruct-v0.2-safeunlearning/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/thu-coai_vicuna-7b-v1.5-safeunlearning/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/Unispac_Gemma-2-9B-IT-With-Deeper-Safety-Alignment/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-42, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/Unispac_Llama2-7B-Chat-Augmented/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
			f"./iterSCAVWeight/Youliang_llama3-8b-instruct-lora-derta-100step/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[{l}, {h}]_clfrs.pt",
	]
	labels = [
		"R2D2",
		"Llama3-CB",
		"Mistral-CB",
		"Llama3-TAR",
		"Llama3-LAT",
		"Llama3-RB",
		"Mistral-RB",
		"Mistral-SU",
		"Vicuna-SU",
		"Gemma-DA",
		"Llama2-DA",
		"Llama3-DeRTA"
	]
	lineName = [
		'[0.75, 0.75]',
		'[0.05, 0.8]',
		'[0.05, 0.4]',
		'[0.01, 0.6]',
		'[0.1, 0.6]',
	]

	ys = []
	xs = []
	for clfP in models:
		allClfr = classifier_manager.load_classifier_manager(clfP)
		allClfr = {k: allClfr[k] for k in sorted(allClfr)}
		y = []
		for iterNum, (score, clfr, negEachLayerProb, posEachLayerProb, usefulEachLayerProb) in list(allClfr.items()):
			y.append(np.array([float(c['Score']) for c in clfr.completion[0]]))
		ys.append(y)
		xs.append(np.arange(1, len(y) + 1, dtype=np.int32))
	plot_line_boxplot(xs, ys, labels, 'Iteration', 'SRF Score', f'./picture/data{l}_{h}.pdf', 2)

