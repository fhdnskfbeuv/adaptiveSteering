import json
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
from typing import List, Optional, Tuple
import matplotlib
import myUtil


def plot_multi_line_subplots_with_legend(
	x_data,
	y_data_list,
	line_names: Optional[List[str]] = None,
	colors: Optional[List[str]] = None,
	share_y: bool = False,
	figsize: tuple = (2048, 2048),  # 2048×2048 pixels at 100 DPI
	x_label: str = "X Axis",
	y_label: str = "Y",
	title: str = "Multi-line Plot",
	grid: bool = True,
	alpha: float = 0.3,
	linewidth: float = 3,  # Increased for larger figure
	hspace: float = 0.1,
	legend_loc: str = 'upper right',
	legend_ncol: int = 1,
	show_plot: bool = True,
	save_path: Optional[str] = None,
	save_legend_separately: bool = True,
	legend_save_path: Optional[str] = None,
	dpi: int = 300,  # Adjusted for 2048×2048 pixels
	legend_figsize: tuple = (20.48, 3),  # Larger legend figure
	legend_fontsize: int = 16,  # Larger legend font
	title_fontsize: int = 24,  # Larger title font
	label_fontsize: int = 14,  # Larger axis label font
	tick_fontsize: int = 10,  # Larger tick font
	nCols = 3,
	**kwargs
) -> Tuple[plt.Figure, List[plt.Axes]]:
	"""
	Plot multiple lines, each in its own subplot, with shared x-axis.
	Supports saving to PDF and exporting legend separately as PDF.

	Parameters:
	----------
	x_data : np.ndarray
		X-axis data shared by all lines
	y_data_list : List[np.ndarray]
		List of y-data arrays, each for one line
	line_names : Optional[List[str]], default=None
		Names for legend
	colors : Optional[List[str]], default=None
		Colors for each line
	share_y : bool, default=False
		Whether to share y-axis across subplots
	figsize : tuple, default=(20.48, 20.48)
		Figure size in inches (20.48×20.48 inches = 2048×2048 pixels at 100 DPI)
	x_label : str, default="X Axis"
		X-axis label
	y_label : str, default="Y"
		Y-axis label prefix
	title : str, default="Multi-line Plot"
		Figure title
	grid : bool, default=True
		Show grid
	alpha : float, default=0.3
		Grid transparency
	linewidth : float, default=3
		Line width (increased for larger figure)
	hspace : float, default=0.1
		Vertical spacing between subplots
	legend_loc : str, default='upper right'
		Legend location
	legend_ncol : int, default=1
		Number of columns in legend
	show_plot : bool, default=True
		Display the plot
	save_path : Optional[str], default=None
		Path to save main figure (PDF format)
	save_legend_separately : bool, default=False
		Save legend as separate PDF
	legend_save_path : Optional[str], default=None
		Path to save legend separately
	dpi : int, default=100
		DPI for saving (20.48 inches × 100 DPI = 2048 pixels)
	legend_figsize : tuple, default=(20.48, 3)
		Legend figure size in inches
	legend_fontsize : int, default=16
		Legend font size
	title_fontsize : int, default=24
		Title font size
	label_fontsize : int, default=18
		Axis label font size
	tick_fontsize : int, default=14
		Tick label font size
	**kwargs :
		Additional arguments for plt.subplots

	Returns:
	----------
	fig : matplotlib.figure.Figure
		Figure object
	axes : List[matplotlib.axes.Axes]
		List of axes objects
	"""

	# Parameter validation and initialization
	if not isinstance(y_data_list, list) or len(y_data_list) == 0:
		raise ValueError("y_data_list must be a non-empty list")

	num_lines = len(y_data_list)

	# Validate data lengths
	for i, y_data in enumerate(y_data_list):
		if len(x_data[i]) != len(y_data):
			raise ValueError(f"Length mismatch: y_data[{i}] has {len(y_data)} points, x_data has {len(x_data)}")

	# Set default line names
	if line_names is None:
		line_names = [f"Line {i + 1}" for i in range(num_lines)]
	elif len(line_names) != num_lines:
		raise ValueError(f"line_names length ({len(line_names)}) must match y_data_list length ({num_lines})")

	# Set default colors
	if colors is None:
		prop_cycle = plt.rcParams['axes.prop_cycle']
		default_colors = prop_cycle.by_key()['color']
		colors = [default_colors[i % len(default_colors)] for i in range(num_lines)]
	elif len(colors) != num_lines:
		raise ValueError(f"colors length ({len(colors)}) must match y_data_list length ({num_lines})")

	# Use matplotlib default color cycle
	colors = plt.cm.tab20(np.linspace(0, 1, num_lines))
	markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'] * (num_lines // 10 + 1)
	markers = markers[:num_lines]
	linestyles = ['-'] * num_lines

	# Create figure and subplots
	fig, axes = plt.subplots(
		ceil(num_lines / 3), 3,
		figsize=(figsize[0] / 300, figsize[1] / 300),
		sharex=True,
		sharey=share_y,
		**kwargs
	)
	totalSubFig = ceil(num_lines / 3) * 3
	# Handle single subplot case
	axes = axes.flatten().tolist()

	# Calculate shared y-axis range if needed
	# if share_y:
	all_y_data = np.concatenate(y_data_list)
	y_min, y_max = np.nanmin(all_y_data), np.nanmax(all_y_data)
	y_padding = (y_max - y_min) * 0.1
	y_min, y_max = y_min - y_padding, y_max + y_padding
	for i in range(num_lines, totalSubFig):
		axes[i].set_visible(False)
	# Plot each line in its own subplot
	for i in range(num_lines):
		ax = axes[i]
		y_mean = y_data_list[i][:, 0]  # Mean values
		y_lower_err = y_data_list[i][:, 1]  # Lower error
		y_upper_err = y_data_list[i][:, 2]  # Upper error

		# Calculate error band boundaries
		y_lower = y_lower_err
		y_upper = y_upper_err

		# Plot error band (shaded area)
		ax.fill_between(x_data[i], y_lower, y_upper,
						color=colors[i], alpha=alpha)

		# Plot center line (mean line)
		ax.plot(x_data[i], y_mean, color=colors[i], marker=markers[i],
				linestyle=linestyles[i], linewidth=linewidth, markersize=0,
				#label=line_names[i]
				)
		ax.plot(x_data[i], y_lower, color=colors[i], marker=markers[i],
				linestyle='--', linewidth=max(1, linewidth // 2), markersize=0)
		ax.plot(x_data[i], y_upper, color=colors[i], marker=markers[i],
				linestyle='--', linewidth=max(1, linewidth // 2), markersize=0)
		ax.set_title(line_names[i], fontsize=10)

		# Draw the line
		# ax.plot(
		# 	x_data,
		# 	y_data,
		# 	color=colors[i],
		# 	linewidth=linewidth,
		# 	label=line_names[i]
		# )

		# Set y-axis label with increased font size
		ax.set_ylabel("", fontsize=label_fontsize)

		# Set tick parameters with increased font size
		ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

		# Add grid
		if grid:
			ax.grid(True, alpha=alpha)

		# Add legend with increased font size
		# if line_names[i]:
		# 	ax.legend(loc=legend_loc, ncol=legend_ncol, fontsize=legend_fontsize)

		# Set unified y-axis range
		# if share_y:
		ax.set_ylim(y_min, y_max)
		if i % nCols == 0:
			ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
		if i + nCols >= num_lines:
			ax.set_xlabel(x_label, fontsize=10, fontweight='bold')

		# Add x-axis label to last subplot with increased font size
		# if i == num_lines - 1:
		# 	ax.set_xlabel(x_label, fontsize=label_fontsize)
	# Add unified y-axis label to the left of all subplots
	# fig.set(0.02, 0.5, y_label, va='center', rotation='vertical',
	# 		 fontsize=18, fontweight='bold')
	#
	# # Add unified x-axis label below the last subplot
	# fig.text(0.5, 0.02, x_label, ha='center',
	# 		 fontsize=18, fontweight='bold')
	# Adjust subplot spacing
	plt.subplots_adjust(hspace=hspace)

	# Add main title with increased font size
	plt.suptitle(title, fontsize=title_fontsize, y=0.98)

	# Auto-adjust layout
	plt.tight_layout()

	# Save main figure as PDF
	if save_path:
		if not save_path.endswith('.pdf'):
			save_path = save_path.rsplit('.', 1)[0] + '.pdf'
		plt.savefig(save_path, dpi=dpi, bbox_inches='tight', format='pdf')
		print(f"Main figure saved to: {save_path}")
		print(f"Figure size: {figsize[0]}×{figsize[1]} inches, DPI: {dpi}")
		print(f"Output resolution: {int(figsize[0] * dpi)}×{int(figsize[1] * dpi)} pixels")

	# Save legend separately as PDF
	if save_legend_separately:
		save_legend_as_pdf(
			line_names=line_names,
			colors=colors,
			save_path=legend_save_path or save_path.replace('.pdf', '_legend.pdf') if save_path else 'legend.pdf',
			ncol=legend_ncol,
			title=title,
			figsize=legend_figsize,
			fontsize=legend_fontsize,
			dpi=dpi
		)

	# Display the plot
	if show_plot:
		plt.show()

	return fig, axes


def save_legend_as_pdf(
	line_names: List[str],
	colors: List[str],
	save_path: str,
	ncol: int = 1,
	title: str = "Legend",
	figsize: tuple = (20.48, 3),  # Larger legend figure
	fontsize: int = 16,  # Larger font
	dpi: int = 100
) -> None:
	"""
	Save legend as a separate PDF file

	Parameters:
	----------
	line_names : List[str]
		Legend entry names
	colors : List[str]
		Colors for each entry
	save_path : str
		Save path
	ncol : int, default=1
		Number of columns in legend
	title : str, default="Legend"
		Legend title
	figsize : tuple, default=(20.48, 3)
		Legend figure size in inches
	fontsize : int, default=16
		Legend font size
	dpi : int, default=100
		DPI for saving
	"""

	# Create new figure for legend
	fig_legend = plt.figure(figsize=figsize)

	# Create dummy lines for legend
	lines = []
	for color in colors:
		line, = plt.plot([], [], color=color, linewidth=3)  # Thicker lines
		lines.append(line)

	# Create legend
	legend = plt.figlegend(
		lines,
		line_names,
		loc='center',
		ncol=ncol,
		title=title,
		frameon=True,
		fancybox=True,
		shadow=True,
		fontsize=fontsize,
		title_fontsize=fontsize + 2
	)

	# Remove axes
	plt.axis('off')

	# Adjust layout
	plt.tight_layout()

	# Save as PDF
	if not save_path.endswith('.pdf'):
		save_path = save_path.rsplit('.', 1)[0] + '.pdf'

	plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=dpi)
	print(f"Legend saved to: {save_path}")
	print(f"Legend size: {figsize[0]}×{figsize[1]} inches, DPI: {dpi}")
	print(f"Legend resolution: {int(figsize[0] * dpi)}×{int(figsize[1] * dpi)} pixels")

	# Close legend figure
	plt.close(fig_legend)

if __name__ == '__main__':
	matplotlib.rcParams['pdf.fonttype'] = 42
	os.makedirs('./picture', exist_ok=True)
	with open(r'./scavACC.json', 'r+') as f:
		data = json.load(f)
	xs, ys, labels = [], [], []
	title = 'SCAV per-layer Accuracy'
	for k, v in list(data.items())[-7:]:
		acc = np.array(v['acc'])  # (TrialNum, LayerNum)
		meanAcc = np.mean(acc, axis=0)
		upAcc = np.max(acc, axis=0)
		lowAcc = np.min(acc, axis=0)
		norm = np.array(v['norm'])  # (LayerNum, sampleNum)
		meanNorm = np.mean(norm, axis=1)
		upNorm = np.max(norm, axis=1)
		lowNorm = np.min(norm, axis=1)
		if title == 'SCAV per-layer Accuracy':
			ys.append(np.stack([meanAcc, lowAcc, upAcc], axis=1))
		else:
			ys.append(np.log10(np.stack([meanNorm, upNorm, lowNorm], axis=1)))
		labels.append(myUtil.nameMap[k])
		xs.append(np.arange(1, len(acc[0]) + 1, dtype=np.int32))

	plot_multi_line_subplots_with_legend(xs, ys, labels,
								 title='', x_label='Layer',
								 y_label='Accuracy' if title == 'SCAV per-layer Accuracy' else r'$L_{2}$-Norm (log scale)',
								 save_path=os.path.join('./picture', title.replace(' ', '_') + '.pdf'),
								   legend_loc='best', linewidth=2)

	xs, ys, labels = [], [], []
	title = 'Per-layer Norm'
	for k, v in list(data.items())[-7:]:
		acc = np.array(v['acc'])  # (TrialNum, LayerNum)
		meanAcc = np.mean(acc, axis=0)
		upAcc = np.max(acc, axis=0)
		lowAcc = np.min(acc, axis=0)
		norm = np.array(v['norm'])  # (LayerNum, sampleNum)
		meanNorm = np.mean(norm, axis=1)
		upNorm = np.max(norm, axis=1)
		lowNorm = np.min(norm, axis=1)
		if title == 'SCAV per-layer Accuracy':
			ys.append(np.stack([meanAcc, lowAcc, upAcc], axis=1))
		else:
			ys.append(np.log10(np.stack([meanNorm, upNorm, lowNorm], axis=1)))
		labels.append(myUtil.nameMap[k])
		xs.append(np.arange(1, len(acc[0]) + 1, dtype=np.int32))

	plot_multi_line_subplots_with_legend(xs, ys, labels,
										 title='', x_label='Layer',
										 y_label='Accuracy' if title == 'SCAV per-layer Accuracy' else r'$L_{2}$-Norm (log scale)',
										 save_path=os.path.join('./picture', title.replace(' ', '_') + '.pdf'),
										 legend_loc='best', linewidth=2)
