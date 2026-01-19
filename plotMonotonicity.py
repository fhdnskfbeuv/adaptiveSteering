import json
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
from typing import List, Optional, Tuple

from SCAV import classifier_manager


def plot_line_with_error_bands_pdf(x_values, y_data, labels=None, title="Line Plot with Error Bands",
								   x_label="X-axis", y_label="Y-axis", output_path="output.pdf",
								   colors=None, markers=None, linestyles=None,
								   linewidth=4, alpha=0.3, grid=True, legend_loc='best'):
	"""
	Plot line chart with error bands and save as PDF

	Parameters:
	----------
	x_values : list
		X-axis values
	y_triplets : list of lists
		Y-axis triplets, each triplet format: [mean, lower_error, upper_error]
	labels : list, optional
		Labels for each line
	title : str, optional
		Plot title
	x_label : str, optional
		X-axis label
	y_label : str, optional
		Y-axis label
	output_path : str, optional
		Output PDF file path
	colors : list, optional
		Line colors
	markers : list, optional
		Marker styles
	linestyles : list, optional
		Line styles
	linewidth : int, optional
		Line width
	alpha : float, optional
		Error band transparency (0-1)
	grid : bool, optional
		Show grid
	legend_loc : str, optional
		Legend location

	Returns:
	----------
	fig : matplotlib.figure.Figure
		Figure object
	ax : matplotlib.axes.Axes
		Axes object
	"""

	n_lines = len(y_data)  # Number of lines

	colors = plt.cm.tab20(np.linspace(0, 1, n_lines))
	markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'] * (n_lines // 10 + 1)
	markers = markers[:n_lines]
	linestyles = ['-'] * n_lines

	# Create large figure (2048×2048 pixels, approx 27×27 cm, 300 DPI)
	fig_width = 2048 / 300  # Convert to inches (300 DPI)
	fig_height = 2048 / 300
	fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

	# Plot each line with its error band
	for i in range(n_lines):
		# Plot center line (mean line)
		ax.plot(x_values[i], y_data[i], color=colors[i], marker=markers[i],
				linestyle=linestyles[i], linewidth=linewidth, markersize=0,
				label=labels[i] if n_lines > 1 else None)

	# Set plot style with larger fonts
	ax.set_xlabel(x_label, fontsize=24, fontweight='bold')
	ax.set_ylabel(y_label, fontsize=24, fontweight='bold')
	ax.set_title(title, fontsize=28, fontweight='bold', pad=20)

	# Set tick label sizes
	ax.tick_params(axis='both', which='major', labelsize=20)
	ax.tick_params(axis='both', which='minor', labelsize=16)

	# Set grid
	if grid:
		ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)

	# Add legend
	if labels:
		handles, labels_legend = ax.get_legend_handles_labels()
		# If only one line with error band label, filter duplicates
		if n_lines == 1 and len(handles) > 1:
			# Keep only the line legend
			ax.legend(handles=[handles[1]], labels=[labels[0]],
					  loc=legend_loc, fontsize=15, framealpha=0.9)
		else:
			ax.legend(loc=legend_loc, fontsize=15, framealpha=0.9, ncol=2)

	# Auto-adjust y-axis range considering error bands
	# y_min_all = np.min(y_data[:, :, 0] - y_data[:, :, 1])
	# y_max_all = np.max(y_data[:, :, 0] + y_data[:, :, 2])
	# y_margin = (y_max_all - y_min_all) * 0.1
	# ax.set_ylim(y_min_all - y_margin, y_max_all + y_margin)

	# Use tight_layout to ensure all elements fit
	plt.tight_layout(pad=3.0)

	# Save as PDF
	with PdfPages(output_path) as pdf:
		pdf.savefig(fig, bbox_inches='tight', pad_inches=0.1, dpi=300)

	print(f"Plot saved as: {os.path.abspath(output_path)}")

	# Display plot
	plt.show()

	return fig, ax


def plot_multi_line_subplots_with_legend(
	x_data,
	y_data_list,
	titles=None,
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
	legend_ncol: int = 5,
	show_plot: bool = True,
	save_path: Optional[str] = None,
	save_legend_separately: bool = True,
	legend_save_path: Optional[str] = None,
	dpi: int = 300,  # Adjusted for 2048×2048 pixels
	legend_figsize: tuple = (2048 / 300, 3),  # Larger legend figure
	legend_fontsize: int = 24,  # Larger legend font
	title_fontsize: int = 24,  # Larger title font
	label_fontsize: int = 18,  # Larger axis label font
	tick_fontsize: int = 10,  # Larger tick font
	nCol=3,
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
		ceil(num_lines / nCol), nCol,
		figsize=(figsize[0] / 300, figsize[1] / 300),
		sharex=True,
		sharey=share_y,
		**kwargs
	)

	# Handle single subplot case
	if num_lines == 1:
		axes = [axes]
	else:
		axes = axes.flatten().tolist()

	# Calculate shared y-axis range if needed
	# if share_y:
	all_y_data = np.concatenate(y_data_list)
	y_min, y_max = np.nanmin(all_y_data), np.nanmax(all_y_data)
	y_padding = (y_max - y_min) * 0.1
	y_min, y_max = y_min - y_padding, y_max + y_padding

	# Plot each line in its own subplot
	for i in range(num_lines):
		ax = axes[i]
		for j, y_mean in enumerate(y_data_list[i]):
			# Plot center line (mean line)
			ax.plot(x_data[i], y_mean, color=colors[j], marker=markers[j],
					linestyle=linestyles[j], linewidth=linewidth, markersize=linewidth + 1,
					# label=line_names[i]
					)
		ax.set_title(titles[i], fontsize=10)

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
		# ax.set_ylim(y_min, y_max)
		# ax.set_yticks(np.arange(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10 + 0.01, 0.1))

		# ax.set_yticks(np.arange(np.floor(y_mean.min() * 10) / 10, np.ceil(y_mean.max() * 10) / 10 + 0.01, 0.1))
		if i % nCol == 0:
			ax.set_ylabel(y_label, fontsize=10, fontweight='bold')
		if i + nCol >= num_lines:
			ax.set_xlabel(x_label, fontsize=10, fontweight='bold')
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

	ys = [
		[np.array([0.453258333, 0.450020283, 0.402870218, 0.373518338][::-1]), np.array([0.2683, 0.346930042, 0.390469134, 0.440846503][::-1])],
		[np.array([0.817168037, 0.726433833, 0.562878102, 0.222240478][::-1]), np.array([0.177602999, 0.452212532, 0.522432715, 0.597944111][::-1])],
		[np.array([0.823587994, 0.823222856, 0.820378621, 0.680576007][::-1]), np.array([0.367357492, 0.615397116, 0.66749543, 0.72015504][::-1])],
		[np.array([0.352290913, 0.312772527, 0.283966089, 0.233678346][::-1]), np.array([0.125931842, 0.199443057, 0.20785629, 0.20172777][::-1])],
		[np.array([0.839252353, 0.782122711, 0.677791317, 0.540261626][::-1]), np.array([0.71197331, 0.781682988, 0.797039072, 0.795082072][::-1])],
		[np.array([0.849637409, 0.841609577, 0.733906786, 0.387357503][::-1]), np.array([0.75, 0.79, 0.81, 0.79][::-1])],
		[np.array([0.685766598, 0.764784157, 0.771435559, 0.628620267][::-1]), np.array([0.450044602, 0.668999155, 0.745656729, 0.774083333][::-1])],
		[np.array([0.599315683, 0.678010921, 0.628856242, 0.535449018][::-1]), np.array([0.47129935, 0.482610971, 0.43698894, 0.322617576][::-1])],
		[np.array([0.743239462, 0.723705173, 0.691041132, 0.628393749][::-1]), np.array([0.735694349, 0.733710806, 0.709149162, 0.569421629][::-1])],
		[np.array([0.766080737, 0.690019687, 0.597876648, 0.501749724][::-1]), np.array([0.556618283, 0.622525692, 0.644567768, 0.653738936][::-1])],
		[np.array([0.761375944, 0.745122234, 0.663163205, 0.531387726][::-1]), np.array([0.110039398, 0.362627253, 0.473618547, 0.542500407][::-1])],
		[np.array([0.76758, 0.78970, 0.54168, 0.24806][::-1]), np.array([0.56702, 0.63401, 0.68002, 0.67562][::-1])],
	]
	xs = [
		['Q1', 'Median', 'Q3', 'Max']
	] * 12
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
		'Best', 'First'
	]
	# for clfPs in models:
	# 	ys.append([])
	# 	for clfP in clfPs:
	# 		allClfr = classifier_manager.load_classifier_manager(clfP)
	# 		allClfr = {k: allClfr[k] for k in sorted(allClfr)}
	# 		y = []
	# 		for iterNum, (score, clfr, negEachLayerProb, posEachLayerProb, usefulEachLayerProb) in list(allClfr.items()):
	# 			y.append(score)
	# 		ys[-1].append(np.array(y))
	# 	xs.append(np.arange(1, len(y) + 1, dtype=np.int32))
	plot_multi_line_subplots_with_legend(xs, ys, titles=labels, line_names=lineName, nCol=3, linewidth=3, title='', x_label='Strength', y_label='Score', save_path='./picture/monotonicity.pdf')
