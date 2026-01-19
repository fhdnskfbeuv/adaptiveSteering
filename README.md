# Code for reviewers

## Installation

Run the command below to install conda environment first:
```commandline
conda env create -f environment.yml
```

Then, copy and paste files in ```forTransformerLens``` to ```Your_Anaconda_Path/envs/as/lib/python3.11/site-packages/transformer_lens/```, 
and files in ```forTransformers``` to ```Your_Anaconda_Path/envs/as/lib/python3.11/site-packages/transformers/```. 
These replacements are crucial. The former is to make Angular, who depends on transformer_lens, run, and the latter is to fix bugs in ```apply_chat_templates```, which may miss bos_token.
(To be fair, only R2D2 will trigger this bug because it does not include bos_token in its chat template. Other LLMs have already handled everything in their chat templates.)

In ```*.sh```, you may find ```"model base_url api_key"```. This is for StrongReject's rubric judge, which is powered by commercial LLMs.
Replace ```"model base_url api_key"``` with LLMs available to you. 
For example, our results are based on Qwen-Plus: ```"qwen-plus https://dashscope.aliyuncs.com/compatible-mode/v1 sk-xxxxxxxxxxxxxx"```.
Here, do replace ```sk-xxxxxxxxxxxxxx``` with the API Key you apply in ```https://www.aliyun.com/```.

If you do not have times to run Algorithm 1, you can download probes in [Coming soon] and ignore all command running ```jailbreakGenSCAVIter.py```.

## Main Results

### Figure 2 and Figure 3

Run commands in ```getAccAndNorm.sh```, and check ```./picture/SCAV_per-layer_Accuracy.pdf``` and ```./picture/Per-layer_Norm.pdf```.


### Table 1

Run commands in ```iterSCAV.sh``` to acquire probes. 
Then, run commands in ```evalMy.sh```. Results of our method will be stored in ```myRes_min.csv```.
To get baselines' results, run commands in ```otherBaseline.sh```, and check ```otherBaselineRes.csv```.


### Table 2

To reproduce results in Table 2, run commands in ```abla_*.sh```:
```commandline
SCAV+DLA+SAT: abla_last_all.sh
SCAV+AS: abla_ada.sh
SCAV+AS+DLA: abla_ada_last.sh
SCAV+AS+DLA+SAT: abla_ada_last_all.sh
```

### Table 3

To reproduce results of the first 5 rows in Table 3, run commands in ```otherBaseline_full.sh```, and check ```otherBaselineRes_full.csv```.
To reproduce results of SCAV+AS+DLA+SAT+NA, run commands in ```iterSCAV_full.sh``` to train probes, 
run commands in ```evalMy_full.sh```, and check ```myRes_full_min.csv```.

## Results in Appendix

### Figure 5

```commandline
python plotProgress.py
```

### Figure 6, 7, and 8

```commandline
python plotDataProgress.py --l 1.0 --h 1.0
python plotDataProgress.py --l 0.5 --h 0.5
python plotDataProgress.py --l 0.0005 --h 0.0005
```

### Figure 9

Run commands in ```benignEval.sh```, and compare ```benignRes.csv``` and ```myRes_min.csv```.


### Table 5

Run commands in ```boostR2D2.sh```, and check ```R2D2Res.csv```

### Figure 10

Run ```plotProgressR2D2.py```, and check ```./picture/iterR2D2.pdf```.

### Figure 11

Run commands in ```evalMyQ1.sh```, ```evalMyQ3.sh```, and ```evalMyMed.sh```, and check
```
Q1: myRes_q3.csv
Med: myRes_mid.csv
Q3: myRes_q1.csv
Max: myRes_min.csv
```

### Table 6

To reproduce the first 5 rows, run commands in ```otherBaseline_gneral.sh```, and check ```otherBaselineRes_general.csv```.
To reproduce ```Ours```, run commands in ```iterSCAV_general.sh```, and check ```myRes_general_min.csv```.

### Figure 12 and 13

Run commands in ```getAccAndNormGeneral.sh```, and check ```./picture/SCAV_per-layer_Accuracy.pdf``` and ```./picture/Per-layer_Norm.pdf```.

### Table 7

Run commands in ```iterSCAV_trick.sh```, run commands in ```evalMy_trick.sh```, and check
```
Row 2: myRes_med_min.csv
Row 3: myRes_response_min.csv
Row 4: myRes_med_response_min.csv
```

