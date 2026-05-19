# Code for reviewers

## Installation

Run the command below to install conda environment first:
```commandline
conda create -n your_env_name python=3.11
conda activate your_env_name
pip install -r requirements.txt
cd ./strong_reject
pip install -e .
```

Then, copy and paste files in ```forTransformerLens``` to ```Your_Anaconda_Path/envs/as/lib/python3.11/site-packages/transformer_lens/```, 
and files in ```forTransformers``` to ```Your_Anaconda_Path/envs/as/lib/python3.11/site-packages/transformers/```. 
These replacements are crucial. The former is to make Angular, who depends on transformer_lens, run, and the latter is to fix bugs in ```apply_chat_templates```, which may miss bos_token.
(To be fair, only R2D2 will trigger this bug because it does not include bos_token in its chat template. Other LLMs have already handled everything in their chat templates.)

In ```*.sh```, you may find ```"model base_url api_key"```. This is for StrongReject's rubric judge, which is powered by commercial LLMs.
Replace ```"model base_url api_key"``` with LLMs available to you. 
For example, our results are based on Qwen-Plus (**In late January 2026, the main branch model of qwen-plus was updated. The SR metric in this paper are based on the previous version of qwen-plus, with the snapshot version being qwen-plus-2025-07-28.**): ```"qwen-plus-2025-07-28 https://dashscope.aliyuncs.com/compatible-mode/v1 sk-xxxxxxxxxxxxxx"```.
Here, do replace ```sk-xxxxxxxxxxxxxx``` with the API Key you apply in ```https://www.aliyun.com/```. Also, specify your HF_TOKEN: ```export HF_TOKEN=hf_xxxxxxx```.

Due to the anonymity constraints during the peer-review process (as anonymous file-sharing websites may be compromised or leak downloader information) and the size limitations for attachments, 
we are unable to provide the ready-made probes to save the reviewers' time in running Algorithm 1.

## Main Results

### Figure 2 and Figure 3

Run commands in ```getAccAndNorm.sh```, and check ```./picture/SCAV_per-layer_Accuracy.pdf``` and ```./picture/Per-layer_Norm.pdf```.


### Table 1

Run commands in ```iterSCAV.sh``` to acquire probes in ```./iterSCAVWeight```. 
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

### Figure 4

```commandline
python plotProgress4.py
```

### Figure 5

Run commands in ```benignEval.sh```, and compare ```benignRes.csv``` and ```myRes_min.csv```.

### Table 4

Run commands in ```boostR2D2.sh```, and check ```R2D2Res.csv```

### Figure 6

Run ```plotProgressR2D2.py```, and check ```./picture/iterR2D2.pdf```.

## Results in Appendix

### Table 5

Run commands in ```evalMy_threshold.sh```, and check ```myRes_threshold_min.csv```

### Figure 7

```commandline
python plotProgress7.py
```

### Figure 8, 9, and 10

```commandline
python plotDataProgress.py --l 1.0 --h 1.0
python plotDataProgress.py --l 0.5 --h 0.5
python plotDataProgress.py --l 0.0005 --h 0.0005
```



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

### Table 8

Table 8's results partially come from our ongoing work. We do not provide code. However, we can provide corresponding adversarial examples and probes for evaluation. This is sound, since, to prove the adversarial attack capability, the attacker only needs to show the found adversarial examples.

Build a new environment with commands below (remember to switch back before performing experiments above):

```commandline
conda create -n your_env_name python=3.11
conda activate your_env_name
pip install -r table8.txt
cd ./strong_reject
pip install -e .
```

Get ready your "model base_url api_key", and run
```commandline
CUDA_VISIBLE_DEVICES="1" python evalMyThink.py --answerOnly --evalData harm --maxL 16384 --model 'Qwen/Qwen3-4B-Thinking-2507' --evalPT "min" --maxL 16384 --csvP myRes.csv --evalClfr 'best' --layer -2 --posi 'all' --evalJudge "sjf" "hb" "model base_url api_key"  # Qwe3-4B-Think No Attack

CUDA_VISIBLE_DEVICES="1" python evalMyThink.py --answerOnly --evalData harm --maxL 16384 --model 'Qwen/Qwen3-4B-Thinking-2507' --evalPT "min" --maxL 16384 --clfP "./qwen3think.pt" --csvP myRes.csv --evalClfr 'best' --layer -2 --posi 'all' --evalJudge "sjf" "hb" "model base_url api_key"  # Qwe3-4B-Think Ours

CUDA_VISIBLE_DEVICES="2" python evalAdv.py --evalData harm --maxL 512 --model 'GraySwanAI/llava-v1.6-mistral-7b-hf-RR' --tokenizer 'llava-hf/llava-v1.6-mistral-7b-hf' --csvP advEvalRes.csv --evalJudge "sjf" "hb" "model base_url api_key"  # Llava-CB  No Attack

CUDA_VISIBLE_DEVICES="2" python evalAdv.py --evalData harm --maxL 512 --model 'GraySwanAI/llava-v1.6-mistral-7b-hf-RR' --tokenizer 'llava-hf/llava-v1.6-mistral-7b-hf' --imgP "./cbAdv.png" --csvP advEvalRes.csv --evalJudge "sjf" "hb" "model base_url api_key"  # Llava-CB PGD+Ours

CUDA_VISIBLE_DEVICES="2" python evalAdv.py --evalData harm --answerOnly --maxL 16384 --model 'zai-org/GLM-4.6V-Flash' --csvP advEvalRes.csv --evalJudge "sjf" "hb" "model base_url api_key"  # GLM No Attack

CUDA_VISIBLE_DEVICES="2" python evalAdv.py --evalData harm --answerOnly --maxL 16384 --model 'zai-org/GLM-4.6V-Flash' --imgP "./glmAdv.png" --csvP advEvalRes.csv --evalJudge "sjf" "hb" "model base_url api_key"  # GLM PGD+Ours
```

### Table 9

AdaSteer's implementation is full of strange magical hyper-parameters (even different from those presented in the published paper) and is not compatible. We re-implement it with hook for better compatibility and align the hyper-parameters with the AdaSteer paper (otherwise, the capability of AdaSteer is poor, leading to trivial robustness). To attack AdaSteer with our method, run

```commandline
CUDA_VISIBLE_DEVICES="4" python jailbreakGenSCAVIter.py --judge sjf --maxIter 20 --trainL 256 --layer -2 --model 'adasteer/meta-
llama/Llama-3.1-8B-Instruct' --saveDir ./iterSCAVWeight --softThres 0.05 0.6 --gpuLR --evalPT "min" --pt 0.5 --posi all --embType last --val

CUDA_VISIBLE_DEVICES="4" python jailbreakGenSCAVIter.py --judge sjf --maxIter 20 --trainL 256 --layer -2 --model 'adasteer/google/gemma-2-9b-it' --saveDir ./iterSCAVWeight --softThres 0.05 0.6 --gpuLR --evalPT "min" --pt 0.5 --posi all --embType last --val

CUDA_VISIBLE_DEVICES="4" python jailbreakGenSCAVIter.py --judge sjf --maxIter 20 --trainL 256 --layer -2 --model 'adasteer/Qwen/Qwen2.5-7B-Instruct' --saveDir ./iterSCAVWeight --softThres 0.05 0.6 --gpuLR --evalPT "min" --pt 0.5 --posi all --embType last --val

CUDA_VISIBLE_DEVICES="4" python evalMy.py --evalData harm --model 'adasteer/meta-llama/Llama-3.1-8B-Instruct' --evalPT "min" --clfP "./iterSCAVWeight/adasteer_meta-llama_Llama-3.1-8B-Instruct/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-32, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[0.05, 0.6]_clfrs.pt" --csvP myRes.csv --evalClfr 'best' --layer -2 --posi 'all' --evalJudge "sjf" "hb" "model base_url api_key"

CUDA_VISIBLE_DEVICES="4" python evalMy.py --evalData harm --model 'adasteer/google/gemma-2-9b-it' --evalPT "min" --clfP "./iterSCAVWeight/adasteer_google_gemma-2-9b-it/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-42, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[0.05, 0.6]_clfrs.pt" --csvP myRes.csv --evalClfr 'best' --layer -2 --posi 'all' --evalJudge "sjf" "hb" "model base_url api_key"

CUDA_VISIBLE_DEVICES="4" python evalMy.py --evalData harm --model 'adasteer/Qwen/Qwen2.5-7B-Instruct' --evalPT "min" --clfP "./iterSCAVWeight/adasteer_Qwen_Qwen2.5-7B-Instruct/harm[50, 50]_benign[50, 50]/judgesjf_embTypelast_posiall_filterDataFalse_layer[-28, -2]_penaltyl2_gpuLRTrue_reweightFalse_maxIter20_trainL256_pt0.5_softThres[0.05, 0.6]_clfrs.pt" --csvP myRes.csv --evalClfr 'best' --layer -2 --posi 'all' --evalJudge "sjf" "hb" "model base_url api_key"

```

To attack AdaSteer with baselines, run commands in ```adaSteerBaseline.sh```


