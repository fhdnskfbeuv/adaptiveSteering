CUDA_VISIBLE_DEVICES="5" python otherBaseline.py --evalData harm --method scav --strength 1e-4 --trainsize 0.1 --model 'adasteer/meta-llama/Llama-3.1-8B-Instruct' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"
CUDA_VISIBLE_DEVICES="5" python otherBaseline.py --evalData harm --method scav --strength 1e-4 --trainsize 0.1 --model 'adasteer/google/gemma-2-9b-it' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"
CUDA_VISIBLE_DEVICES="5" python otherBaseline.py --evalData harm --method scav --strength 1e-4 --trainsize 0.1 --model 'adasteer/Qwen/Qwen2.5-7B-Instruct' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"

CUDA_VISIBLE_DEVICES="7" python otherBaseline.py --evalData harm --method rep --strength 1.0 --trainsize 1.0 --model 'adasteer/meta-llama/Llama-3.1-8B-Instruct' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"
CUDA_VISIBLE_DEVICES="7" python otherBaseline.py --evalData harm --method rep --strength 1.0 --trainsize 1.0 --model 'adasteer/google/gemma-2-9b-it' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"
CUDA_VISIBLE_DEVICES="7" python otherBaseline.py --evalData harm --method rep --strength 1.0 --trainsize 1.0 --model 'adasteer/Qwen/Qwen2.5-7B-Instruct' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"

CUDA_VISIBLE_DEVICES="7" python otherBaseline.py --evalData harm --method rd --trainsize 0.8 --model 'adasteer/meta-llama/Llama-3.1-8B-Instruct' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"
CUDA_VISIBLE_DEVICES="7" python otherBaseline.py --evalData harm --method rd --trainsize 0.8 --model 'adasteer/google/gemma-2-9b-it' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"
CUDA_VISIBLE_DEVICES="7" python otherBaseline.py --evalData harm --method rd --trainsize 0.8 --model 'adasteer/Qwen/Qwen2.5-7B-Instruct' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"

CUDA_VISIBLE_DEVICES="5" python otherBaseline.py --evalData harm --method angular --trainsize 1.0 --model 'adasteer/meta-llama/Llama-3.1-8B-Instruct' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"
CUDA_VISIBLE_DEVICES="5" python otherBaseline.py --evalData harm --method angular --trainsize 1.0 --model 'adasteer/google/gemma-2-9b-it' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"
CUDA_VISIBLE_DEVICES="5" python otherBaseline.py --evalData harm --method angular --trainsize 1.0 --model 'adasteer/Qwen/Qwen2.5-7B-Instruct' --csvP ./otherBaselineRes.csv --evalJudge "sjf" "hb" "model base_url api_key"
