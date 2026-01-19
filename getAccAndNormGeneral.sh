CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model "PawanKrd/Meta-Llama-3-8B-Instruct"
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'meta-llama/Llama-2-7b-chat-hf'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model "Qwen/Qwen3-4B-Instruct-2507"
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model "Qwen/Qwen2.5-14B-Instruct"
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model "lmsys/vicuna-7b-v1.5"
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model "mistralai/Mistral-7B-Instruct-v0.2"
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model "google/gemma-2-9b-it"
python plotAccAndNorm.py