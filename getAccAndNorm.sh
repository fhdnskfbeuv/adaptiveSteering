CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'Unispac/Gemma-2-9B-IT-With-Deeper-Safety-Alignment' --tokenizer 'google/gemma-2-9b-it'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'LLM-LAT/robust-llama3-8b-instruct'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'GraySwanAI/Mistral-7B-Instruct-RR'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'GraySwanAI/Llama-3-8B-Instruct-RR'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'Unispac/Llama2-7B-Chat-Augmented' --tokenizer 'meta-llama/Llama-2-7b-chat-hf'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'thu-coai/vicuna-7b-v1.5-safeunlearning'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'thu-coai/Mistral-7B-Instruct-v0.2-safeunlearning'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model "lapisrocks/Llama-3-8B-Instruct-TAR-Refusal" --tokenizer "PawanKrd/Meta-Llama-3-8B-Instruct"
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'cais/zephyr_7b_r2d2'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'thkim0305/RepBend_Mistral_7B'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'thkim0305/RepBend_Llama3_8B'
CUDA_VISIBLE_DEVICES="3" python getAccAndNorm.py --model 'Youliang/llama3-8b-instruct-lora-derta-100step' --tokenizer "PawanKrd/Meta-Llama-3-8B-Instruct"
python plotAccAndNorm.py
