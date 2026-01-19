from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str, model=None, tokenizer=None) -> ModelBase:

    if 'qwen' in model_path.lower():
        from pipeline.model_utils.qwen_model import QwenModel
        return QwenModel(model_path, model=model, tokenizer=tokenizer)
    if 'llama' in model_path.lower() and '3' in model_path.lower():
        from pipeline.model_utils.llama3_model import Llama3Model
        return Llama3Model(model_path, model=model, tokenizer=tokenizer)
    elif 'llama' in model_path.lower():
        from pipeline.model_utils.llama2_model import Llama2Model
        return Llama2Model(model_path, model=model, tokenizer=tokenizer)
    elif 'gemma' in model_path.lower():
        from pipeline.model_utils.gemma_model import GemmaModel
        return GemmaModel(model_path, model=model, tokenizer=tokenizer)
    elif 'yi' in model_path.lower():
        from pipeline.model_utils.yi_model import YiModel
        return YiModel(model_path, model=model, tokenizer=tokenizer)
    elif 'mistral' in model_path.lower():
        from pipeline.model_utils.mistral_model import MistralModel
        return MistralModel(model_path, model=model, tokenizer=tokenizer)
    elif 'zephyr' in model_path.lower():
        from pipeline.model_utils.zephyr_model import ZephyrModel
        return ZephyrModel(model_path, model=model, tokenizer=tokenizer)
    elif 'vicuna' in model_path.lower():
        from pipeline.model_utils.vicuna_model import vicunaModel
        return vicunaModel(model_path, model=model, tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
