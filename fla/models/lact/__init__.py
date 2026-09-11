
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.lact.configuration_lact import LaCTConfig
from fla.models.lact.modeling_lact import LaCTForCausalLM, LaCTModel

AutoConfig.register(LaCTConfig.model_type, LaCTConfig, exist_ok=True)
AutoModel.register(LaCTConfig, LaCTModel, exist_ok=True)
AutoModelForCausalLM.register(LaCTConfig, LaCTForCausalLM, exist_ok=True)

__all__ = ['LaCTConfig', 'LaCTForCausalLM', 'LaCTModel']
