from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from vidur.config.base_fixed_config import BaseFixedConfig
from vidur.logger import init_logger
from vidur.types import ActivationType, NormType

logger = init_logger(__name__)


@dataclass
class BaseModelConfig(BaseFixedConfig):
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    embedding_dim: int
    mlp_hidden_dim: int
    max_position_embeddings: int
    use_gated_mlp: bool
    use_bias: bool
    use_qkv_bias: bool
    activation: ActivationType
    norm: NormType
    post_attn_norm: bool
    vocab_size: int
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False


@dataclass
class Llama2ModelConfig(BaseModelConfig):
    max_position_embeddings: int = 16384
    use_gated_mlp: bool = True
    use_bias: bool = False
    use_qkv_bias: bool = False
    activation: ActivationType = ActivationType.SILU
    norm: NormType = NormType.RMS_NORM
    post_attn_norm: bool = True
    vocab_size: int = 32768
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 10000
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-Config"


@dataclass
class CodeLlama34BModelConfig(Llama2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 22016
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "codellama/CodeLlama-34b-Instruct-hf"


@dataclass
class Llama2_7BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 11008
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-7b-hf"


@dataclass
class Llama2_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 4096

    @staticmethod
    def get_name():
        return "meta-llama/Llama-2-70b-hf"


@dataclass
class Llama3_8BModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int = 4096
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-8B"


@dataclass
class Llama3_70BModelConfig(Llama2ModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 8
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 28672
    max_position_embeddings: int = 8192
    rope_theta: Optional[float] = 500000
    vocab_size: int = 128256

    @staticmethod
    def get_name():
        return "meta-llama/Meta-Llama-3-70B"


@dataclass
class InternLMModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 4096
    vocab_size: int = 103168


@dataclass
class InternLM_20BModelConfig(InternLMModelConfig):
    num_layers: int = 60
    num_q_heads: int = 40
    num_kv_heads: int = 40
    embedding_dim: int = 5120
    mlp_hidden_dim: int = 13824

    @staticmethod
    def get_name():
        return "internlm/internlm-20b"


@dataclass
class InternLM2ModelConfig(Llama2ModelConfig):
    max_position_embeddings: int = 32768
    vocab_size: int = 92544


@dataclass
class InternLM2_20BModelConfig(InternLM2ModelConfig):
    num_layers: int = 48
    num_q_heads: int = 48
    num_kv_heads: int = 8
    embedding_dim: int = 6144
    mlp_hidden_dim: int = 16384
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "internlm/internlm2-20b"


@dataclass
class Phi2ModelConfig(Llama2ModelConfig):
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 32
    embedding_dim: int = 2560
    mlp_hidden_dim: int = 10240
    max_position_embeddings: int = 2048
    use_gated_mlp: bool = False
    use_bias: bool = True
    use_qkv_bias: bool = True
    activation: ActivationType = ActivationType.GELU
    norm: NormType = NormType.LAYER_NORM
    post_attn_norm: bool = False
    vocab_size: int = 51200
    rope_scaling: Optional[Dict[str, Any]] = None
    rope_theta: Optional[float] = 10000
    partial_rotary_factor: float = 0.4
    no_tensor_parallel: bool = True

    @staticmethod
    def get_name():
        return "microsoft/phi-2"


@dataclass
class QwenModelConfig(Llama2ModelConfig):
    use_qkv_bias: bool = True
    max_position_embeddings: int = 32768
    vocab_size: int = 152064

    @staticmethod
    def get_name():
        return "Qwen/Qwen-Config"


@dataclass
class Qwen72BModelConfig(QwenModelConfig):
    num_layers: int = 80
    num_q_heads: int = 64
    num_kv_heads: int = 64
    embedding_dim: int = 8192
    mlp_hidden_dim: int = 24576
    rope_theta: Optional[float] = 1000000

    @staticmethod
    def get_name():
        return "Qwen/Qwen-72B"

@dataclass
class Qwen30BModelConfig(QwenModelConfig):#2025.10.10添加
    num_layers: int = 48
    num_q_heads: int = 32
    num_kv_heads: int = 4
    embedding_dim: int = 2048
    mlp_hidden_dim: int = 6144
    rope_theta: Optional[float] = 1000000.0

    @staticmethod
    def get_name():
        return "Qwen/Qwen3-30B-A3B"

@dataclass
class Qwen235BModelConfig(QwenModelConfig):#2025.10.18添加
    num_layers: int = 94
    num_q_heads: int=64
    num_kv_heads: int=4
    embedding_dim: int=4096
    mlp_hidden_dim: int=12288
    max_position_embeddings: int=40960
    vocab_size: int=151936
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 1000000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False
    
    @staticmethod
    def get_name():
        return "Qwen/Qwen3-235B-A22B"

@dataclass
class Mixtral8x7BModelConfig(QwenModelConfig):#2025.10.18添加
    num_layers: int = 32
    num_q_heads: int = 32
    num_kv_heads: int = 8
    embedding_dim: int = 4096
    mlp_hidden_dim: int = 14336
    max_position_embeddings: int=32768
    vocab_size: int=32000
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 1000000.0
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    @staticmethod
    def get_name():
        return "mistralai/Mixtral-8x7B-v0.1"

@dataclass
class Llama17BModelConfig(QwenModelConfig):#2025.10.18添加
    num_layers: int = 48
    num_q_heads: int = 40
    num_kv_heads: int = 8
    embedding_dim: int = 5120
    mlp_hidden_dim: int = 16384
    max_position_embeddings: int=10485760
    vocab_size: int=202048
    is_neox_style: Optional[bool] = True
    rope_theta: Optional[float] = 500000.0
    partial_rotary_factor: float = 1.0
    no_tensor_parallel: bool = False

    @staticmethod
    def get_name():
        return "RedHatAI/Llama-4-Scout-17B-16E-Instruct"
