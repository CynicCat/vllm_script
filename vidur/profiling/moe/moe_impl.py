import torch

from sarathi.model_executor.layers.fused_moe import FusedMoE
from sarathi.model_executor.layers.layernorm import RMSNorm
from sarathi.model_executor.parallel_utils.tensor_parallel.layers import (
    ReplicatedLinear,
    VocabParallelEmbedding,
)

from vidur.profiling.common.cuda_timer import CudaTimer
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.mlp.mlp_impl import CausalSelfAttention


class MoEFeedForward(torch.nn.Module):
    def __init__(self, config: ModelConfig, world_size: int):
        super().__init__()

        if (
            config.moe_num_experts is None
            or config.moe_top_k is None
            or config.moe_intermediate_dim is None
        ):
            raise ValueError(
                "MoE configuration parameters (num_experts, top_k, "
                "intermediate_dim) must be provided in ModelConfig."
            )

        assert config.embedding_dim % world_size == 0
        assert config.moe_intermediate_dim % world_size == 0

        self.hidden_size = config.embedding_dim
        self.world_size = world_size

        self.gate = ReplicatedLinear(
            self.hidden_size,
            config.moe_num_experts,
            bias=False,
            metric_name="moe_gate",
        )

        self.experts = FusedMoE(
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            hidden_size=self.hidden_size,
            intermediate_size=config.moe_intermediate_dim,
            reduce_results=False,
            renormalize=True,
            linear_metric_name="moe_linear",
            communication_metric_name="moe_all_reduce",
            world_size=world_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, router_logits)
        return hidden_states.view(original_shape)


class GPTMoEBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, world_size: int):
        super().__init__()

        if config.norm == "layer_norm":
            self.input_layernorm = torch.nn.LayerNorm(config.embedding_dim)
        elif config.norm == "rms_norm":
            self.input_layernorm = RMSNorm(config.embedding_dim)
        else:
            raise ValueError(f"Unknown norm: {config.norm} for input_layernorm")

        self._post_attn_norm = config.post_attn_norm
        if config.post_attn_norm:
            if config.norm == "rms_norm":
                self.post_attention_layernorm = RMSNorm(config.embedding_dim)
            else:
                raise ValueError(
                    f"Unknown norm: {config.norm} for post_attention_layernorm"
                )

        self.attn = CausalSelfAttention(config, world_size)
        self.moe = MoEFeedForward(config, world_size)

        self.input_layernorm_timer = CudaTimer("input_layernorm")
        self.post_attention_layernorm_timer = CudaTimer(
            "post_attention_layernorm"
        )
        self.add_timer = CudaTimer("add")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        if self._post_attn_norm:
            return self._forward_with_post_attn_norm(positions, hidden_states, residual)
        return self._forward_without_post_attn_norm(positions, hidden_states)

    def _forward_with_post_attn_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        with self.input_layernorm_timer:
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(positions=positions, hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        with self.post_attention_layernorm_timer:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe(hidden_states)
        with self.add_timer:
            hidden_states = residual + hidden_states
        return hidden_states

    def _forward_without_post_attn_norm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        with self.input_layernorm_timer:
            hidden_states = self.input_layernorm(hidden_states)
        attn_outputs = self.attn(positions=positions, hidden_states=hidden_states)
        feed_forward_hidden_states = self.moe(hidden_states)
        with self.add_timer:
            hidden_states = attn_outputs + feed_forward_hidden_states + residual
        return hidden_states


class MoEModel(torch.nn.Module):
    def __init__(self, config: ModelConfig, world_size: int, num_repeat_steps: int = 1):
        super().__init__()

        self.num_repeat_steps = num_repeat_steps
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.embedding_dim,
            linear_metric_name="emb",
            reduce_results=False,
            world_size=world_size,
            rank=0,
        )

        self.block = GPTMoEBlock(config, world_size=world_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = hidden_states
        for _ in range(self.num_repeat_steps):
            hidden_states = self.embed_tokens(input_ids)
            hidden_states = self.block(positions, hidden_states, residual)
        return hidden_states
