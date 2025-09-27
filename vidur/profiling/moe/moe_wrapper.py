import os

import sarathi.metrics.cuda_timer
import torch

from vidur.profiling.common.cuda_timer import CudaTimer

# monkey patching the CudaTimer class to use the sarathi implementation
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

from sarathi.model_executor.parallel_utils.parallel_state import (
    set_pipeline_model_parallel_rank,
    set_pipeline_model_parallel_world_size,
    set_tensor_model_parallel_rank,
    set_tensor_model_parallel_world_size,
)
from sarathi.model_executor.weight_utils import initialize_dummy_weights

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.timer_stats_store import TimerStatsStore
from vidur.profiling.moe.moe_impl import MoEModel
from vidur.profiling.utils import ProfileMethod
from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer

WARMUP_STEPS = 2
ACTIVE_STEPS = 20


class MoeWrapper:
    def __init__(
        self,
        model_config: ModelConfig,
        num_tensor_parallel_workers: int,
        profile_method: str,
        rank: int,
        output_dir: str,
    ):
        super().__init__()

        if (
            model_config.moe_num_experts is None
            or model_config.moe_top_k is None
            or model_config.moe_intermediate_dim is None
        ):
            raise ValueError(
                "Selected model does not provide MoE metadata. "
                "Ensure moe_num_experts, moe_top_k, and moe_intermediate_dim "
                "are defined in the configuration."
            )

        if model_config.moe_intermediate_dim % num_tensor_parallel_workers != 0:
            raise ValueError(
                "moe_intermediate_dim must be divisible by the tensor parallel "
                "world size."
            )

        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)

        self.model_config = model_config
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.profile_method = profile_method
        self.rank = rank
        self.output_dir = output_dir
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)

        # Configure Sarathi's tensor/pipeline parallel globals so fused MoE kernels
        # can infer sizes without initializing a distributed process group.
        set_tensor_model_parallel_world_size(self.num_tensor_parallel_workers)
        set_tensor_model_parallel_rank(0)
        set_pipeline_model_parallel_world_size(1)
        set_pipeline_model_parallel_rank(0)

        repeat_steps = (
            ACTIVE_STEPS if self.profile_method == ProfileMethod.RECORD_FUNCTION.value else 1
        )

        self.model = MoEModel(
            model_config,
            num_tensor_parallel_workers,
            repeat_steps,
        )
        initialize_dummy_weights(self.model)
        self.model = self.model.to(dtype=torch.float16).cuda().eval()

    @torch.inference_mode()
    def profile(self, num_tokens: int):
        vocab_range = self.model_config.vocab_size // self.num_tensor_parallel_workers
        input_ids = torch.randint(
            low=0,
            high=vocab_range,
            size=(num_tokens,),
            device="cuda",
            dtype=torch.long,
        )
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.long)

        if self.profile_method == ProfileMethod.RECORD_FUNCTION.value:
            self.model(
                input_ids,
                positions,
            )
            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            record_function_tracer = RecordFunctionTracer(self.output_dir)

            with record_function_tracer:
                self.model(
                    input_ids,
                    positions,
                )

            time_stats = record_function_tracer.get_operation_time_stats()
        else:
            for _ in range(WARMUP_STEPS):
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()

            self.timer_stats_store.clear_stats()

            for _ in range(ACTIVE_STEPS):
                self.model(
                    input_ids,
                    positions,
                )

            torch.cuda.synchronize()

            time_stats = self.timer_stats_store.get_stats()

        stats = {
            "time_stats": time_stats,
            "n_head": self.model_config.num_q_heads,
            "n_kv_head": self.model_config.num_kv_heads,
            "n_embd": self.model_config.embedding_dim,
            "n_expanded_embd": self.model_config.moe_intermediate_dim,
            "vocab_size": self.model_config.vocab_size,
            "num_experts": self.model_config.moe_num_experts,
            "top_k": self.model_config.moe_top_k,
            "num_tokens": num_tokens,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }
        self.timer_stats_store.clear_stats()

        return stats
