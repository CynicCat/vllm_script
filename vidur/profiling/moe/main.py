import argparse
import datetime
import itertools
import os
from typing import Any, List

import pandas as pd
import ray
import yaml
from tqdm import tqdm

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.moe.moe_wrapper import MoeWrapper
from vidur.profiling.utils import ProfileMethod, get_num_tokens_to_profile


def parse_args():
    parser = argparse.ArgumentParser(description="MoE Profiling")
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Disable Ray",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for profiling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiling_outputs",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "Qwen/Qwen3-30B-A3B",
        ],
        help="Models to profile",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[1],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to profile",
    )
    parser.add_argument(
        "--profile_method",
        default="record_function",
        choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations (default: %(default)s)",
    )
    args = parser.parse_args()

    args.output_dir = (
        f"{args.output_dir}/moe/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    return args
    


      
        

         


def profile_model(
    args: argparse.Namespace, model: str, num_tokens_to_profile: List[int], pbar: Any
):
    model_config = ModelConfig.from_model_name(model)

    if (
        model_config.moe_num_experts is None
        or model_config.moe_top_k is None
        or model_config.moe_intermediate_dim is None
    ):
        raise ValueError(
            f"Model {model} does not expose MoE configuration. Please provide "
            "a model with moe metadata."
        )

    use_ray = not args.disable_ray

    promises = []
    all_results: List[Any] = []

    if use_ray:
        model_wrapper_actor = ray.remote(
            num_cpus=1,
            num_gpus=1,
        )(
            MoeWrapper,
        ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

    for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
        if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:
            pbar.update(len(num_tokens_to_profile))
            continue

        if model_config.moe_intermediate_dim % num_tensor_parallel_workers != 0:
            pbar.update(len(num_tokens_to_profile))
            continue

        if not use_ray and num_tensor_parallel_workers > 1:
            pbar.update(len(num_tokens_to_profile))
            continue

        if use_ray:
            model_wrappers = [
                model_wrapper_actor.remote(
                    model_config,
                    num_tensor_parallel_workers,
                    args.profile_method,
                    rank,
                    args.output_dir,
                )
                for rank in range(args.num_gpus)
            ]
        else:
            model_wrappers = [
                MoeWrapper(
                    model_config,
                    num_tensor_parallel_workers,
                    args.profile_method,
                    rank=0,
                    output_dir=args.output_dir,
                )
            ]

        for num_tokens in num_tokens_to_profile:
            if use_ray:
                worker_id = len(promises)
                promise = model_wrappers[worker_id].profile.remote(
                    num_tokens,
                )
                promises.append(promise)

                if len(promises) >= args.num_gpus:
                    results = ray.get(promises)
                    all_results.extend(results)
                    promises = []
            else:
                result = model_wrappers[0].profile(num_tokens)
                all_results.append(result)

            pbar.update(1)

    if use_ray and promises:
        results = ray.get(promises)
        all_results.extend(results)

    df = pd.DataFrame(all_results)
    if df.empty:
        return df
    df = (
        pd.json_normalize(df["time_stats"])
        .add_prefix("time_stats.")
        .join(df.drop(columns=["time_stats"]))
    )

    return df


def main():
    args = parse_args()
    yaml.dump(vars(args), open(f"{args.output_dir}/config.yaml", "w"))

    num_tokens_to_profile = get_num_tokens_to_profile(args.max_tokens)

    total_combos = itertools.product(
        args.models,
        num_tokens_to_profile,
        args.num_tensor_parallel_workers,
    )

    pbar = tqdm(total=len(list(total_combos)))

    for model in args.models:
        result_df = profile_model(
            args,
            model,
            num_tokens_to_profile,
            pbar,
        )
        os.makedirs(f"{args.output_dir}/{model}", exist_ok=True)
        result_df.to_csv(f"{args.output_dir}/{model}/moe.csv", index=False)


if __name__ == "__main__":
    main()
