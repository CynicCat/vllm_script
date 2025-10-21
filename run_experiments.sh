
set -u

# Configs (如有需要可再修改)
MAMBA_ENV_PATH="/share/home/u25159/vidur-main/run_vidur_env"
PYTHON_MODULE="python -m vidur.main"
DEVICE="a800"
TP_SIZE=1
PIPELINE_STAGES=1
MAX_TOKENS=4096
SCHEDULER_TYPE="sarathi"
BATCH_CAP=128
CHUNK_SIZE=128
NO_STORE_PLOTS="--no-metrics_config_store_plots"
ENABLE_CHROME_TRACE="--metrics_config_enable_chrome_trace"

# Directory where simulator outputs are written
SIM_OUTPUT_BASE="/share/home/u25159/vidur-main/simulator_output"

# Models and tags
declare -a MODELS=(
  "Qwen/Qwen3-235B-A22B"
  "mistralai/Mixtral-8x7B-v0.1"
  "RedHatAI/Llama-4-Scout-17B-16E-Instruct"
  "Qwen/Qwen3-30B-A3B"
)
declare -A MODEL_TAG
MODEL_TAG["Qwen/Qwen3-235B-A22B"]="q235b"
MODEL_TAG["mistralai/Mixtral-8x7B-v0.1"]="m"
MODEL_TAG["RedHatAI/Llama-4-Scout-17B-16E-Instruct"]="l"
MODEL_TAG["Qwen/Qwen3-30B-A3B"]="q30b"

# Replicas
REPLICAS_LIST=(1 2 4 8)

# Trace files and tags
declare -a TRACES=(
  "./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv"
  "./data/processed_traces/bwb_stats_llama2_tokenizer_filtered_v2.csv"
  "./data/processed_traces/lmsys_chat_1m_conversation_stats_llama2_tokenizer.csv"
  "./data/processed_traces/workload.csv"
)
declare -A TRACE_TAG
TRACE_TAG["./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv"]="a"
TRACE_TAG["./data/processed_traces/bwb_stats_llama2_tokenizer_filtered_v2.csv"]="b"
TRACE_TAG["./data/processed_traces/lmsys_chat_1m_conversation_stats_llama2_tokenizer.csv"]="l"
TRACE_TAG["./data/processed_traces/workload.csv"]="d"

# QPS
QPS_LIST=(1 2 4 8 16)

# Output
OUTDIR="outputs"
mkdir -p "${OUTDIR}"
SUMMARY_CSV="${OUTDIR}/results_summary.csv"
if [ ! -f "${SUMMARY_CSV}" ]; then
  echo "tag,model,trace,replicas,qps,TBT_P99,TFTT_P90" > "${SUMMARY_CSV}"
fi

# Function to run one experiment and collect metrics
run_cmd_and_collect() {
  local model="$1"
  local model_tag="$2"
  local replicas="$3"
  local trace_file="$4"
  local trace_tag="$5"
  local qps="$6"

  local tag="${model_tag}-${trace_tag}-replicas${replicas}-qps${qps}"
  echo "================================================================"
  echo "RUNNING: ${tag}"
  echo "model: ${model}"
  echo "replicas: ${replicas}"
  echo "trace: ${trace_file}"
  echo "qps: ${qps}"
  echo "================================================================"

  # Build command as array to avoid quoting issues
  CMD=(mamba run -p "$MAMBA_ENV_PATH" $PYTHON_MODULE
       --replica_config_device "$DEVICE"
       --replica_config_model_name "$model"
       --cluster_config_num_replicas "$replicas"
       --replica_config_tensor_parallel_size "$TP_SIZE"
       --replica_config_num_pipeline_stages "$PIPELINE_STAGES")

  # request generator: workload.csv -> trace_replay, others -> synthetic
  if [[ "$(basename "$trace_file")" == "workload.csv" ]]; then
    CMD+=(--request_generator_config_type trace_replay
          --trace_request_generator_config_trace_file "$trace_file")
  else
    CMD+=(--request_generator_config_type synthetic
          --synthetic_request_generator_config_num_requests 512)
  fi

  # length generator (trace)
  CMD+=(--length_generator_config_type trace
        --trace_request_length_generator_config_trace_file "$trace_file"
        --trace_request_length_generator_config_max_tokens "$MAX_TOKENS")

  # interval, scheduler, misc
  CMD+=(--interval_generator_config_type poisson
        --poisson_request_interval_generator_config_qps "$qps"
        --replica_scheduler_config_type "$SCHEDULER_TYPE"
        --sarathi_scheduler_config_batch_size_cap "$BATCH_CAP"
        --sarathi_scheduler_config_chunk_size "$CHUNK_SIZE")

  # append flags if non-empty
  if [ -n "$NO_STORE_PLOTS" ]; then
    CMD+=("$NO_STORE_PLOTS")
  fi
  if [ -n "$ENABLE_CHROME_TRACE" ]; then
    CMD+=("$ENABLE_CHROME_TRACE")
  fi

  # Show command
  echo "Command preview:"
  printf ' %q' "${CMD[@]}"
  echo
  echo "Starting experiment (this may run a long time)..."

  # Execute and capture return code (do not exit script on rc != 0)
  "${CMD[@]}"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "WARNING: command exited with status ${rc} for tag ${tag}. Continuing..."
  fi

  # --- New CSV discovery logic: find latest simulator_output/* directory and use its request_metrics.csv ---
  csv_path=""
  if [ -d "${SIM_OUTPUT_BASE}" ]; then
    # get newest directory under SIM_OUTPUT_BASE (works with trailing slash)
    latest_sim_dir=$(ls -1dt "${SIM_OUTPUT_BASE}"/*/ 2>/dev/null | head -n1 || true)
    if [ -n "${latest_sim_dir}" ]; then
      # remove trailing slash if any
      latest_sim_dir="${latest_sim_dir%/}"
      cand="${latest_sim_dir}/request_metrics.csv"
      if [ -f "${cand}" ]; then
        csv_path="${cand}"
        echo "Found metrics at: ${csv_path} (from latest simulator_output directory ${latest_sim_dir})"
      else
        echo "No request_metrics.csv in latest simulator_output dir ${latest_sim_dir}"
      fi
    else
      echo "No subdirectories found in ${SIM_OUTPUT_BASE}"
    fi
  else
    echo "Simulator output base ${SIM_OUTPUT_BASE} does not exist"
  fi

  # Fallbacks (previous behavior): check current dir, metrics/ or any request_metrics*.csv
  if [ -z "${csv_path}" ]; then
    if [ -f "./request_metrics.csv" ]; then
      csv_path="./request_metrics.csv"
    elif [ -f "./metrics/request_metrics.csv" ]; then
      csv_path="./metrics/request_metrics.csv"
    else
      file_found=$(ls -1t request_metrics*.csv 2>/dev/null | head -n 1 || true)
      if [ -n "${file_found}" ]; then
        csv_path="${file_found}"
      fi
    fi
  fi

  if [ -z "${csv_path}" ]; then
    echo "ERROR: cannot find request_metrics.csv for tag ${tag}. Skipping metric calculation."
    return
  fi

  # move/copy csv to outputs with tag
  safe_csv="${OUTDIR}/request_metrics_${tag}.csv"
  mv -f "${csv_path}" "${safe_csv}" 2>/dev/null || cp -f "${csv_path}" "${safe_csv}"

  # Export env for python
  export TAG="${tag}"
  export MODEL_NAME="${model}"
  export TRACE_NAME="${trace_file}"
  export REPLICAS="${replicas}"
  export QPS="${qps}"
  export CSVPATH="${safe_csv}"
  export SUMMARY_CSV="${SUMMARY_CSV}"

  # Use a quoted heredoc so bash won't try to expand Python code contents
  python - <<'PY'
import os, sys, traceback
try:
    import pandas as pd
except Exception as e:
    print(f"{os.environ.get('TAG','')} ERROR: pandas not available: {e}")
    sys.exit(0)

tag = os.environ.get("TAG", "")
csv = os.environ.get("CSVPATH", "request_metrics.csv")
model = os.environ.get("MODEL_NAME","")
trace = os.environ.get("TRACE_NAME","")
replicas = os.environ.get("REPLICAS","")
qps = os.environ.get("QPS","")
summary = os.environ.get("SUMMARY_CSV", "outputs/results_summary.csv")

try:
    df = pd.read_csv(csv)
    TBT_P99 = df['request_e2e_time'].quantile(0.99)
    TFTT_P90 = df['prefill_e2e_time'].quantile(0.90)
    print(f"{tag} TBT(P99): {TBT_P99:.4f}")
    print(f"{tag} TFTT(P90): {TFTT_P90:.4f}")
    # append to summary
    with open(summary, "a") as fh:
        fh.write(f"{tag},{model},{trace},{replicas},{qps},{TBT_P99:.4f},{TFTT_P90:.4f}\n")
except Exception as e:
    print(f"{tag} ERROR processing {csv}: {e}")
    traceback.print_exc()
PY

  echo "Finished tag ${tag}. Raw CSV saved as ${safe_csv}."
  echo ""
}

# Main loops
for model in "${MODELS[@]}"; do
  model_tag="${MODEL_TAG[$model]}"
  for replicas in "${REPLICAS_LIST[@]}"; do
    for trace in "${TRACES[@]}"; do
      trace_tag="${TRACE_TAG[$trace]}"
      for qps in "${QPS_LIST[@]}"; do
        run_cmd_and_collect "${model}" "${model_tag}" "${replicas}" "${trace}" "${trace_tag}" "${qps}"
        sleep 1
      done
    done
  done
done

echo "All experiments finished. Summary file: ${SUMMARY_CSV}"