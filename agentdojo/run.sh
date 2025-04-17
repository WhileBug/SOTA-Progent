#!/bin/bash
log_dir="logs/no-update"

mkdir -p $log_dir

model="gpt-4o-2024-08-06"
# model="o3-mini-2025-01-31"
# model="claude-3-7-sonnet-20250219"

export SECAGENT_POLICY_MODEL="gpt-4o-2024-08-06"
# export SECAGENT_POLICY_MODEL="o3-mini-2025-01-31"
# export SECAGENT_POLICY_MODEL="claude-3-7-sonnet-20250219"
# export SECAGENT_POLICY_MODEL="o3-2025-04-03"
# export SECAGENT_POLICY_MODEL="gpt-4o-mini-2024-07-18"
# export SECAGENT_POLICY_MODEL="meta-llama/Llama-3.3-70B-Instruct" # vllm serve meta-llama/Llama-3.3-70B-Instruct -tp 2 --gpu_memory_utilization 0.98 --max_model_len 32768 --disable-log-requests
# export SECAGENT_POLICY_MODEL="Qwen/Qwen2.5-72B-Instruct" # vllm serve Qwen/Qwen2.5-72B-Instruct -tp 2 --gpu_memory_utilization 0.982 --max_model_len 15000 --disable-log-requests
# export SECAGENT_UPDATE="True"
# export SECAGENT_IGNORE_UPDATE_ERROR="True"

export COLUMNS=300

SECAGENT_SUITE=banking nohup python -m agentdojo.scripts.benchmark -s banking --model $model --logdir $log_dir > $log_dir/banking-no-attack.log 2>&1 &
SECAGENT_SUITE=slack nohup python -m agentdojo.scripts.benchmark -s slack --model $model --logdir $log_dir > $log_dir/slack-no-attack.log 2>&1 &
SECAGENT_SUITE=travel nohup python -m agentdojo.scripts.benchmark -s travel --model $model --logdir $log_dir > $log_dir/travel-no-attack.log 2>&1 &
SECAGENT_SUITE=workspace nohup python -m agentdojo.scripts.benchmark -s workspace --model $model --logdir $log_dir > $log_dir/workspace-no-attack.log 2>&1 &

SECAGENT_SUITE=banking nohup python -m agentdojo.scripts.benchmark -s banking --model $model --attack important_instructions --logdir $log_dir > $log_dir/banking-attack.log 2>&1 &
SECAGENT_SUITE=slack nohup python -m agentdojo.scripts.benchmark -s slack --model $model --attack important_instructions --logdir $log_dir > $log_dir/slack-attack.log 2>&1 &
SECAGENT_SUITE=travel nohup python -m agentdojo.scripts.benchmark -s travel --model $model --attack important_instructions --logdir $log_dir > $log_dir/travel-attack.log 2>&1 &
SECAGENT_SUITE=workspace nohup python -m agentdojo.scripts.benchmark -s workspace --model $model --attack important_instructions --logdir $log_dir > $log_dir/workspace-attack.log 2>&1 &

echo "started"
