set -xe

source activate equiscore-md

commit_id=$(git rev-parse HEAD)

export PATH=/opt/conda/envs/gppi/bin:$PATH

wandb_key=XXXXXXXXXXXXXXXXXXXXXXXXXXX
wandb_name=NB_Scoring_test_v2_trial110
PROJECT_DIR=$(cd "$(dirname $0)" && pwd)/../..
experiment_name=NB_Scoring_2
yaml_path=${PROJECT_DIR}/yamls/scoring_train_yamls/train_trial110.yaml
device_ids=0
gpus=1
epochs=100000
max_epochs=1000
log_interval=100
eval_interval=2000
test_eval_interval=20000
save_interval=2000
start_decay_after_n_steps=50000
decay_every_n_steps=50000
warmup_no_steps=8000
base_lr=0
output_dir=${PROJECT_DIR}/train_output/trial110/
mkdir -p ${output_dir}
#launch_mode=$1
launch_mode=k8s


if [[ x$launch_mode == xlocal ]]; then
    CUDA_VISIBLE_DEVICES=${device_ids} deepspeed --include="localhost:${device_ids}" --master_port=12342 ${PROJECT_DIR}/run_train.py \
        --deepspeed-init-dist \
        --deepspeed_config ${PROJECT_DIR}/deepspeed_config.json \
        --yaml-path $yaml_path \
        --epochs $epochs \
        --log_interval $log_interval \
        --eval_interval $eval_interval \
        --save_interval $save_interval \
        --strict_eval \
        --save $output_dir\
        --commit-id $commit_id \
        --wandb \
        --wandb-key $wandb_key \
        --wandb-name $wandb_name \
        --experiment-name $experiment_name \
        --max_epochs $max_epochs \
        --start_decay_after_n_steps $start_decay_after_n_steps \
        --decay_every_n_steps $decay_every_n_steps \
        --test_eval_interval $test_eval_interval \
        --warmup_no_steps $warmup_no_steps \
        --base_lr $base_lr\
        --version 2
else
    gpu_count=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
    echo "detect visiable gpus: $gpu_count"
    torchrun --nproc_per_node=${gpu_count} ${PROJECT_DIR}/run_train.py \
        --deepspeed-init-dist \
        --deepspeed_config ${PROJECT_DIR}/deepspeed_config.json \
        --yaml-path $yaml_path \
        --epochs $epochs \
        --log_interval $log_interval \
        --eval_interval $eval_interval \
        --save_interval $save_interval \
        --strict_eval \
        --save $output_dir\
        --commit-id $commit_id \
        --wandb \
        --wandb-key $wandb_key \
        --wandb-name $wandb_name \
        --experiment-name $experiment_name \
        --max_epochs $max_epochs \
        --start_decay_after_n_steps $start_decay_after_n_steps \
        --decay_every_n_steps $decay_every_n_steps \
        --test_eval_interval $test_eval_interval \
        --warmup_no_steps $warmup_no_steps \
        --base_lr $base_lr\
        --version 2
fi

# =====> END <======
