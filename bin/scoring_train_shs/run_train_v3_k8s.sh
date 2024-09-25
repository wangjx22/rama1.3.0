set -xe

source activate equiscore-efv2-2

commit_id=$(git rev-parse HEAD)

export PATH=/opt/conda/envs/gppi/bin:$PATH

wandb_key=f2e6fc06edcf1bacccb2b0f3814b38c3fd67503b
wandb_name=NB_Scoring_test_v3_test
PROJECT_DIR=$(cd "$(dirname $0)" && pwd)/../../
experiment_name=NB_Scoring_v3
yaml_path=${PROJECT_DIR}/yamls/scoring_train_yamls/train1.3_trial_test.yaml
device_ids=0
gpus=1
epochs=100000        # not used, controlled by max_epochs
max_epochs=1000
log_interval=100
eval_interval=2000
test_eval_interval=20000
save_interval=2000
base_lr=0
max_lr=0.0001
output_dir=/nfs_beijing_ai/ziqiao-2/tmp/rama-scoring-v1.3.0-dev-outputs/test/
mkdir -p ${output_dir}
#launch_mode=$1
launch_mode=k8s       # 设置成local应该就是本地跑，设置成k8s就是在k8s上分布跑


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
        --test_eval_interval $test_eval_interval \
        --base_lr $base_lr \
        --max_lr $max_lr \
        --version 2 #> ./logs/run_train_debug.logs &
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
        --test_eval_interval $test_eval_interval \
        --base_lr $base_lr \
        --max_lr $max_lr \
        --version 2
fi

# =====> END <======
