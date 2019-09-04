set -eux

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_sync_nccl_allreduce=1
export CUDA_VISIBLE_DEVICES=0
export TASK_DATA_PATH=/content/fn
export MODEL_PATH=/content/ernie_model

python -u run_classifier.py \
                   --use_cuda true \
                   --verbose true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 24 \
                   --init_pretraining_params ${MODEL_PATH}/params \
                   --train_set ${TASK_DATA_PATH}/train.tsv \
                   --dev_set ${TASK_DATA_PATH}/dev.tsv \
                   --test_set ${TASK_DATA_PATH}/test.tsv \
                   --vocab_path ${MODEL_PATH}/vocab.txt \
                   --test_save /content/output/test_out.tsv \
                   --checkpoints /content/checkpoints \
                   --save_steps 100 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.0 \
                   --validation_steps 100 \
                   --epoch 10 \
                   --max_seq_len 256 \
                   --ernie_config_path ${MODEL_PATH}/ernie_config.json \
                   --learning_rate 5e-5 \
                   --skip_steps 10 \
                   --num_iteration_per_drop_scope 1 \
                   --num_labels 2 \
                   --random_seed 1 \
                   --predict_batch_size 24

