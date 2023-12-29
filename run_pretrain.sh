#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# 设置你的配置路径和配置名
export PYTHON_FILE="/home/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py"
export CONFIG_PATH="/home/NeMo/examples/nlp/language_modeling/conf"
export CONFIG_NAME="megatron_gpt_config"

export LOG_DIR="/home/NeMo/log"
export TRAINING_RESULT="/home/NeMo/training_result.txt"
export THEORETICAL_THROUGHPUT_16=312
export THEORETICAL_THROUGHPUT_32=19.5

# trainer
# export TRAINER_PRECISION=("16 bf16 32")
export TRAINER_PRECISION=16
export TRAINER_DEVICES=("1 4")
export TRAINER_MAX_STEPS=50
export TRAINER_ENABLE_CHECKPOINTING=False

# exp_manager

# model
export MODEL_TOKENIZER_MODEL="tokenizer.model"
export MODEL_TOKENIZER_VOCAB_FILE="gpt2-vocab.json"
export MODEL_TOKENIZER_MERGE_FILE="gpt2-merges.txt"
export MODEL_DATA_DATA_PREFIX="[1.0,thepiledata_text_document]"
export MODEL_ENCODER_SEQ_LENGTH=512
export MODEL_NUM_LAYERS=32
export MODEL_HIDDEN_SIZE=768
export MODEL_MICRO_BATCH_SIZE=8
export MODEL_GLOBAL_BATCH_SIZE=32
export MODEL_TENSOR_MODEL_PARALLEL_SIZE=("1 2")
export MODEL_PIPELINE_MODEL_PARALLEL_SIZE=1
export MODEL_USE_FLASH_ATTENTION=False
export MODEL_SEQUENCE_PARALLEL=False
export MODEL_OPTIM_SCHED_WARMUP_STEPS=10


echo "GPUS | PRECISION | MAX_STEPS | ENCODER_SEQ_LENGTH | HIDDEN_SIZE | MICRO_BATCH_SIZE | GLOBAL_BATCH_SIZE | NUM_LAYERS | USE_FLASH_ATTENTION | SEQUENCE_PARALLEL | ENABLE_CHECKPOINTINT | TENSOR_MODEL_PARALLEL_SIZE | PIPELINE_MODEL_PARALLEL_SIZE | TRAIN_STEP_TIMING(s) | TOKENS_PER_SECOND | MFU | MEMORY_USAGE" > $TRAINING_RESULT

# 创建运行参数
for device in $TRAINER_DEVICES; do
    for precision in $TRAINER_PRECISION; do
        for encoder_seq_length in $MODEL_ENCODER_SEQ_LENGTH; do
            for hidden_size in $MODEL_HIDDEN_SIZE; do
                for micro_batch_size in $MODEL_MICRO_BATCH_SIZE; do
                    for global_batch_size in $MODEL_GLOBAL_BATCH_SIZE; do
                        for num_layers in $MODEL_NUM_LAYERS; do
                            for use_flash_attention in $MODEL_USE_FLASH_ATTENTION; do
                                for sequence_parallel in $MODEL_SEQUENCE_PARALLEL; do
                                    for enable_checkpointing in $TRAINER_ENABLE_CHECKPOINTING; do
                                        for tensor_model_parallel_size in $MODEL_TENSOR_MODEL_PARALLEL_SIZE; do
                                            for pipeline_model_parallel_size in $MODEL_PIPELINE_MODEL_PARALLEL_SIZE; do
                                                for optim_sched_warmup_steps in $MODEL_OPTIM_SCHED_WARMUP_STEPS; do

                                                    export CURRENT_DEVICE=$device
                                                    export PRECISION=$precision
                                                    export ENCODER_SEQ_LENGTH=$encoder_seq_length
                                                    export HIDDEN_SIZE=$hidden_size
                                                    export MICRO_BATCH_SIZE=$micro_batch_size
                                                    export GLOBAL_BATCH_SIZE=$global_batch_size
                                                    export NUM_LAYERS=$num_layers
                                                    export USE_FLASH_ATTENTION=$use_flash_attention
                                                    export SEQUENCE_PARALLEL=$sequence_parallel
                                                    export ENABLE_CHECKPOING=$enable_checkpointing
                                                    export TENSOR_MODEL_PARALLEL_SIZE=$tensor_model_parallel_size
                                                    export PIPELINE_MODEL_PARALLEL_SIZE=$pipeline_model_parallel_size
                                                    export OPTIM_SCHED_WARMUP_STEPS=$optim_sched_warmup_steps

                                                    # 创建日志文件名
                                                    if [[ "$precision" == "16" ]]; then
                                                        log_file="pretrain_${CONFIG_NAME}_gpus_${device}_fp16_${ENCODER_SEQ_LENGTH}_${HIDDEN_SIZE}_${MICRO_BATCH_SIZE}_${GLOBAL_BATCH_SIZE}_${NUM_LAYERS}_${USE_FLASH_ATTENTION}_${SEQUENCE_PARALLEL}_${ENABLE_CHECKPOING}_tp_${tensor_model_parallel_size}_pp_${pipeline_model_parallel_size}.log"
                                                    elif [[ "$precision" == "32" ]]; then
                                                        log_file="pretrain_${CONFIG_NAME}_gpus_${device}_fp32_${ENCODER_SEQ_LENGTH}_${HIDDEN_SIZE}_${MICRO_BATCH_SIZE}_${GLOBAL_BATCH_SIZE}_${NUM_LAYERS}_${USE_FLASH_ATTENTION}_${SEQUENCE_PARALLEL}_${ENABLE_CHECKPOING}_tp_${tensor_model_parallel_size}_pp_${pipeline_model_parallel_size}.log"
                                                    else
                                                        log_file="pretrain_${CONFIG_NAME}_gpus_${device}_${precision}_${ENCODER_SEQ_LENGTH}_${HIDDEN_SIZE}_${MICRO_BATCH_SIZE}_${GLOBAL_BATCH_SIZE}_${NUM_LAYERS}_${USE_FLASH_ATTENTION}_${SEQUENCE_PARALLEL}_${ENABLE_CHECKPOING}_tp_${tensor_model_parallel_size}_pp_${pipeline_model_parallel_size}.log"
                                                    fi
                                                    export LOG_FILE=$log_file

                                                    # 将命令输出重定向到日志文件和控制台 
                                                    python pretrain.py
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done