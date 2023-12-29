import sys    
import os    
import re
import time
import subprocess

global MAX_MEMORY

def run_pretrain():    
    # 从环境变量中获取参数值 
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    pretrain_script_path = os.environ.get('PYTHON_FILE')
    config_path = os.environ.get('CONFIG_PATH')
    config_name = os.environ.get('CONFIG_NAME')
    command = f"cuda_visible_devices={cuda_visible_devices} python {pretrain_script_path} "
    command += f"--config-path={config_path} "
    command += f"--config-name={config_name} "

    # trainer
    devices = os.environ.get('CURRENT_DEVICE')
    precision = os.environ.get('PRECISION')
    max_steps = os.environ.get('TRAINER_MAX_STEPS')

    command += f"trainer.devices={devices} "
    command += f"trainer.precision={precision} "
    command += f"trainer.max_steps={max_steps} "


    # exp_manager


    # model
    tokenizer_model = os.environ.get('MODEL_TOKENIZER_MODEL')
    tokenizer_vocab_file = os.environ.get('MODEL_TOKENIZER_VOCAB_FILE')
    tokenizer_merge_file = os.environ.get('MODEL_TOKENIZER_MERGE_FILE')
    data_data_prefix = os.environ.get('MODEL_DATA_DATA_PREFIX')
    # data_splits_string = os.environ.get('MODEL_DATA_SPLITS_STRING')
    encoder_seq_length = os.environ.get('ENCODER_SEQ_LENGTH')
    micro_batch_size = os.environ.get('MICRO_BATCH_SIZE')
    global_batch_size = os.environ.get('GLOBAL_BATCH_SIZE')
    num_layers = os.environ.get('NUM_LAYERS')
    use_flash_attention = os.environ.get('USE_FLASH_ATTENTION')
    sequence_parallel = os.environ.get('SEQUENCE_PARALLEL')
    enable_checkpointing = os.environ.get('ENABLE_CHECKPOING')
    tensor_model_parallel_size = os.environ.get('TENSOR_MODEL_PARALLEL_SIZE')
    pipeline_model_parallel_size = os.environ.get('PIPELINE_MODEL_PARALLEL_SIZE')
    optim_sched_warmup_steps = os.environ.get('OPTIM_SCHED_WARMUP_STEPS')
    command += f"model.tokenizer.model={tokenizer_model} "
    command += f"model.tokenizer.vocab_file={tokenizer_vocab_file} "
    command += f"model.tokenizer.merge_file={tokenizer_merge_file} "
    command += f"model.data.data_prefix={data_data_prefix} "
    # command += f"model.data.splits_string={data_splits_string} "
    command += f"model.encoder_seq_length={encoder_seq_length} "
    command += f"model.micro_batch_size={micro_batch_size} "
    command += f"model.global_batch_size={global_batch_size} "
    command += f"model.num_layers={num_layers} "
    command += f"model.use_flash_attention={use_flash_attention} "
    command += f"model.sequence_parallel={sequence_parallel} "
    command += f"trainer.enable_checkpointing={enable_checkpointing} "
    command += f"model.tensor_model_parallel_size={tensor_model_parallel_size} "
    command += f"model.pipeline_model_parallel_size={pipeline_model_parallel_size} "
    command += f"model.optim.sched.warmup_steps={optim_sched_warmup_steps} "

    # log
    log_dir = os.environ.get('LOG_DIR')
    log_file = os.environ.get('LOG_FILE')
    log_file = os.path.join(log_dir, log_file)

    command += f" > {log_file} 2>&1"

    print(f"running command: {command}")

    # 使用 subprocess.Popen 运行命令
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    global MAX_MEMORY
    MAX_MEMORY = 0
    # 每隔一秒检查一次进程是否结束
    while True:
        # poll() 方法会返回进程的退出状态，如果进程已经结束，返回非零值
        exit_code = process.poll()
        if exit_code is not None:
            # 进程已经结束，输出结果
            print("Command finished with exit code:", exit_code)
            _, _ = process.communicate()
            break
        else:
            # 进程还在运行，获取 nvidia-smi 的输出
            nvidia_smi_output = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # 解析 nvidia-smi 输出，更新 MAX_MEMORY
            lines = nvidia_smi_output.stdout.split('\n')
            # 跳过表头行
            for i in range(1, len(lines) - 1):
                # 检查当前行是否是指定 GPU 的行
                for gpu_id in cuda_visible_devices.split(','):
                    if f"|   {gpu_id}" in lines[i]:
                        # 检查下一行是否有显存使用信息
                        if i + 1 < len(lines) and re.search(r'\d+MiB / \d+MiB', lines[i + 1]):
                            # 提取显存使用量
                            match = re.search(r'\d+MiB / \d+MiB', lines[i + 1])
                            if match:
                                used_memory = int((match.group(0).split(' ')[0])[:-3])
                                # 更新 MAX_MEMORY
                                MAX_MEMORY = max(MAX_MEMORY, used_memory)
                                break  # 找到显存信息后跳出循环

            # 进程还在运行，等待一秒
            time.sleep(1)

    get_training_result(log_file)
    MAX_MEMORY = 0


def get_training_result(log_file : str):
    training_result_file = os.environ.get('TRAINING_RESULT')

    # trainer
    devices = os.environ.get('CURRENT_DEVICE')
    precision = os.environ.get('PRECISION')
    max_steps = os.environ.get('TRAINER_MAX_STEPS')

    # exp_manager

    # model
    encoder_seq_length = os.environ.get('ENCODER_SEQ_LENGTH')
    hidden_size = os.environ.get('HIDDEN_SIZE')
    micro_batch_size = os.environ.get('MICRO_BATCH_SIZE')
    global_batch_size = os.environ.get('GLOBAL_BATCH_SIZE')
    num_layers = os.environ.get('NUM_LAYERS')
    use_flash_attention = os.environ.get('USE_FLASH_ATTENTION')
    sequence_parallel = os.environ.get('SEQUENCE_PARALLEL')
    enable_checkpointing = os.environ.get('ENABLE_CHECKPOING')
    tensor_model_parallel_size = os.environ.get('TENSOR_MODEL_PARALLEL_SIZE')
    pipeline_model_parallel_size = os.environ.get('PIPELINE_MODEL_PARALLEL_SIZE')
    optim_sched_warmup_steps = os.environ.get('OPTIM_SCHED_WARMUP_STEPS')

    print(f"log_file: {log_file}")
    print("\n")

    with open(log_file, 'r') as f:
        # 从文件末尾开始读取
        match_flag = False
        content = f.readlines()[-1::-1] 
        for line in content:  
            # 使用正则表达式匹配 "out of memory" 字段
            if re.search(r'out of memory', line):
                match_flag = True
                with open(training_result_file, 'a') as result_file:
                    result_line = ' | '.join(str(param) for param in [devices, precision, max_steps, encoder_seq_length, hidden_size, micro_batch_size, global_batch_size, num_layers, use_flash_attention, sequence_parallel, enable_checkpointing, tensor_model_parallel_size, pipeline_model_parallel_size, optim_sched_warmup_steps, "out of memory", "0", "0", "0"])
                    result_file.write(f"{result_line}\n")
                return
            elif re.search(r'ValueError: ', line):
                match_flag = True
                match = re.search(r'ValueError:\s*(.*)', line)
                with open(training_result_file, 'a') as result_file:
                    result_line = ' | '.join(str(param) for param in [devices, precision, max_steps, encoder_seq_length, hidden_size, micro_batch_size, global_batch_size, num_layers, use_flash_attention, sequence_parallel, enable_checkpointing, tensor_model_parallel_size, pipeline_model_parallel_size, optim_sched_warmup_steps, match.group(1), "0", "0", "0"])
                    result_file.write(f"{result_line}\n")
                return
            elif re.search(r'Error: ', line):
                match_flag = True
                match = re.search(r'Error:\s*(.*)', line)
                with open(training_result_file, 'a') as result_file:
                    result_line = ' | '.join(str(param) for param in [devices, precision, max_steps, encoder_seq_length, hidden_size, micro_batch_size, global_batch_size, num_layers, use_flash_attention, sequence_parallel, enable_checkpointing, tensor_model_parallel_size, pipeline_model_parallel_size, optim_sched_warmup_steps, match.group(1), "0", "0", "0"])
                    result_file.write(f"{result_line}\n")
                return
            elif re.search(r'train_step_timing in s=', line):
                match_flag = True
                match = re.search(r'train_step_timing in s=(.*?)(,|$)', line)
                if match:
                    try:
                        tokens_per_second = float(int(global_batch_size) / int(devices) * int(encoder_seq_length) / float(match.group(1).strip()))
                        if precision == 32:
                            theoretical_throughput = os.environ.get('THEORETICAL_THROUGHPUT_32')
                        else:
                            theoretical_throughput = os.environ.get('THEORETICAL_THROUGHPUT_16')
                        mfu = float(tokens_per_second / 1024 * 6 * (12 * int(num_layers) * int(hidden_size) * int(hidden_size)) / 1e9) / float(float(theoretical_throughput) * int(devices))
                    except Exception as e:
                        tokens_per_second = str(e)

                    global MAX_MEMORY
                    memory_usage = f"{MAX_MEMORY} MB ({round(MAX_MEMORY / 1024.0, 2)} GB)"
                    with open(training_result_file, 'a') as result_file:
                        result_line = ' | '.join(str(param) for param in [devices, precision, max_steps, encoder_seq_length, hidden_size, micro_batch_size, global_batch_size, num_layers, use_flash_attention, sequence_parallel, enable_checkpointing, tensor_model_parallel_size, pipeline_model_parallel_size, optim_sched_warmup_steps, match.group(1).strip(), tokens_per_second, mfu, memory_usage])
                        result_file.write(f"{result_line}\n")
                    return
        if match_flag == False:
            with open(training_result_file, 'a') as result_file:
                result_line = ' | '.join(str(param) for param in [devices, precision, max_steps, encoder_seq_length, hidden_size, micro_batch_size, global_batch_size, num_layers, use_flash_attention, sequence_parallel, enable_checkpointing, tensor_model_parallel_size, pipeline_model_parallel_size, optim_sched_warmup_steps, "Failed", "0", "0", "0"])
                result_file.write(f"{result_line}\n")
            return


if __name__ == "__main__":
    run_pretrain()