export HF_ENDPOINT="https://hf-mirror.com"
deepspeed --num_gpus 2 va_train.py --train_batch_size 256 --path /root/autodl-tmp/dataset