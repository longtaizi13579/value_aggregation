export HF_ENDPOINT="https://hf-mirror.com"
deepspeed --num_gpus 2 va_train.py \
  --model_name Qwen/Qwen3-0.6B \
  --train_batch_size 128 \
  --path /root/autodl-tmp/dataset \
  --eval_path /root/autodl-tmp/process_code/mteb_selected_sentences.txt