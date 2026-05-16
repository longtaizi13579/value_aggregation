# python ./download_model.py
deepspeed --num_gpus 8 FPPVA_train.py --train_batch_size 1024 --path /root/autodl-tmp/dataset --save_dir ./fppva_model