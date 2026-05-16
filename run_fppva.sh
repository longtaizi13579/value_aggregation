# python ./download_model.py
deepspeed --num_gpus 2 FPPVA_train.py --train_batch_size 512 --path /root/autodl-tmp/dataset