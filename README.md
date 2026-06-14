# Value Aggregation Training

This branch trains a sentence embedding model with an InfoNCE objective plus an
eigenvalue-style HamJEPA regularization term. The main entry point is
`run_va_gather.sh`, which launches `va_train.py` with DeepSpeed.

## Quick Start

Run training from this directory:

```bash
bash run_va_gather.sh
```

The default command is:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
deepspeed --num_gpus 2 va_train.py \
  --model_name Qwen/Qwen3-0.6B \
  --train_batch_size 128 \
  --path /root/autodl-tmp/dataset \
  --eval_path /root/autodl-tmp/process_code/mteb_selected_sentences.txt
```

## Main Files

- `run_va_gather.sh`: training launch script.
- `va_train.py`: data loading, tokenization, DeepSpeed setup, loss computation,
  periodic evaluation, and checkpoint saving.
- `models.py`: model wrappers. `InforNCE_and_Eigenvalue` loads the backbone from
  `--model_name`, applies LoRA, and returns normalized query/positive/negative
  embeddings.
- `arguments_va.py`: command-line argument defaults.
- `va_deepspeed_stage_new.json`: DeepSpeed config. It uses bf16, ZeRO stage 1,
  AdamW, warmup decay LR, and global train batch size 128.

## Data Format

`--path` should point to a directory containing `.jsonl` files. Each line must be
a JSON object with these fields:

```json
{"query": "...", "positive": "...", "negative": "..."}
```

The file name before `.jsonl` must match a key in the `Instructions` dictionary
inside `va_train.py`, for example `allnli.jsonl`, `nq.jsonl`, or
`msmarco_passage.jsonl`.

The script reads all `.jsonl` files, validates the required fields, and samples
up to `1,024,000` examples for training.

## Evaluation Data

`--eval_path` points to a plain text file. The script shuffles the lines, keeps
up to `12,000` lines, and splits them into three equal parts as eval
query/positive/negative text.

During training, evaluation runs every 30 steps. It saves Gram matrix
eigenvalue statistics under:

```text
gram_results/step_<global_step>/
```

Each evaluation directory contains:

- `gram_stats.txt`
- `gram_eigenvalues.txt`
- `gram_eigenvalues.pt`

## Training Logic

For each batch, the model encodes query, positive, and negative text separately.
The embeddings are L2-normalized before loss computation.

Across GPUs, embeddings are gathered with gradient support. The InfoNCE logits
are computed as:

```text
query_embeddings @ [positive_embeddings; negative_embeddings].T / temperature
```

The correct label for query `i` is positive `i`. Other positives and all
negatives serve as contrastive negatives. The current temperature is `0.05`.

The final loss is:

```text
final_loss = InfoNCE loss + HamJEPAReg loss
```

`HamJEPAReg` contains:

- a norm budget loss, keeping embedding norms near `1.0`;
- a projected log-det loss, encouraging non-collapsed covariance volume;
- a participation-ratio loss, reducing concentration in only a few directions.

## Important Arguments

- `--model_name`: Hugging Face model name or local checkpoint. Default:
  `Qwen/Qwen3-0.6B`.
- `--train_batch_size`: global batch size across all GPUs. Default: `128`.
- `--path`: training jsonl directory.
- `--eval_path`: evaluation text file.
- `--max_length`: tokenizer max length. Default: `512`.
- `--save_dir`: checkpoint prefix. Default: `./va_embedding`.
- `--deepspeed`: DeepSpeed config path. Default:
  `./va_deepspeed_stage_new.json`.
- `--train_epoch`: number of epochs. Default: `1`.

## Outputs

Checkpoints are saved by DeepSpeed:

```text
<save_dir>_epoch_<epoch>_step_<step>/
<save_dir>_epoch_<epoch>/
```

Training logs are written to:

```text
wiki_train/wiki_va_scl.log
```

## Notes

- The default model is `Qwen/Qwen3-0.6B`, but any compatible causal LM can be
  passed through `--model_name`.
- `InforNCE_and_Eigenvalue` uses last-token pooling and LoRA adapters.
- The tokenizer must have either `pad_token` or `eos_token`; otherwise training
  stops early with a clear error.
