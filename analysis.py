import os
import json
import logging
import random
from datetime import datetime
import gc
import numpy as np
import torch
import torch.nn.functional as F
import deepspeed
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, HfArgumentParser
from arguments_va import ModelArguments, DataTrainingArguments, TrainingArguments
from models import Value_Aggregation_Gather, VAPPT_Gather, InforNCE_and_Eigenvalue, Plain_Model_Evaluation
from torch.utils.data import DataLoader, DistributedSampler
use_auth_token = os.getenv("HUGGING_FACE_TOKEN")

# ========= 工具函数 =========
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 确保 CUDA 操作具有确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"随机种子已设置为 {seed}")


def load_eval_data(path):
    s1_name = f"{path}/s1.txt"
    s2_name = f"{path}/s2.txt"
    n_name = f"{path}/n.txt"
    s1_file_in = open(s1_name, "r")
    s2_file_in = open(s2_name, "r")
    n_file_in = open(n_name, "r")
    part1 = s1_file_in.readlines()
    part2 = s2_file_in.readlines()
    part3 = n_file_in.readlines()
    # all_sentences = file_in.readlines()
    # random.shuffle(all_sentences)
    # all_sentences = all_sentences[:12000]
    # n = len(all_sentences) // 3  # 每份长度，自动舍弃不能整除的尾部

    # part1 = all_sentences[:n]
    # part2 = all_sentences[n:2*n]
    # part3 = all_sentences[2*n:3*n]
    return part1, part2, part3


def load_eval_data(path):
    s1_name = f"{path}/s1.txt"
    s2_name = f"{path}/s2.txt"
    n_name = f"{path}/n.txt"
    s1_file_in = open(s1_name, "r")
    s2_file_in = open(s2_name, "r")
    n_file_in = open(n_name, "r")
    part1 = [x for x in s1_file_in.readlines()]
    part2 = [x for x in s2_file_in.readlines()]
    part3 = [x for x in n_file_in.readlines()]
    # [f"This sentence: “ {sentence} ” means in one word: “" for sentence in sentences_batch]
    # all_sentences = file_in.readlines()
    # random.shuffle(all_sentences)
    # all_sentences = all_sentences[:12000]
    # n = len(all_sentences) // 3  # 每份长度，自动舍弃不能整除的尾部

    # part1 = all_sentences[:n]
    # part2 = all_sentences[n:2*n]
    # part3 = all_sentences[2*n:3*n]
    return part1, part2, part3

def load_eval_data_promptEOL(path):
    s1_name = f"{path}/s1.txt"
    s2_name = f"{path}/s2.txt"
    n_name = f"{path}/n.txt"
    s1_file_in = open(s1_name, "r")
    s2_file_in = open(s2_name, "r")
    n_file_in = open(n_name, "r")
    part1 = [f"This sentence: “ {sentence} ” means in one word: “" for sentence in s1_file_in.readlines()]
    part2 = [f"This sentence: “ {sentence} ” means in one word: “" for sentence in s2_file_in.readlines()]
    part3 = [f"This sentence: “ {sentence} ” means in one word: “" for sentence in n_file_in.readlines()]
    # [f"This sentence: “ {sentence} ” means in one word: “" for sentence in sentences_batch]
    # all_sentences = file_in.readlines()
    # random.shuffle(all_sentences)
    # all_sentences = all_sentences[:12000]
    # n = len(all_sentences) // 3  # 每份长度，自动舍弃不能整除的尾部

    # part1 = all_sentences[:n]
    # part2 = all_sentences[n:2*n]
    # part3 = all_sentences[2*n:3*n]
    return part1, part2, part3

def build_dataset(data_args: DataTrainingArguments, training_args: TrainingArguments, query: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
    """
    直接在脚本内做 tokenization，返回 HF Dataset（已经 map 好）。
    使用全局 tokenizer。
    """
    set_seed(2025)

    dataset = Dataset.from_dict(
        {"query": query, "positive": positive, "negative": negative}
    )
    dataset = dataset.shuffle(seed=42)

    max_length = data_args.max_length

    def tokenize(examples):
        # 一个 batch 内统一处理
        def process(texts):
            encoded = tokenizer(
                texts,
                add_special_tokens=True,
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            return encoded["input_ids"], encoded["attention_mask"]

        query_input_ids, query_attention_mask = process(examples["query"])
        positive_input_ids, positive_attention_mask = process(examples["positive"])
        negative_input_ids, negative_attention_mask = process(examples["negative"])

        return {
            "query_input_ids": query_input_ids,
            "positive_input_ids": positive_input_ids,
            "negative_input_ids": negative_input_ids,
            "query_attention_mask": query_attention_mask,
            "positive_attention_mask": positive_attention_mask,
            "negative_attention_mask": negative_attention_mask,
        }

    encode_ds = dataset.map(
        tokenize,
        batched=True,
        batch_size=training_args.train_batch_size,
        num_proc=None,
    )

    return encode_ds
    
def evaluation(eval_dataloader, model_engine, step, output_dir="./gram_results"):
    os.makedirs(output_dir, exist_ok=True)

    curr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    all_query = []
    all_positive = []
    all_negative = []

    model_engine.eval()
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            dataset_batch = {
                k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            query_embedding, positive_embedding, negative_embedding = model_engine(**dataset_batch)

            all_query.append(query_embedding.detach().to(torch.float32).cpu())
            all_positive.append(positive_embedding.detach().to(torch.float32).cpu())
            all_negative.append(negative_embedding.detach().to(torch.float32).cpu())

    Q = torch.cat(all_query, dim=0)       # [N, D]
    P = torch.cat(all_positive, dim=0)    # [N, D]
    N = torch.cat(all_negative, dim=0)    # [N, D]

    # Q 和 P 的 alignment loss
    alignment_loss = (Q - P).norm(dim=1).pow(2).mean()


    # uniformity loss
    def lunif(x, t=2):
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    uniformity_loss = (lunif(Q) + lunif(P)) / 2

    print(f"[Loss] alignment loss        : {alignment_loss:.6f}")
    print(f"[Loss] uniformity loss       : {uniformity_loss:.6f}")

    # 保存 Loss 信息
    loss_stats_path = os.path.join(output_dir, "loss_stats.txt")
    with open(loss_stats_path, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Step={step} Time={curr_time}\n")
        f.write(f"Alignment loss: {alignment_loss:.10f}\n")
        f.write(f"Uniformity loss: {uniformity_loss:.10f}\n")

    all_embeddings = torch.cat([Q, P, N], dim=0)  # [3N, D]

    gram_matrix = all_embeddings.T @ all_embeddings  # [D, D]

    # 对称矩阵特征值，升序
    eigenvalues, eigenvectors = torch.linalg.eigh(gram_matrix)

    max_eigenvalue = eigenvalues[-1].item()
    min_eigenvalue = eigenvalues[0].item()
    eigenvalue_variance = torch.var(eigenvalues, unbiased=False).item()
    sum_eigenvalue = torch.sum(eigenvalues).item()

    print(f"[Gram] matrix shape        : {gram_matrix.shape}")
    print(f"[Gram] Max eigenvalue      : {max_eigenvalue:.6f}")
    print(f"[Gram] Min eigenvalue      : {min_eigenvalue:.6f}")
    print(f"[Gram] Sum of eigenvalues  : {sum_eigenvalue:.6f}")
    print(f"[Gram] Eigenvalue variance : {eigenvalue_variance:.6f}")

    # 样本协方差（行=样本，无偏估计 ddof=1）
    n_samples = all_embeddings.shape[0]
    denom = max(n_samples - 1, 1)
    centered = all_embeddings - all_embeddings.mean(dim=0, keepdim=True)
    cov_matrix = (centered.T @ centered) / denom  # [D, D]
    cov_eigenvalues, cov_eigenvectors = torch.linalg.eigh(cov_matrix)  # [D]，升序

    cov_max_eigenvalue = cov_eigenvalues[-1].item()
    cov_min_eigenvalue = cov_eigenvalues[0].item()
    cov_eigenvalue_variance = torch.var(cov_eigenvalues, unbiased=False).item()
    cov_sum_eigenvalue = torch.sum(cov_eigenvalues).item()

    print(f"[Cov]  matrix shape        : {cov_matrix.shape}")
    print(f"[Cov]  Max eigenvalue      : {cov_max_eigenvalue:.6f}")
    print(f"[Cov]  Min eigenvalue      : {cov_min_eigenvalue:.6f}")
    print(f"[Cov]  Sum of eigenvalues  : {cov_sum_eigenvalue:.6f}")
    print(f"[Cov]  Eigenvalue variance : {cov_eigenvalue_variance:.6f}")

    # 1. 保存统计信息
    stats_path = os.path.join(output_dir, "gram_stats.txt")
    with open(stats_path, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Step={step} Time={curr_time}\n")
        f.write(f"Gram matrix shape: {tuple(gram_matrix.shape)}\n")
        f.write(f"Max eigenvalue: {max_eigenvalue:.10f}\n")
        f.write(f"Min eigenvalue: {min_eigenvalue:.10f}\n")
        f.write(f"Sum of eigenvalues: {sum_eigenvalue:.10f}\n")
        f.write(f"Eigenvalue variance: {eigenvalue_variance:.10f}\n")

    # 2. 保存全部特征值到文本文件
    eigenvalues_txt_path = os.path.join(output_dir, "gram_eigenvalues.txt")
    with open(eigenvalues_txt_path, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Step={step} Time={curr_time}\n")
        for i, val in enumerate(eigenvalues.tolist()):
            f.write(f"{i}\t{val:.10f}\n")

    # 3. 保存全部特征值到 pt 文件，便于之后直接加载
    eigenvalues_pt_path = os.path.join(output_dir, "gram_eigenvalues.pt")
    torch.save(eigenvalues, eigenvalues_pt_path)

    # 4. 协方差：统计与特征值落盘（与 Gram 并行）
    cov_stats_path = os.path.join(output_dir, "cov_stats.txt")
    with open(cov_stats_path, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Step={step} Time={curr_time}\n")
        f.write(f"Covariance matrix shape: {tuple(cov_matrix.shape)}\n")
        f.write(f"Sample size n: {n_samples}, denominator (n-1): {denom}\n")
        f.write(f"Max eigenvalue: {cov_max_eigenvalue:.10f}\n")
        f.write(f"Min eigenvalue: {cov_min_eigenvalue:.10f}\n")
        f.write(f"Sum of eigenvalues: {cov_sum_eigenvalue:.10f}\n")
        f.write(f"Eigenvalue variance: {cov_eigenvalue_variance:.10f}\n")

    cov_eigenvalues_txt_path = os.path.join(output_dir, "cov_eigenvalues.txt")
    with open(cov_eigenvalues_txt_path, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Step={step} Time={curr_time}\n")
        for i, val in enumerate(cov_eigenvalues.tolist()):
            f.write(f"{i}\t{val:.10f}\n")

    cov_eigenvalues_pt_path = os.path.join(output_dir, "cov_eigenvalues.pt")
    torch.save(cov_eigenvalues, cov_eigenvalues_pt_path)

    return {
        "Q": Q,
        "P": P,
        "N": N,
        "all_embeddings": all_embeddings,
        "gram_matrix": gram_matrix,
        "eigenvalues": eigenvalues,
        "max_eigenvalue": max_eigenvalue,
        "min_eigenvalue": min_eigenvalue,
        "sum_eigenvalue": sum_eigenvalue,
        "eigenvalue_variance": eigenvalue_variance,
        "stats_path": stats_path,
        "eigenvalues_txt_path": eigenvalues_txt_path,
        "eigenvalues_pt_path": eigenvalues_pt_path,
        "cov_matrix": cov_matrix,
        "cov_eigenvalues": cov_eigenvalues,
        "cov_max_eigenvalue": cov_max_eigenvalue,
        "cov_min_eigenvalue": cov_min_eigenvalue,
        "cov_sum_eigenvalue": cov_sum_eigenvalue,
        "cov_eigenvalue_variance": cov_eigenvalue_variance,
        "cov_stats_path": cov_stats_path,
        "cov_eigenvalues_txt_path": cov_eigenvalues_txt_path,
        "cov_eigenvalues_pt_path": cov_eigenvalues_pt_path,
        "eigenvectors": eigenvectors,
        "cov_eigenvectors": cov_eigenvectors
    }


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name,
    use_auth_token=use_auth_token,
    add_eos_token=True
)

# 确保有 pad_token
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # 退一步，新增一个 pad_token
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

eval_q, eval_p, eval_n = load_eval_data(data_args.eval_path)
evaluation_ds = build_dataset(data_args, training_args, eval_q, eval_p, eval_n)

evaluation_ds.set_format(
    type="torch",
    columns=[
        "query_input_ids",
        "positive_input_ids",
        "negative_input_ids",
        "query_attention_mask",
        "positive_attention_mask",
        "negative_attention_mask",
    ],
)

eval_dataloader = DataLoader(
    evaluation_ds,
    batch_size=training_args.train_batch_size,
    shuffle=False
)

model = InforNCE_and_Eigenvalue(training_args.local_rank, model_args.model_name)
checkpoint = torch.load("/root/autodl-tmp/va_embedding_epoch_0_step_799__root_autodl-tmp_hf_cache_Qwen3-0.6B_bs_1024_lambda_0.0/global_step799/mp_rank_00_model_states.pt")
model.load_state_dict(checkpoint['module'], strict=True)
model.cuda()

# model = Plain_Model_Evaluation(training_args.local_rank, model_args.model_name)
# model.cuda()
eval_path = "./eval_res"
os.makedirs(eval_path, exist_ok=True)
exp_id = "infonce_training_test"
result1 = evaluation(
    eval_dataloader,
    model,
    0,
    output_dir=os.path.join(
        eval_path, exp_id
    ),
)

# del checkpoint
del model

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# eval_q, eval_p, eval_n = load_eval_data_promptEOL(data_args.eval_path)
# evaluation_ds = build_dataset(data_args, training_args, eval_q, eval_p, eval_n)

# evaluation_ds.set_format(
#     type="torch",
#     columns=[
#         "query_input_ids",
#         "positive_input_ids",
#         "negative_input_ids",
#         "query_attention_mask",
#         "positive_attention_mask",
#         "negative_attention_mask",
#     ],
# )

# eval_dataloader = DataLoader(
#     evaluation_ds,
#     batch_size=training_args.train_batch_size,
#     shuffle=False
# )

model = Plain_Model_Evaluation(training_args.local_rank, model_args.model_name)
model.cuda()
eval_path = "./eval_res"
os.makedirs(eval_path, exist_ok=True)
exp_id = "plain_model_test"
result2 = evaluation(
    eval_dataloader,
    model,
    0,
    output_dir=os.path.join(
        eval_path, exp_id
    ),
)
import pdb
pdb.set_trace()
logit = model.model.encoder.lm_head(result1["eigenvectors"][:1].cuda())
v,i = torch.topk(logit, k=100, dim=-1)
top100_tokens = [tokenizer.decode(x) for x in i[0]]

logit = model.model.encoder.lm_head(result2["eigenvectors"][:1].cuda())
v,i = torch.topk(logit, k=100, dim=-1)
plain_top100_tokens = [tokenizer.decode(x) for x in i[0]]
print("Evaluation completed.")

