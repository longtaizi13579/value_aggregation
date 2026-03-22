import os
import json
import logging
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import deepspeed
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
from transformers import AutoTokenizer, HfArgumentParser

from arguments_va import ModelArguments, DataTrainingArguments, TrainingArguments
from models import Value_Aggregation_Gather, VAPPT_Gather, InforNCE_and_Eigenvalue
from loss_utils import mismatched_sizes_all_gather
use_auth_token = os.getenv("HUGGING_FACE_TOKEN")

# ========= 任务说明 =========
Instructions = {
    'allnli': 'Given a premise, retrieve a hypothesis that is entailed by the premise. Retrieve semantically similar text.',
    'dureader': 'Given a Chinese search query, retrieve web passages that answer the question.',
    'eli5_question_answer': 'Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum.',
    'fever': 'Given a claim, retrieve documents that support or refute the claim.',
    'hotpot_qa': 'Given a multi-hop question, retrieve documents that can help answer the question.',
    'miracl': 'Given a question, retrieve Wikipedia passages that answer the question.',
    'mrtydi': 'Given a question, retrieve Wikipedia passages that answer the question.',
    'msmarco_document': 'Given a web search query, retrieve relevant documents that answer the query.',
    'msmarco_passage': 'Given a web search query, retrieve relevant passages that answer the query.',
    'nq': 'Given a question, retrieve Wikipedia passages that answer the question.',
    'quora_duplicates': [
        "Given a question, retrieve questions that are semantically equivalent to the given question.",
        "Find questions that have the same meaning as the input question.",
    ],
    'squad': 'Retrieve Wikipedia passages that answer the question.',
    't2ranking': 'Given a Chinese search query, retrieve web passages that answer the question.',
    'trivia_qa': 'Retrieve Wikipedia passages that answer the question.'
}

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


def get_distributed_dataloader(dataset, batch_size: int, shuffle: bool = False):
    """
    dataset: HF Dataset 或任意 map-style Dataset
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=shuffle,
        drop_last=False,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def load_data_and_sampling(file_path: str):
    """
    读取 file_path 下所有 jsonl 文件，构造 query / positive / negative。
    """
    all_files = os.listdir(file_path)
    all_data = []
    idx = 0

    for every_file in tqdm(all_files, desc="Loading raw data"):
        print(every_file)
        now_file = os.path.join(file_path, every_file)
        base_name = every_file[:-6]  # 假设文件名类似 allnli.jsonl

        with open(now_file, "r", encoding="utf-8") as f:
            for line in f:
                idx += 1
                inst_cfg = Instructions[base_name]
                if isinstance(inst_cfg, str):
                    instruction = inst_cfg
                else:
                    # quora_duplicates: 两条描述交替使用
                    instruction = inst_cfg[idx % 2]

                line = line.strip()
                if not line:
                    continue
                a_dict = json.loads(line)

                a_dict["query"] =  a_dict["query"] 
                # 保留原 positive / negative
                a_dict["positive"] = a_dict["positive"]
                a_dict["negative"] = a_dict["negative"]
                all_data.append(a_dict)

    if len(all_data) == 0:
        raise ValueError(f"未在 {file_path} 读取到任何样本")

    # 最多采样 1024000 条，样本不够就全用
    num_samples = min(1024000, len(all_data))
    samples = random.sample(all_data, num_samples)

    query = [s["query"] for s in samples]
    positive = [s["positive"] for s in samples]
    negative = [s["negative"] for s in samples]

    return query, positive, negative

def load_eval_data(path):
    file_in = open(path, "r")
    all_sentences = file_in.readlines()
    random.shuffle(all_sentences)
    all_sentences = all_sentences[:12000]
    n = len(all_sentences) // 3  # 每份长度，自动舍弃不能整除的尾部

    part1 = all_sentences[:n]
    part2 = all_sentences[n:2*n]
    part3 = all_sentences[2*n:3*n]
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


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    标准 all_gather，当前脚本实际用的是 loss_utils.mismatched_sizes_all_gather，
    这个函数可以作为备用。
    """
    world_size = torch.distributed.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def normalize_embedding(x):
    """
    对嵌入做全局零均值+单位方差归一化（支持DDP分布式）
    参数：
    - x: (N, K) 张量，N=batch size，K=嵌入维度
    返回：
    - x_norm: 归一化后的嵌入张量
    """
    # 分布式训练：聚合所有GPU的均值/方差
    # if dist.is_available() and dist.is_initialized():
    #     # 计算单卡均值/方差
    #     mean = x.mean(dim=0, keepdim=True)
    #     var = x.var(dim=0, keepdim=True, unbiased=False)
        
    #     # all_reduce聚合（求全局均值/方差）
    #     dist.all_reduce(mean, op=dist.ReduceOp.AVG)
    #     dist.all_reduce(var, op=dist.ReduceOp.AVG)
        
    #     # 防止方差为0（避免除零错误）
    #     var = var.clamp(min=1e-6)
    # else:
    # 单卡训练
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    
    # 零均值 + 单位方差归一化
    x_norm = (x - mean) / torch.sqrt(var)
    return x_norm

def SIGReg(x, global_step, num_slices=256):
    """
    SIGReg实现（基于Epps-Pulley统计量）
    支持DDP分布式训练，无额外启发式超参数
    
    参数说明：
    - x: (N, K) 张量，N为batch size，K为嵌入维度
    - global_step: 全局训练步数，用于跨设备同步投影方向采样（单卡训练可省略）
    - num_slices: 投影方向数量（对应论文中的|M|），默认256
    返回：
    - T: SIGReg损失值（所有投影方向的Epps-Pulley统计量均值）
    """
    # 设备配置
    dev = dict(device=x.device)
    x = x.to(torch.float32)  # 输入张量转float32
    # 1. 生成同步的随机投影方向（跨GPU保持一致）
    g = torch.Generator(**dev)
    g.manual_seed(global_step)  # 确保不同GPU的投影方向相同
    proj_shape = (x.size(1), num_slices)  # (K, M)，M=num_slices
    A = torch.randn(proj_shape, generator=g, **dev)
    A /= A.norm(p=2, dim=0)  # 归一化投影方向为单位向量
    
    # 2. Epps-Pulley统计量配置
    # 积分点（t的取值范围，覆盖标准高斯的主要分布区域）
    t = torch.linspace(-5, 5, 17, **dev)  # 17个积分点，论文验证足够精确
    # 标准高斯分布的理论特征函数（目标特征函数）
    exp_f = torch.exp(-0.5 * t ** 2)
    
    # 3. 计算经验特征函数（ECF）
    # 嵌入投影：(N, K) @ (K, M) = (N, M) → 扩展为(N, M, T)，T为积分点数量
    x_t = (x @ A).unsqueeze(2) * t  # (N, M, 17)
    # 复数指数计算（特征函数核心）
    ecf = (1j * x_t).exp().mean(0)  # (M, 17)，按batch维度求平均
    
    # 4. 计算加权L2距离（Epps-Pulley损失核心）
    err = (ecf - exp_f).abs().square().mul(exp_f)  # 加权平方误差
    N = x.size(0) 
    # 梯形积分近似计算积分值（论文验证精度满足要求）
    T = torch.trapz(err, t, dim=1) * N  # 每个投影方向的损失
    
    # 5. 所有投影方向的损失均值
    return T.mean()

# def regularization_loss(full_query_embedding, full_positive_embedding, full_negative_embedding):
    # all_embeddings = torch.cat([full_query_embedding, full_positive_embedding, full_negative_embedding], dim=0) # 3*b, d
    # gram_matrix = all_embeddings.T @ all_embeddings # d, d， gram 矩阵
    # gram_matrix_abs_wo_diag = torch.abs(gram_matrix.clone())
    # gram_matrix_abs_wo_diag[torch.eye(gram_matrix.shape[0]).to(torch.bool)] = 0
    # gram_matrix = gram_matrix.diag() - torch.mean(gram_matrix_abs_wo_diag, dim=0)
    # loss = 1 - min(gram_matrix)




    




    # 修改版infoNCE 优化
    # 提升 gram matrix 的最小特征值
    # temperature_adjustment = torch.full_like(gram_matrix, temperature / gram_matrix.shape[0])
    # num_positive_pairs = gram_matrix.shape[0]
    # diag_temp = torch.eye(num_positive_pairs) * temperature
    # diag_temp = diag_temp.to(temperature_adjustment.device, dtype=temperature_adjustment.dtype)
    # diag_temp[~torch.eye(num_positive_pairs).to(torch.bool)] = temperature_adjustment[~torch.eye(num_positive_pairs).to(torch.bool)]
    # temperature_adjustment[:num_positive_pairs, :num_positive_pairs] = diag_temp

    # temp_apply = gram_matrix / temperature_adjustment #temperature_adjustment

    
    # probs = F.log_softmax(temp_apply, dim=1)

    # ground_truth = torch.arange(
    #     probs.shape[0], device=probs.device, dtype=torch.long
    # )

    # loss = F.nll_loss(probs, ground_truth)

    return loss

# dataset cover classification task, retrieval task, clustering task, reranking task, sts task
def evaluation(eval_dataloader, model_engine, step, output_dir="./gram_results"):
    os.makedirs(output_dir, exist_ok=True)

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

    all_embeddings = torch.cat([Q, P, N], dim=0)  # [3N, D]

    gram_matrix = all_embeddings.T @ all_embeddings  # [D, D]

    # 对称矩阵特征值，升序
    eigenvalues = torch.linalg.eigvalsh(gram_matrix)   # [D]

    max_eigenvalue = eigenvalues[-1].item()
    min_eigenvalue = eigenvalues[0].item()
    eigenvalue_variance = torch.var(eigenvalues, unbiased=False).item()
    sum_eigenvalue = torch.sum(eigenvalues).item()

    print(f"Gram matrix shape        : {gram_matrix.shape}")
    print(f"Max eigenvalue           : {max_eigenvalue:.6f}")
    print(f"Min eigenvalue           : {min_eigenvalue:.6f}")
    print(f"Sum of eigenvalues       : {sum_eigenvalue:.6f}")
    print(f"Eigenvalue variance      : {eigenvalue_variance:.6f}")

    # 1. 保存统计信息
    stats_path = os.path.join(output_dir, "gram_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Gram matrix shape: {tuple(gram_matrix.shape)}\n")
        f.write(f"Max eigenvalue: {max_eigenvalue:.10f}\n")
        f.write(f"Min eigenvalue: {min_eigenvalue:.10f}\n")
        f.write(f"Sum of eigenvalues: {sum_eigenvalue:.10f}\n")
        f.write(f"Eigenvalue variance: {eigenvalue_variance:.10f}\n")

    # 2. 保存全部特征值到文本文件
    eigenvalues_txt_path = os.path.join(output_dir, "gram_eigenvalues.txt")
    with open(eigenvalues_txt_path, "w", encoding="utf-8") as f:
        for i, val in enumerate(eigenvalues.tolist()):
            f.write(f"{i}\t{val:.10f}\n")

    # 3. 保存全部特征值到 pt 文件，便于之后直接加载
    eigenvalues_pt_path = os.path.join(output_dir, "gram_eigenvalues.pt")
    torch.save(eigenvalues, eigenvalues_pt_path)

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
    }


# ========= 解析命令行，初始化 tokenizer =========
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


# ========= 日志 =========
logger = logging.getLogger(name="my_logger")
os.makedirs("./wiki_train", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("./wiki_train", "wiki_va_scl.log"),
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)

# ========= 构建数据集 =========
query, positive, negative = load_data_and_sampling(data_args.path)
encode_ds = build_dataset(data_args, training_args, query, positive, negative)
# encode_ds.save_to_disk("va_scl_data")  # 如果不想落盘可以注释掉

eval_q, eval_p, eval_n = load_eval_data(data_args.eval_path)
evaluation_ds = build_dataset(data_args, training_args, eval_q, eval_p, eval_n)

eval_dataloader = DataLoader(
    evaluation_ds,
    batch_size=training_args.train_batch_size,
    shuffle=False
)

# 只保留真实存在的列
encode_ds.set_format(
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
# ========= 模型与 DeepSpeed =========
torch.cuda.set_device(training_args.local_rank)

model = InforNCE_and_Eigenvalue(training_args.local_rank)

model_engine, _,  _, _ = deepspeed.initialize(
    args=training_args,
    config_params=training_args.deepspeed,
    model=model,
    # model_parameters=model.parameters(),
)

# ========= DataLoader =========
# 根据 world_size 计算每卡 batch_size，而不是写死除以 8
world_size = torch.distributed.get_world_size()
per_device_batch_size = max(1, training_args.train_batch_size // world_size)


data_loader = get_distributed_dataloader(
    encode_ds,
    batch_size=per_device_batch_size,
    shuffle=False,  # 数据集已经 shuffle 过一次，如果想每轮再 shuffle 可以改成 True 并在每轮调用 sampler.set_epoch
)

# ========= 训练循环 =========
model_engine.train()
temperature = 0.05

for epoch in range(training_args.train_epoch):
    # 如果想让 DistributedSampler 每轮重排，可以在这里：
    # data_loader.sampler.set_epoch(epoch)

    for idx, batch in enumerate(
        tqdm(data_loader, desc=f"Epoch: {epoch + 1}", total=1000)
    ):
        batch = {k: v.cuda() for k, v in batch.items()}
        # InfoNCE loss
        # 这里假设 va_model.forward 接受和 batch 键同名的参数
        # 比如 def forward(self, query_input_ids, query_attention_mask, positive_input_ids, ...)
        query_embedding, positive_embedding, negative_embedding = model_engine(**batch)

        full_query_embedding = mismatched_sizes_all_gather(query_embedding)
        full_query_embedding = torch.cat(full_query_embedding, dim=0)

        full_positive_embedding = mismatched_sizes_all_gather(positive_embedding)
        full_positive_embedding = torch.cat(full_positive_embedding, dim=0)

        full_negative_embedding = mismatched_sizes_all_gather(negative_embedding)
        full_negative_embedding = torch.cat(full_negative_embedding, dim=0)

        full_weight_embedding = torch.cat(
            [full_positive_embedding, full_negative_embedding], dim=0
        )
        dot_products = full_query_embedding @ full_weight_embedding.T
        # temperature_adjustment = torch.full_like(dot_products, temperature / full_weight_embedding.shape[0])
        # num_positive_pairs = full_positive_embedding.shape[0]
        # diag_temp = torch.eye(num_positive_pairs) * temperature
        # diag_temp = diag_temp.to(temperature_adjustment.device, dtype=temperature_adjustment.dtype)
        # diag_temp[~torch.eye(num_positive_pairs).to(torch.bool)] = temperature_adjustment[:num_positive_pairs, :num_positive_pairs][~torch.eye(num_positive_pairs).to(torch.bool)]
        # temperature_adjustment[:num_positive_pairs, :num_positive_pairs] = diag_temp
        temp_apply = dot_products / temperature #temperature_adjustment
        probs = F.log_softmax(temp_apply, dim=1)

        ground_truth = torch.arange(
            probs.shape[0], device=probs.device, dtype=torch.long
        )

        loss = F.nll_loss(probs, ground_truth)


        # r_loss = regularization_loss(full_query_embedding, full_positive_embedding, full_negative_embedding)
        all_embeddings = torch.cat([full_query_embedding, full_positive_embedding, full_negative_embedding], dim=0)
        norm_embeddings = normalize_embedding(all_embeddings)
        r_loss = SIGReg(norm_embeddings, idx)
        final_loss = loss + r_loss

        model_engine.backward(final_loss)

        current_lr = model_engine.get_lr()[0]
        if training_args.local_rank == 0:
            logger.info(
                f"Epoch: {epoch + 1}, Batch: {idx + 1}, Loss: {loss.item()}, Regularization Loss: {r_loss.item()}, LR: {current_lr}"
            )

        if (idx + 1) % 200 == 0:
            model_engine.save_checkpoint(f"{model_args.save_dir}_epoch_{epoch}_step_{idx}")

        if idx % 30 == 0:
            model_engine.eval()
            result = evaluation(eval_dataloader, model_engine)
            model_engine.train()
        model_engine.step()

    # 每个 epoch 结束再存一份
    model_engine.save_checkpoint(f"{model_args.save_dir}_epoch_{epoch}")


