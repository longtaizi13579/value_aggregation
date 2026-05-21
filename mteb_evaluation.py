from mteb import MTEB
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig, LlamaConfig, AutoModelForCausalLM
import torch
from torch import nn
from transformers import BertLMHeadModel, BertTokenizer, RobertaTokenizer, RobertaForMaskedLM, RobertaModel, BertModel
from tqdm import tqdm
import json
from transformers import LlamaTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from models import Value_Aggregation_Eval, InforNCE_and_Eigenvalue_Eval, InforNCE_and_Generative_Hops_Eval, F_PPVA_Eval
import numpy as np
import copy
from torch import nn
import torch.nn.functional as F
import os
#import deepspeed
from datasets import Dataset
from arguments_va_eval import ModelArguments, DataTrainingArguments, TrainingArguments
# from mteb.models.text_formatting_utils import corpus_to_texts
# from mteb.encoder_interface import PromptType
import torch
from transformers import HfArgumentParser
from transformers import AutoTokenizer, AutoModel
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Add the TTC font to the font manager
font_path = '/usr/share/fonts/wqy-microhei.ttc'
font_manager.fontManager.addfont(font_path)

# Set the font for matplotlib
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # Use the font name from the TTC file


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


task_to_prompt = {
    "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim:",
    "HotpotQA": "Given a multi-hop question, retrieve documents that can help answer the question:",
    "FEVER": "Given a claim, retrieve documents that support or refute the claim:",
    "MSMARCO": "Given a web search query, retrieve relevant passages that answer the query:",
    "DBPedia": "Given a query, retrieve relevant entity descriptions from DBPedia:",
    "NQ": "Given a question, retrieve Wikipedia passages that answer the question:",
    "QuoraRetrieval": "Given a question, retrieve questions that are semantically equivalent to the given question:",
    "SCIDOCS": "Given a scientific paper title, retrieve paper abstracts that are cited by the given paper:",
    "TRECCOVID": "Given a query on COVID-19, retrieve documents that answer the query:",
    "Touche2020": "Given a question, retrieve detailed and persuasive arguments that answer the question:",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim:",
    "NFCorpus": "Given a question, retrieve relevant documents that best answer the question:",
    "ArguAna": "Given a claim, find documents that refute the claim:",
    "CQADupstackTexRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackWebmastersRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackEnglishRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackGamingRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackGisRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackUnixRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackMathematicaRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackStatsRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackPhysicsRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackProgrammersRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackAndroidRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "CQADupstackWordpressRetrieval": "Given a question, retrieve detailed question descriptions from Stackexchange that are duplicates to the given question:",
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question:",
    "AskUbuntuDupQuestions": "Retrieve duplicate questions from AskUbuntu forum:",
    "StackOverflowDupQuestions": "Retrieve duplicate questions from StackOverflow forum:",
    "SciDocsRR": "Given a title of a scientific paper, retrieve the titles of other relevant papers:",
    "MindSmallReranking": "Retrieve relevant news articles based on user browsing history:",
    "Banking77Classification": "Given a online banking query, find the corresponding intents:",
    "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions: anger, fear, joy, love, sadness, and surprise:",
    "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral:",
    "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or notcounterfactual:",
    "ImdbClassification": "Classify the sentiment expressed in the given movie review text from the IMDB dataset:",
    "MassiveIntentClassification": "Given a user utterance as query, find the user intents:",
    "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios:",
    "MTOPDomainClassification": "Classify the intent domain of the given utterance in task-oriented conversation:",
    "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation:",
    "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic:",
    "AmazonPolarityClassification": "Classify Amazon reviews into positive or negative sentiment:",
    "AmazonReviewsClassification": "Classify the given Amazon review into its appropriate rating category:",
    "TwentyNewsgroupsClustering": "Identify the topic or theme of the given news articles:",
    "BiorxivClusteringS2S": "Identify the main category of Biorxiv papers based on the titles:",
    "MedrxivClusteringS2S": "Identify the main category of Medrxiv papers based on the titles:",
    "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts:",
    "ArxivClusteringS2S": "Identify the main and secondary category of Arxiv papers based on the titles:",
    "BiorxivClusteringP2P": "Identify the main category of Biorxiv papers based on the titles and abstracts:",
    "MedrxivClusteringP2P": "Identify the main category of Medrxiv papers based on the titles and abstracts:",
    "RedditClustering": "Identify the topic or theme of Reddit posts based on the titles:",
    "RedditClusteringP2P": "Identify the topic or theme of Reddit posts based on the titles and posts:",
    "StackExchangeClustering": "Identify the topic or theme of StackExchange posts based on the titles:",
    "StackExchangeClusteringP2P": "Identify the topic or theme of StackExchange posts based on the given paragraphs:",
    "TwitterSemEval2015": "Retrieve tweets that are semantically similar to the given tweet:",
    "TwitterURLCorpus": "Retrieve tweets that are semantically similar to the given tweet:",
    "SprintDuplicateQuestions": "Retrieve duplicate questions from Sprint forum:",
    "STS12": "Retrieve semantically similar text:",
    "STS13": "Retrieve semantically similar text:",
    "STS14": "Retrieve semantically similar text:",
    "STS15": "Retrieve semantically similar text:",
    "STS16": "Retrieve semantically similar text:",
    "STS17": "Retrieve semantically similar text:",
    "STS22": "Retrieve semantically similar text:",
    "BIOSSES": "Retrieve semantically similar text:",
    "SICK-R": "Retrieve semantically similar text:",
    "STSBenchmark": "Retrieve semantically similar text:",
    "SummEval": "Given a news summary, retrieve other semantically similar summaries:"
}
use_auth_token = os.getenv("HUGGING_FACE_TOKEN")
class MyModel():
    def __init__(self, model, tokenizer, data_args, each_task, model2=None) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.gpu_count = torch.cuda.device_count() 
        self.max_length = data_args.max_length
        self.batch_size = 16
        self.device = torch.device("cuda")
        self.model.to(self.device)
        self.model.eval()
        self.prompt = task_to_prompt[each_task]
        self.whole_length = 0
        self.global_count = 0
        self.lm_head = self.model.model.encoder.base_model.lm_head
        self.is_query = False
        # self.A_pinv = torch.linalg.pinv(self.lm_head.weight.T) 
        if model2 is not None:
            self.model = model2
            


    # def encode(self, sentences, **kwargs):
    #     # if 'prompt_type' in kwargs and kwargs['prompt_type']==PromptType.query:
    #     #     sentences = [self.prompt + '\n' + text for text in sentences]
    #     # if 'prompt_type' not in kwargs:
    #     #     sentences = [self.prompt + '\n' + text for text in sentences]
    #     if kwargs["prompt_name"] in ["ArguAna", "SciFact", "NFCorpus"]:
    #         if self.global_count == 0:
    #             sentences = [self.prompt + '\n' + text for text in sentences]
    #     else:
    #         sentences = [self.prompt + '\n' + text for text in sentences]
    #     self.global_count += 1
    #     with torch.no_grad():
    #         output_embedding = []
    #         for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches"):
    #             sentences_batch = sentences[start_index:start_index + self.batch_size]
    #             output_embedding.append(self.value_aggregation(sentences_batch))
    #             logits = self.lm_head(output_embedding[0][0].cuda())
    #             v, i = torch.topk(logits, k=10)
    #             tokens = [self.tokenizer.decode(index).replace(' ', '_') for index in i]

    #             # Another set of tokens and values corresponding to lambda = 0.02
    #             another_i = [6837, 19515, 17428, 104563, 73896, 2841, 56428, 41615, 1874, 5864]
    #             another_v = [26.8473, 26.5822, 23.8806, 23.5959, 22.6909, 22.3632, 21.5759, 20.7435, 20.7122, 20.6674]

    #             # Identify common tokens between the original tokens and the new tokens
    #             matching_indices = [index for index in i if index in another_i]
    #             matching_tokens = [self.tokenizer.decode(index).replace(' ', '_') for index in matching_indices]

    #             # Find the corresponding values for the matching tokens
    #             matching_v = [v[i].item() for i, token in enumerate(tokens) if token in matching_tokens]
    #             matching_another_v = [another_v[another_i.index(index)] for index in matching_indices]
    #             index = range(len(matching_tokens))
    #             bar_width = 0.35
    #             # Plotting the matching tokens with corresponding v and another_v
    #             plt.figure(figsize=(12, 6))

    #             # Plot the original logits (v) for matching tokens
    #             plt.bar([i - bar_width/2 for i in index], matching_v, width=bar_width, label='lambda=0', alpha=0.7)

    #             # Plot the new logits (another_v) for matching tokens
    #             plt.bar([i + bar_width/2 for i in index], matching_another_v, width=bar_width, label='lambda=0.02', alpha=0.7)

    #             # Labels and title
    #             plt.xlabel("Tokens", fontsize=12)
    #             plt.ylabel("Logits", fontsize=12)
    #             plt.title("Comparison of Logits for Matching Tokens", fontsize=14)
    #             plt.xticks(rotation=45, ha='right', fontsize=10)
    #             plt.xticks(index, matching_tokens, rotation=45, ha='right', fontsize=10)
    #             # Add legend
    #             plt.legend()

    #             # Adjust layout to make space for the sentence
    #             plt.subplots_adjust(bottom=0.3)  # Increase space at the bottom

    #             # Add the sentence at the bottom of the figure
    #             plt.figtext(0.5, 0.07, f"Original Sentence: {sentences_batch[0]}", wrap=True, ha="center", fontsize=14)

    #             # Save the plot
    #             plt.savefig(f"logits_matching_plot_{start_index}.png")
    #             plt.show()
                
    #         return torch.cat(output_embedding, dim=0)

    def encode(self, sentences, **kwargs):
        if kwargs["task_name"] in ["ArguAna", "SciFact", "NFCorpus"]:
            if self.global_count == 0:
                sentences = [self.prompt + '\n' + text for text in sentences]
                self.is_query = True
        else:
            sentences = [self.prompt + '\n' + text for text in sentences]
        self.global_count += 1
        with torch.no_grad():
            output_embedding = []
            for start_index in tqdm(range(0, len(sentences), self.batch_size), desc="Batches"):
                sentences_batch = sentences[start_index:start_index + self.batch_size]
                output_embedding.append(self.value_aggregation(sentences_batch).cpu())
            self.is_query = False
            return torch.cat(output_embedding, dim=0)
    
    
    def value_aggregation(self, sentences_batch):
        inputs = self.tokenizer(sentences_batch, add_special_tokens=True, padding=True, max_length=self.max_length, return_tensors='pt')
        inputs = {k:v.cuda() for k,v in inputs.items()}
        embeddings = self.model(inputs['input_ids'], inputs['attention_mask'])
        return embeddings.to(torch.float32)


    def n_steps_embedding(self, sentences_batch, is_query=False):
        sentences_batch[0] = "What is the nationality of the author of the"
        # sentences_batch[0] = "长沙的特色菜？"
        inputs = self.tokenizer(sentences_batch, add_special_tokens=True, padding=True, max_length=self.max_length, return_tensors='pt')
        inputs = {k:v.cuda() for k,v in inputs.items()}
        n_step_embeddings = self.model(inputs['input_ids'], inputs['attention_mask'])
        import pdb
        pdb.set_trace()
        logits = self.lm_head(n_step_embeddings)
        v, i = torch.topk(logits, k=100)
        tokens = [self.tokenizer.decode(index).replace(' ', '_') for index in i[0]]
        return n_step_embeddings

    # def value_aggregation(self, sentences_batch):
    #     inputs = self.tokenizer(sentences_batch, add_special_tokens=True, padding=True, max_length=self.max_length, return_tensors='pt')
    #     inputs = {k:v.cuda() for k,v in inputs.items()}
    #     embeddings = self.model(inputs['input_ids'], inputs['attention_mask'])
    #     logits = self.model.model.encoder.base_model.lm_head(embeddings)
    #     input_ids = inputs['input_ids']
    #     attention_mask = inputs['attention_mask']
    #     all_embeddings = []
    #     topk_vals, topk_ids = torch.topk(logits, k=10000, dim=-1)   # [B, 100], [B, 100]
    #     all_embeddings = (topk_vals.unsqueeze(-1) * self.A_pinv[topk_ids]).sum(dim=1)
    #     # for b in range(input_ids.size(0)):
    #     #     valid_input_ids = input_ids[b][attention_mask[b].bool()]   # 去掉 padding
    #     #     valid_input_ids = torch.unique(valid_input_ids)             # 去重，可选

    #     #     match_mask = torch.isin(topk_ids[b], valid_input_ids)      # [100]
    #     #     matched_token_ids = topk_ids[b][match_mask]                # [M]
    #     #     matched_logits = topk_vals[b][match_mask]                  # [M]
    #     #     embedding = matched_logits.unsqueeze(0) @ self.model.model.encoder.base_model.lm_head.weight[matched_token_ids]
    #     #     all_embeddings.append(embedding)  # [M, D]
    #     # all_embeddings = torch.cat(all_embeddings, dim=0)  # [B, D]
    #     # tokens = [self.tokenizer.decode(index).replace(' ', '_') for index in i]
    #     # dot_product = self.model.model.encoder.base_model.lm_head.weight[22798].unsqueeze(0) @ self.model.model.encoder.base_model.lm_head.weight.T
    #     # v,i = torch.topk(dot_product, k=100)
    #     # new_tokens = [self.tokenizer.decode(index).replace(' ', '_') for index in i[0]]
    #     # import pdb
    #     # pdb.set_trace()
    #     # print(tokens)
    #     # print(new_tokens)
    #     return all_embeddings.to(torch.float32)

    def qwen3_embedding_eval(self, sentences_batch):
        # Tokenize the input texts
        batch_dict = tokenizer(
            sentences_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict.to(self.model.device)
        outputs = self.model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        return embeddings

    
  
    

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_auth_token=use_auth_token,
    add_eos_token=True)
tokenizer.padding_side = "left"
if 'llama' in model_args.model_name.lower(): 
    tokenizer.pad_token = tokenizer.eos_token
model = F_PPVA_Eval(training_args.local_rank, model_args.model_name)
# model = InforNCE_and_Generative_Hops_Eval(training_args.local_rank, model_args.model_name) # InforNCE_and_Eigenvalue_Eval(training_args.local_rank, model_args.model_name)
checkpoint_path = model_args.checkpoint_path
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['module'], strict=True)
for each_task in ['Banking77Classification', 'EmotionClassification','ArguAna', 'SciFact', 'STS17', 'NFCorpus', 'SICK-R', 'STSBenchmark', 'MedrxivClusteringS2S', 'StackOverflowDupQuestions', 'TwentyNewsgroupsClustering', 'BiorxivClusteringS2S', 'SciDocsRR', 'SprintDuplicateQuestions']:#['Banking77Classification', 'EmotionClassification','ArguAna', 'SciFact', 'STS17', 'NFCorpus', 'SICK-R', 'STSBenchmark', 'MedrxivClusteringS2S', 'StackOverflowDupQuestions', 'TwentyNewsgroupsClustering', 'BiorxivClusteringS2S', 'SciDocsRR', 'SprintDuplicateQuestions']:
    mymodel = MyModel(model, tokenizer, data_args, each_task)
    evaluation = MTEB(tasks=[each_task])
    results = evaluation.run(mymodel, output_folder=f"5_18_qwen3_8b_fppva", eval_splits=["test"])
# model2 = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B')
# for each_task in ['SICK-R']:
#     mymodel = MyModel(model, tokenizer, data_args, each_task)
#     evaluation = MTEB(tasks=[each_task])
#     results = evaluation.run(mymodel, output_folder=f"3_28_qwen3_0.6b_step200_lambda_0_test", eval_splits=["test"])
