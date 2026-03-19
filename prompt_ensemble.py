# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
from typing import Union, List

import torch
import torch.nn as nn
from pkg_resources import packaging

from AnomalyCLIP_lib.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

# 给类别套入模版并编码
def encode_text_with_prompt_ensemble(model, texts, device):
    # 定义一个包含正常状态提示词的列表，使用 {} 作为占位符
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    #定义一个包含异常状态提示词的列表，使用 {} 作为占位符。
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    # 将正常和异常状态提示词列表组合成一个列表。
    prompt_state = [prompt_normal, prompt_abnormal]
    # 定义一个包含各种模板句子的列表，使用{}    作为占位符。
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.',
                        'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.',
                        'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.',
                        'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.',
                        'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.',
                        'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.',
                        'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
                        'this is the {} in the scene.', 'this is one {} in the scene.']

    text_features = []
    # 初始化一个空列表，用于存储文本特征。

    # 遍历prompted_state和prompt_templates，将每个状态提示词填充到每个模板中，生成完整的句子并添加到prompted_sentence列表中。
    for i in range(len(prompt_state)):
        prompted_state = [state.format(texts[0]) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        # 将prompted_sentence列表中的句子进行分词，生成对应的token
        prompted_sentence = tokenize(prompted_sentence)
        # 使用模型对分词后的句子进行编码，生成文本特征，并将其移动到指定设备上。
        class_embeddings = model.encode_text(prompted_sentence.to(device))
        # 对生成的文本特征进行归一化处理
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        # 计算均值
        class_embedding = class_embeddings.mean(dim=0)
        # 对类特征进行归一化处理
        class_embedding /= class_embedding.norm()
        # 将处理后的类特征添加到text_features列表中。
        text_features.append(class_embedding)
    # 将text_features列表中的所有特征堆叠成一个张量，并移动到指定设备上，然后进行转置。
    text_features = torch.stack(text_features, dim=1).to(device).t()

    return text_features



def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])
class AnomalyCLIP_PromptLearner(nn.Module):

    def __init__(self, clip_model, design_details):
        super().__init__()
        # 定义类别名称
        classnames = ["object"]
        self.n_cls = len(classnames)
        self.n_ctx = design_details["Prompt_length"]
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()

        # 获取上下文向量的维度
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.classnames = classnames

        # 定义正常状态的模板
        self.state_normal_list = [
            "{}",
        ]

        # 定义异常状态的模板
        self.state_anomaly_list = [
            "damaged {}",
        ]

        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num

        if ctx_init_pos and ctx_init_neg:
            # 使用给定的词初始化上下文向量
            ctx_init_pos = ctx_init_pos.replace("_", " ")
            ctx_init_neg = ctx_init_neg.replace("_", " ")
            n_ctx_pos = len(ctx_init_pos.split(" "))
            n_ctx_neg = len(ctx_init_neg.split(" "))
            # 初始化文本成bpd编码
            prompt_pos = tokenize(ctx_init_pos)
            prompt_neg = tokenize(ctx_init_neg)
            with torch.no_grad():
                # 生成相应的文本嵌入
                embedding_pos = clip_model.token_embedding(prompt_pos).type(dtype)
                embedding_neg = clip_model.token_embedding(prompt_neg).type(dtype)
            # 去除EOS和CLS，获得可学习的文本提示
            ctx_vectors_pos = embedding_pos[0, 1: 1 + n_ctx_pos, :]
            ctx_vectors_neg = embedding_neg[0, 1: 1 + n_ctx_neg, :]
            prompt_prefix_pos = ctx_init_pos
            prompt_prefix_neg = ctx_init_neg
            if True:
                ctx_vectors_pos_ = []
                ctx_vectors_neg_ = []
                for _ in range(self.n_cls):
                    ctx_vectors_pos_.append(deepcopy(ctx_vectors_pos))
                    ctx_vectors_neg_.append(deepcopy(ctx_vectors_neg))
                ctx_vectors_pos = torch.stack(ctx_vectors_pos_, dim=0)
                ctx_vectors_neg = torch.stack(ctx_vectors_neg_, dim=0)

        else:
            # 随机初始化
            if True:
                print("Initializing class-specific contexts")
                # 这里是cls是类的个数，n_ctx_pos代表可学习token的长度，ctx_dim表示提示的维度
                ctx_vectors_pos = torch.empty(self.n_cls, self.normal_num, n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(self.n_cls, self.anormaly_num, n_ctx_neg, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
                ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors_pos, std=0.02)
            nn.init.normal_(ctx_vectors_neg, std=0.02)
            prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
            prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)
        self.compound_prompts_depth = design_details["learnabel_text_embedding_depth"]
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            print("single_para", single_para.shape)
            nn.init.normal_(single_para, std=0.02)

        single_layer = nn.Linear(ctx_dim, 896)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        # 定义可优化的上下文向量
        self.ctx_pos = nn.Parameter(ctx_vectors_pos)
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        # 生成正向和负向的提示
        prompts_pos = [prompt_prefix_pos + " " + template.format(name) + "." for template in self.state_normal_list for
                       name in classnames]
        prompts_neg = [prompt_prefix_neg + " " + template.format(name) + "." for template in self.state_anomaly_list for
                       name in classnames]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []

        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        # 生成相应的文本嵌入
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            # print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(normal_num, self.n_cls, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, self.n_cls, l, d).permute(1, 0, 2, 3)

        # 注册前缀和后缀缓冲区
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, self.n_cls, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, self.n_cls, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg
        # 注册tokenized_prompts缓冲区
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        # print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)

    def forward(self, cls_id=None):
        # 获取正负上下文向量
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        # 获取正负前缀和后缀向量
        prefix_pos = self.token_prefix_pos  # 正常样本的前缀token嵌入
        prefix_neg = self.token_prefix_neg  # 异常样本的前缀token嵌入
        suffix_pos = self.token_suffix_pos  # 正常样本的后缀token嵌入
        suffix_neg = self.token_suffix_neg  # 异常样本的后缀token嵌入

        # 构建完整的正样本提示向量
        prompts_pos = torch.cat(
            [
                prefix_pos,  # 前缀向量 (n_cls, 1, dim)
                ctx_pos,  # 可学习的上下文向量 (n_cls, n_ctx, dim)
                suffix_pos,  # 后缀向量 (n_cls, *, dim)
            ],
            dim=2,  # 在第3维度上拼接
        )

        # 构建完整的负样本提示向���
        prompts_neg = torch.cat(
            [
                prefix_neg,  # 前缀向量
                ctx_neg,  # 可学习的上下文向量
                suffix_neg,  # 后缀向量
            ],
            dim=2,
        )

        # 重塑正样本提示向量维度
        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)  # 展平为 (batch_size, length, dim)

        # 重塑负样本提示向量维度
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)  # 展平为 (batch_size, length, dim)

        # 合并正���样本提示向量
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)

        # 处理tokenized prompts的维度变换
        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, d)  # 展平正样本token

        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, d)  # 展平负样本token

        # 合并正负样本的tokenized prompts
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim=0)

        # 返回处理后的提示向量、tokenized提示和复合提示文本
        return prompts, tokenized_prompts, self.compound_prompts_text,prompts_pos,prompts_neg