import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer




class KnowledgeDrivenPromptLearner(nn.Module):
    def __init__(self, clip_model, prompt_length=12, num_classes=10, margin=0.2):
        super().__init__()

        # CLIP模型组件
        self.clip = clip_model


        # 冻结CLIP参数
        for param in self.clip_text.parameters():
            param.requires_grad = False

        # 可学习的提示词嵌入
        self.normal_ctx = nn.Embedding(prompt_length, self.clip_text.config.hidden_size)
        self.abnormal_ctx = nn.Embedding(prompt_length, self.clip_text.config.hidden_size)

        # 类别相关token
        self.class_tokens = nn.Parameter(torch.randn(num_classes, self.clip_text.config.hidden_size))

        # 初始化函数
        nn.init.normal_(self.normal_ctx.weight, std=0.02)
        nn.init.normal_(self.abnormal_ctx.weight, std=0.02)
        nn.init.normal_(self.class_tokens, std=0.02)

        # 定义损失函数和超参数
        self.margin = margin
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def build_knowledge_base(self, class_name, images):
        # 从VQA中获取描述
        vqa_description = get_VQA_des(images,class_name)
        return vqa_description


    def encode_descriptions(self, descriptions):
        """将文本描述编码为特征向量"""
        inputs = self.tokenizer(
            descriptions,
            padding=True,
            return_tensors="pt",
            max_length=77,
            truncation=True
        ).to(self.class_tokens.device)

        return self.clip_text(**inputs).last_hidden_state[:, 0, :]  # 取[CLS] token

    def forward(self, class_ids):
        """
        Args:
            class_ids: [batch_size] 类别ID
        Returns:
            kd_loss: 知识驱动损失
        """
        batch_size = class_ids.size(0)

        # === 构建知识库 ===
        class_names = [f"class_{i.item()}" for i in class_ids]
        descriptions = self.build_knowledge_base(class_names, images=None)
        knowledge_features = self.encode_descriptions(descriptions)  # [2 * num_classes, d] 对大模型输出编码
        w_k = torch.mean(knowledge_features, dim=0)  # [d]

        # 将知识特征拆分为正常和异常特征
        normal_knowledge = knowledge_features[::2]  # [num_classes, d]
        abnormal_knowledge = knowledge_features[1::2]  # [num_classes, d]

        losses = []

        for i in range(batch_size):
            class_token = self.class_tokens[class_ids[i]]  # # 学生模型生成的类别表示

            # === 构建可学习提示 ===
            normal_ctx = self.normal_ctx.weight  # [prompt_length, d]
            abnormal_ctx = self.abnormal_ctx.weight  # [prompt_length, d]

            # 正常提示 = 正常上下文 + 类别token
            normal_prompt = torch.cat([
                normal_ctx,
                self.encode_descriptions([f"normal {class_names[i]}"])[0].unsqueeze(0),
                class_token.unsqueeze(0)
            ], dim=0)  # [prompt_length + 2, d]

            # 异常提示 = 异常上下文 + 类别token
            abnormal_prompt = torch.cat([
                abnormal_ctx,
                self.encode_descriptions([f"abnormal {class_names[i]}"])[0].unsqueeze(0),
                class_token.unsqueeze(0)
            ], dim=0)  # [prompt_length + 2, d]

            # === 计算提示向量的中心 ===
            w_n = torch.mean(normal_prompt, dim=0)  # [d]
            w_a = torch.mean(abnormal_prompt, dim=0)  # [d]

            # === 计算损失 ===
            normal_knowledge_i = normal_knowledge[class_ids[i]]
            abnormal_knowledge_i = abnormal_knowledge[class_ids[i]]

            # 1️⃣ 正常提示与正常知识的对齐（正样本）
            pos_sim = self.cosine_similarity(w_n, normal_knowledge_i)
            # 2️⃣ 异常提示与异常知识的对齐（正样本）
            pos_sim_abnormal = self.cosine_similarity(w_a, abnormal_knowledge_i)
            # 3️⃣ 正常提示与异常知识的分离（负样本）
            neg_sim = self.cosine_similarity(w_n, abnormal_knowledge_i)

            # 采用Kd-loss计算知识驱动损失
            loss = self.compute_kd_loss(w_k, w_n, w_a, margin=0.2)

            losses.append(loss)

        kd_loss = torch.mean(torch.stack(losses))
        return kd_loss

    def compute_kd_loss(self, w_k, w_n, w_a, margin=0.2):
        # L2 归一化
        w_k_norm = w_k / w_k.norm(p=2)
        w_n_norm = w_n / w_n.norm(p=2)
        w_a_norm = w_a / w_a.norm(p=2)

        # 计算欧几里得距离
        positive_distance = torch.norm(w_k_norm - w_a_norm, p=2)  # 异常距离
        negative_distance = torch.norm(w_k_norm - w_n_norm, p=2)  # 正常距离

        # 计算损失
        loss = torch.max(torch.tensor(0.0).to(w_k.device), positive_distance - negative_distance + margin)
        return loss



