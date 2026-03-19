import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel

# 加载模型与工具类
from KnPL_test import KnowledgeDrivenPromptLearner
from dataset import Dataset
from utils import get_transform

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model = CLIPModel.from_pretrained(args.clip_path).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_path)
    preprocess, target_transform = get_transform(args)

    prompt_learner = KnowledgeDrivenPromptLearner(
        clip_model_path=args.clip_path,
        prompt_length=args.prompt_length,
        num_classes=args.num_classes,
        margin=args.margin
    ).to(device)

    train_data = Dataset(
        root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name = args.dataset
    )
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(
        prompt_learner.parameters(),
        lr=args.learning_rate,
        betas=(0.5, 0.999)
    )

    criterion = nn.CrossEntropyLoss()

    print("Starting training...")
    for epoch in range(args.epochs):
        prompt_learner.train()
        total_loss = 0
        batch_count = 0

        for batch in tqdm(train_dataloader):
            images = batch['img'].to(device)
            labels = batch['anomaly'].to(device)

            # === 生成类别标签 ===
            class_ids = labels

            # === 知识驱动损失 ===
            kd_loss = prompt_learner(class_ids)  # 直接输出 L_kd 损失

            # === 图像与文本特征匹配 ===
            image_features = clip_model.get_image_features(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = prompt_learner.encode_descriptions(["normal", "abnormal"])
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = image_features @ text_features.T
            image_loss = criterion(similarity, labels)

            # === 总损失 ===
            loss = kd_loss + image_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {avg_loss:.4f}")

        # === 保存模型 ===
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_path, f"epoch_{epoch + 1}.pth")
            torch.save({
                "prompt_learner_state_dict": prompt_learner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)
            print(f"Model saved to {save_path}")


# === 定义参数 ===
class Args:
    train_data_path = "./data/mvtec"
    save_path = "./checkpoints/knowledge_driven_prompt_learner"
    batch_size = 8
    epochs = 2
    learning_rate = 1e-4
    prompt_length = 12
    num_classes = 10
    margin = 0.2
    log_freq = 1
    save_freq = 5
    clip_path = 'Clip_lib/clip-vit-large-patch14'
    image_size = 224
    dataset = 'mvtec'


# === 启动训练 ===
if __name__ == "__main__":
    args = Args()
    train(args)