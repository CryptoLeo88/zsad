import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F

from HSF import HybridSemanticFusion
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss, KD_Loss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import os

import random
from utils import get_transform
import warnings

# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


os.environ["OMP_NUM_THREADS"] = '1'
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(args):
    # 获取日志记录器
    logger = get_logger(args.save_path)

    # 获取数据预处理和目标变换函数
    preprocess, target_transform = get_transform(args)
    # 设置设备为 GPU 或 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 设置 AnomalyCLIP 参数
    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    # 加载 AnomalyCLIP 模型
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()  # 设置模型为评估模式


    hsf = HybridSemanticFusion(20)
    # 加载训练数据集
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform, dataset_name=args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # 初始化 AnomalyCLIP_PromptLearner 对象
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    prompt_learner.to(device)  # 将 prompt_learner 移动到设备
    model.to(device)  # 将模型移动到设备
    model.visual.DAPM_replace(DPAM_layer=20)  # 替换模型的视觉部分

    # 初始化优化器
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=args.learning_rate, betas=(0.5, 0.999))

    # 初始化损失函数
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_kd = KD_Loss()

    model.eval()  # 设置模型为评估模式
    prompt_learner.train()  # 设置 prompt_learner 为训练模式

    # 开始训练循环
    for epoch in tqdm(range(args.epoch)):
        model.eval()  # 设置模型为评估模式
        prompt_learner.train()  # 设置 prompt_learner 为训练模式
        loss_list = []
        image_loss_list = []

        # 遍历训练数据
        for items in tqdm(train_dataloader):
            image = items['img'].to(device)  # 获取图像并移动到设备
            label = items['anomaly']  # 获取标签
            gt = items['img_mask'].squeeze().to(device)  # 获取图像掩码并移动到设备
            gt[gt > 0.5] = 1  # 将掩码值大于0.5的部分设为1
            gt[gt <= 0.5] = 0  # 将掩码值小于等于0.5的部分设为0

            with torch.no_grad():
                # 编码图像特征
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 归一化图像特征

            # 获取文本特征
            prompts, tokenized_prompts, compound_prompts_text,prompts_pos,prompts_neg = prompt_learner(cls_id=None)

            # 用可学习/冻结的clip 文本编码器
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()

            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)  # 归一化文本特征

            # 计算图像和文本特征的相似度
            text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().cuda())  # 计算图像损失
            image_loss_list.append(image_loss.item())

            similarity_map_list = []
            anomaly_maps = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)  # 归一化 patch 特征
                    # F_s_a, F_t_a = Zero_try(text_embeddings.permute(0, 2, 1), dense_feature)
                    # anomaly_map_new = (Zero_try.prompt_temp_l1.exp() * dense_feature @ F_t_a.permute(0, 2, 1))
                    similarity, anomaly_map = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    anomaly_maps.append(anomaly_map)

                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size).permute(0, 3, 1, 2)

                    similarity_map_list.append(similarity_map)

            alpha = 0.2
            clustered_feature = hsf.forward(patch_features, anomaly_maps)
            # aggregate the class token and the clustered features for more comprehensive information
            cur_image_feature = alpha * clustered_feature + (1 - alpha) * image_features
            cur_image_feature = F.normalize(cur_image_feature, dim=1)
            anomaly_score = (100.0 * cur_image_feature.unsqueeze(1) @ text_features.permute(0, 2, 1))
            anomaly_score = anomaly_score.squeeze(1)
            anomaly_score = torch.softmax(anomaly_score, dim=1)

            # 计算损失
            loss = 0

            # classification loss
            is_anomaly = items['anomaly'].to(device)
            is_anomaly[is_anomaly > 0.5] = 1
            is_anomaly[is_anomaly <= 0.5] = 0


            classification_loss = loss_focal(anomaly_score, is_anomaly.unsqueeze(1))
            loss += classification_loss
            for i in range(len(similarity_map_list)):
                loss += loss_focal(similarity_map_list[i], gt)
                loss += loss_dice(similarity_map_list[i][:, 1, :, :], gt)
                loss += loss_dice(similarity_map_list[i][:, 0, :, :], 1 - gt)

            optimizer.zero_grad()  # 清空梯度
            loss_KD = loss_kd(items['llm_embedding'].to(device), prompts_pos, prompts_neg)
            # print(loss_KD)
            loss += loss_KD

            (loss + image_loss).backward()  # 反向传播
            optimizer.step()  # 更新参数
            loss_list.append(loss.item())

        # 记录日志
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list), np.mean(image_loss_list)))

        # 保存模型
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

if __name__ == '__main__':
    print(torch.version.__version__)

    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/mvtec", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint/noForzen/hsf', help='path to save results')

    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="the depth of learnable text prompt")
    parser.add_argument("--n_ctx", type=int, default=12, help="the length of learnable text prompt") #消融实验建议在12-16之间选择
    parser.add_argument("--t_n_ctx", type=int, default=4, help="learnabel_text_embedding_length")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="zero shot")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")

    parser.add_argument("--epoch", type=int, default=15, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--print_freq", type=int, default=1, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=1, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
