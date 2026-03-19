import base64
import io
from ollama import Client

import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss
from utils import normalize
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_transform
from prompt_ensemble import tokenize
from PIL import Image


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):


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

    # 加载训练数据集
    train_data = Dataset(root=args.train_data_path, transform=preprocess, target_transform=target_transform,
                         dataset_name=args.dataset)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    model.to(device)  # 将模型移动到设备

    # 开始训练循环
    model.eval()  # 设置模型为评估模式
    base_path = args.train_data_path


    # 遍历训练数据
    for items in tqdm(train_dataloader):
        index = 1
        # 获取图像和标签
        images_path = items['img_path']
        cls_names = items['cls_name']


        embedding_KNs = []
        for i in range(len(images_path)):
            image_path = images_path[i]

            prefix_name, _ = os.path.splitext(image_path)
            cls_name = cls_names[i]
            if os.path.exists(prefix_name + '.pt'):
                llm_embdding = torch.load(prefix_name + '.pt')
                llm_embdding = llm_embdding[-1]  # 取出数组的最后一个元素
                torch.save(llm_embdding, prefix_name + '.pt')  # 重新存入源文件，覆盖之前的内容
                continue
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            image = image_base64

            llm, vqa = generate_vqa_prompt(image, cls_name)
            dtype = model.transformer.get_cast_dtype()
            with torch.no_grad():
                # 生成相应的文本嵌入
                embedding_llm = model.token_embedding(tokenize(llm).to(device)).type(dtype)
                embedding_vqa = model.token_embedding(tokenize(vqa).to(device)).type(dtype)

            embedding_KN = torch.cat([embedding_llm, embedding_vqa], dim=2)

            torch.save(embedding_KN, prefix_name+'.pt')
            print(f"save {prefix_name}pt")


def generate_vqa_prompt(image_path, cls_name):
    """生成图像描述"""
    client = Client(
        host='http://192.168.18.62:11434',
    )

    # 准备请求数据
    response_img = client.chat(
        model='llama3.2-vision',

        messages=[
            {
                'role': 'user',
                'content': f'Identify anomalies in the input image of the {cls_name}. Describe each '
                           'anomaly\'s location, color, shape, size, and other characteristics.',
                'images': [image_path],
            },
        ], )
    response_text = client.chat(
        model='llama3.1',

        messages=[
            {
                'role': 'user',
                'content': f'Describe what an abnormal image of {cls_name} looks like? give five answers',
            },
        ], )
    return response_text.message.content, response_img.message.content


if __name__ == '__main__':
    print(torch.version.__version__)

    parser = argparse.ArgumentParser("AnomalyCLIP", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/mvtec", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoint/noForzen', help='path to save results')

    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")

    parser.add_argument("--depth", type=int, default=9, help="the depth of learnable text prompt")
    parser.add_argument("--n_ctx", type=int, default=12, help="the length of learnable text prompt")  # 消融实验建议在12-16之间选择
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
