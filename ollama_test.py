import base64
import io
from typing import List

import torch
from ollama import Client
from pathlib import Path
from PIL import Image
from statsmodels.datasets.utils import Dataset
from tqdm import tqdm

client = Client(
    host='http://192.168.18.62:11434',
)

# print(client.list())
# response = client.chat(model='deepseek-r1:8b',stream=True, messages=[
#     {
#         'role': 'user',
#         'content': 'Why is the sky blue?，用中文回答',
#     },
# ])

# 流式输出
# for chunk in response:
#     print(chunk['message']['content'], end='', flush=True)

# 直接输出
# print(response['message']['content'])
# or access fields directly from the response object
# print(response.message.content)
# Define the schema for image objects



def generate_llm_descriptions(class_name: str, n: int = 5) -> List[str]:
    """调用LLM生成类别的通用异常描述"""
    prompt = f"Q: Describe what an abnormal image of class {class_name} looks like? Provide {n} possible answers."
    response_text = client.chat(
        model='deepseek-r1:8b',

        messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ],
    )
    # 解析返回的文本，提取n条描述（此处简化处理）
    descriptions = [f"An abnormal {class_name} has {desc}" for desc in response_text.choices[0].text.split("\n")[:n]]
    return descriptions

def generate_vqa_descriptions(image: Image.Image, class_name: str, m: int = 1) -> List[str]:
    """调用VQA生成图像特定的异常描述"""
    vqa_prompt = f"<IMAGE> Q: Identify anomalies in this {class_name}. Describe their location, color, shape, and size."
    response_img = client.chat(
        model='llama3.2-vision',

        messages=[
            {
                'role': 'user',
                'content': vqa_prompt,
                'images': [image],
            },
        ],
    )
    return response_img.message.content




if __name__ == '__main__':
    train_data = Dataset(root="./data/mvtec",
                         dataset_name="mvtec")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)
    # 设置设备为 GPU 或 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for items in tqdm(train_dataloader):
        image = items['img'].to(device)  # 获取图像并移动到设备
        label = items['anomaly']  # 获取标签
        print(label+"===="+image)