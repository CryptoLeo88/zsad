
import requests
import base64
from PIL import Image
import io
import torch
import numpy as np
from ollama import Client

class VQAGenerator:
    def __init__(self, model_url="http://192.168.18.62:11434/"):
        self.model_url = model_url
        self.normal_template = "Describe this image focusing on its normal appearance and quality."
        self.anomaly_template = "Describe any defects, damages, or abnormalities in this image."
        self.client = Client(host='http://192.168.18.62:11434',)
        
    def _encode_image(self, image_path):
        """将图像转换为base64编码"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def generate_prompt(self, image_path, prompt_type="normal"):
        """生成图像描述"""
        template = self.normal_template if prompt_type == "normal" else self.anomaly_template
        
        # 准备请求数据
        response_img = self.client.chat(
            model='llama3.2-vision',

            messages=[
                {
                    'role': 'user',
                    'content': 'Identify anomalies in the input image of the pill. Describe each '
                            'anomaly\'s location, color, shape, size, and other characteristics.',
                    'images': [self._encode_image(image_path)],
                },
            ],)

        return response_img['messages'];

    def get_vqa_embeddings(self, image_path, clip_model):
        """获取VQA生成的文本嵌入"""
        # 获取正常和异常描述
        normal_desc = self.generate_prompt(image_path, "normal")
        anomaly_desc = self.generate_prompt(image_path, "anomaly")
        
        # 使用CLIP模型编码文本
        with torch.no_grad():
            normal_embedding = clip_model.encode_text([normal_desc])
            anomaly_embedding = clip_model.encode_text([anomaly_desc])
            
        # 归一化嵌入
        normal_embedding = normal_embedding / normal_embedding.norm(dim=-1, keepdim=True)
        anomaly_embedding = anomaly_embedding / anomaly_embedding.norm(dim=-1, keepdim=True)
        
        return normal_embedding, anomaly_embedding

