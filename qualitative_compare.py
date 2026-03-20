import argparse
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

import AnomalyCLIP_lib
from HSF import HybridSemanticFusion
from adaptive_fusion import AdaptiveFeatureFusion
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform, normalize
from visual_prompt import VisualPromptAligner


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


class InferenceModel:
    def __init__(self, method_cfg, common_args, device):
        self.cfg = method_cfg
        self.device = device
        self.kind = method_cfg["type"]
        self.features_list = method_cfg.get("features_list", common_args.features_list)
        self.feature_map_layer = method_cfg.get("feature_map_layer", common_args.feature_map_layer)
        self.image_size = method_cfg.get("image_size", common_args.image_size)
        self.sigma = method_cfg.get("sigma", common_args.sigma)

        design_details = {
            "Prompt_length": method_cfg.get("n_ctx", common_args.n_ctx),
            "learnabel_text_embedding_depth": method_cfg.get("depth", common_args.depth),
            "learnabel_text_embedding_length": method_cfg.get("t_n_ctx", common_args.t_n_ctx),
        }
        self.design_details = design_details

        self.model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=design_details)
        self.model.eval()

        self.prompt_learner = AnomalyCLIP_PromptLearner(self.model.to("cpu"), design_details)
        checkpoint = torch.load(method_cfg["checkpoint"], map_location=device)
        self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        self.prompt_learner.to(device)
        self.model.to(device)
        self.model.visual.DAPM_replace(DPAM_layer=20)

        self.visual_prompt = None
        self.adaptive_fusion = None
        self.hsf = None

        if self.kind in {"visual_prompt", "visual_prompt_gate"}:
            self.visual_prompt = VisualPromptAligner(
                embed_dim=self.model.text_projection.shape[1],
                hidden_dim=method_cfg.get("visual_prompt_hidden_dim", common_args.visual_prompt_hidden_dim),
                dropout=method_cfg.get("visual_prompt_dropout", common_args.visual_prompt_dropout),
            ).to(device)
            self.visual_prompt.load_state_dict(checkpoint["visual_prompt"])
            self.visual_prompt.eval()
            self.hsf = HybridSemanticFusion(method_cfg.get("k_clusters", 20))

        if self.kind == "visual_prompt_gate":
            self.adaptive_fusion = AdaptiveFeatureFusion(
                embed_dim=self.model.text_projection.shape[1],
                hidden_dim=method_cfg.get("gate_hidden_dim", common_args.gate_hidden_dim),
                dropout=method_cfg.get("gate_dropout", common_args.gate_dropout),
            ).to(device)
            self.adaptive_fusion.load_state_dict(checkpoint["adaptive_fusion"])
            self.adaptive_fusion.eval()

        prompts, tokenized_prompts, compound_prompts_text, _, _ = self.prompt_learner(cls_id=None)
        text_features = self.model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.cluster_alpha = method_cfg.get("cluster_alpha", common_args.cluster_alpha)

    def infer_map(self, image_tensor):
        with torch.no_grad():
            image_features, patch_features = self.model.encode_image(image_tensor, self.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            if self.kind == "baseline":
                final_image_feature = image_features
            elif self.kind == "visual_prompt":
                enhanced_image_features, _, _ = self.visual_prompt(image_features, self.text_features)
                anomaly_maps = self._collect_anomaly_maps(patch_features)
                clustered_feature = self.hsf.forward(patch_features, anomaly_maps)
                final_image_feature = self.cluster_alpha * clustered_feature + (1 - self.cluster_alpha) * enhanced_image_features
                final_image_feature = F.normalize(final_image_feature, dim=1)
            elif self.kind == "visual_prompt_gate":
                aligned_visual_features, _, _ = self.visual_prompt(image_features, self.text_features)
                gated_visual_features, _ = self.adaptive_fusion.fuse_visual_prompt(image_features, aligned_visual_features)
                anomaly_maps = self._collect_anomaly_maps(patch_features)
                clustered_feature = self.hsf.forward(patch_features, anomaly_maps)
                final_image_feature, _ = self.adaptive_fusion.fuse_cluster_feature(gated_visual_features, clustered_feature)
            else:
                raise ValueError(f"Unsupported model type: {self.kind}")

            anomaly_map = self._build_map(patch_features)
            score = final_image_feature.unsqueeze(1) @ self.text_features.permute(0, 2, 1)
            score = torch.softmax(100.0 * score.squeeze(1), dim=-1)[:, 1].item()
            return anomaly_map, score

    def _collect_anomaly_maps(self, patch_features):
        anomaly_maps = []
        for idx, patch_feature in enumerate(patch_features):
            if idx >= self.feature_map_layer[0]:
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                _, anomaly_map = AnomalyCLIP_lib.compute_similarity(patch_feature, self.text_features[0])
                anomaly_maps.append(anomaly_map)
        return anomaly_maps

    def _build_map(self, patch_features):
        anomaly_map_list = []
        for idx, patch_feature in enumerate(patch_features):
            if idx >= self.feature_map_layer[0]:
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, self.text_features[0])
                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], self.image_size)
                anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                anomaly_map_list.append(anomaly_map)

        anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
        anomaly_map = torch.stack(
            [torch.from_numpy(gaussian_filter(i, sigma=self.sigma)) for i in anomaly_map.detach().cpu()],
            dim=0,
        )
        return anomaly_map[0].numpy()


def load_font(size):
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def build_figure(images, overlays, row_labels, col_labels, output_path, title=None):
    cell_w = 150
    cell_h = 150
    left_pad = 135
    top_pad = 80
    title_h = 60 if title else 20
    margin = 25
    rows = len(row_labels)
    cols = len(col_labels)
    canvas_w = left_pad + cols * cell_w + margin
    canvas_h = title_h + top_pad + rows * cell_h + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28)
    label_font = load_font(20)
    small_font = load_font(16)

    if title:
        draw.text((canvas_w // 2, 20), title, fill=(20, 20, 20), font=title_font, anchor="ma")

    for col, label in enumerate(col_labels):
        x = left_pad + col * cell_w + cell_w // 2
        y = title_h + 20
        draw.text((x, y), label, fill=(30, 30, 30), font=small_font, anchor="ma")

    for row, row_label in enumerate(row_labels):
        y = title_h + top_pad + row * cell_h + cell_h // 2
        draw.text((left_pad - 15, y), row_label, fill=(30, 30, 30), font=label_font, anchor="rm")
        for col in range(cols):
            img = overlays[row][col].resize((cell_w - 8, cell_h - 8))
            x = left_pad + col * cell_w + 4
            yy = title_h + top_pad + row * cell_h + 4
            canvas.paste(img, (x, yy))
            draw.rectangle([x, yy, x + cell_w - 8, yy + cell_h - 8], outline=(210, 210, 210), width=1)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def suggest_diverse_paths(image_paths, max_images):
    groups = {}
    for path in image_paths:
        parts = Path(path).parts
        label = parts[-4] if len(parts) >= 4 else Path(path).stem
        groups.setdefault(label, []).append(path)

    selected = []
    for label in sorted(groups):
        selected.append(groups[label][0])
        if len(selected) >= max_images:
            break

    if len(selected) < max_images:
        for path in image_paths:
            if path not in selected:
                selected.append(path)
                if len(selected) >= max_images:
                    break
    return selected


def main():
    parser = argparse.ArgumentParser("Qualitative Comparison Figure")
    parser.add_argument("--config", type=str, required=True, help="path to json config")
    parser.add_argument("--output", type=str, required=True, help="output figure path")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--visual_prompt_hidden_dim", type=int, default=0)
    parser.add_argument("--visual_prompt_dropout", type=float, default=0.1)
    parser.add_argument("--gate_hidden_dim", type=int, default=0)
    parser.add_argument("--gate_dropout", type=float, default=0.1)
    parser.add_argument("--cluster_alpha", type=float, default=0.2)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    image_paths = [item["path"] if isinstance(item, dict) else item for item in cfg["images"]]
    if cfg.get("auto_select_max_images"):
        image_paths = suggest_diverse_paths(image_paths, cfg["auto_select_max_images"])

    col_labels = []
    input_row = []
    preprocess_args = argparse.Namespace(image_size=args.image_size)
    preprocess, _ = get_transform(preprocess_args)

    for path in image_paths:
        p = Path(path)
        label = p.stem
        if len(p.parts) >= 3:
            label = f"{p.parts[-3]}/{p.stem}"
        col_labels.append(label)
        img = Image.open(path).convert("RGB")
        input_row.append(img)

    methods = cfg["methods"]
    models = [InferenceModel(method_cfg, args, device) for method_cfg in methods]

    overlays = [input_row]
    row_labels = ["Input"]
    for model_cfg, model in zip(methods, models):
        row_images = []
        for path in image_paths:
            pil_img = Image.open(path).convert("RGB")
            img = preprocess(pil_img)
            image_tensor = img.reshape(1, 3, args.image_size, args.image_size).to(device)
            anomaly_map, score = model.infer_map(image_tensor)
            overlay = apply_ad_scoremap(
                np.array(pil_img.resize((args.image_size, args.image_size))),
                normalize(anomaly_map),
                alpha=model_cfg.get("alpha", 0.5),
            )
            overlay_img = Image.fromarray(overlay.astype(np.uint8))
            draw = ImageDraw.Draw(overlay_img)
            score_font = load_font(18)
            draw.rounded_rectangle((8, 8, 88, 34), radius=8, fill=(255, 255, 255))
            draw.text((48, 22), f"{score:.3f}", fill=(20, 20, 20), font=score_font, anchor="mm")
            row_images.append(overlay_img)
        overlays.append(row_images)
        row_labels.append(model_cfg["name"])

    build_figure(overlays[0], overlays, row_labels, col_labels, args.output, cfg.get("title"))
    print(args.output)


if __name__ == "__main__":
    main()
