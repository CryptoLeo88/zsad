import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageFilter


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(text: str) -> str:
    safe = [ch.lower() if ch.isalnum() else "-" for ch in text.strip()]
    slug = "".join(safe).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "inspection"


def _score_to_label(score: float) -> str:
    if score >= 0.7:
        return "高风险异常"
    if score >= 0.4:
        return "疑似异常"
    return "状态正常"


def _normalize_map(array: np.ndarray) -> np.ndarray:
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if abs(max_value - min_value) < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - min_value) / (max_value - min_value)).astype(np.float32)


@dataclass
class DetectionConfig:
    image_size: int = 518
    features_list: tuple = (6, 12, 18, 24)
    feature_map_layer: tuple = (0, 1, 2, 3)
    depth: int = 9
    n_ctx: int = 12
    t_n_ctx: int = 4
    sigma: int = 4
    checkpoint_path: Optional[str] = None


class RealAnomalyClipEngine:
    def __init__(self, config: DetectionConfig) -> None:
        if not config.checkpoint_path or not Path(config.checkpoint_path).exists():
            raise FileNotFoundError("checkpoint not found")

        import torch
        from scipy.ndimage import gaussian_filter

        import AnomalyCLIP_lib
        from prompt_ensemble import AnomalyCLIP_PromptLearner
        from utils import get_transform

        self.torch = torch
        self.gaussian_filter = gaussian_filter
        self.AnomalyCLIP_lib = AnomalyCLIP_lib
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        parameters = {
            "Prompt_length": config.n_ctx,
            "learnabel_text_embedding_depth": config.depth,
            "learnabel_text_embedding_length": config.t_n_ctx,
        }
        self.model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=self.device, design_details=parameters)
        self.model.eval()
        self.prompt_learner = AnomalyCLIP_PromptLearner(self.model.to("cpu"), parameters)
        checkpoint = torch.load(config.checkpoint_path, map_location=self.device)
        self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        self.prompt_learner.to(self.device)
        self.model.to(self.device)
        self.model.visual.DAPM_replace(DPAM_layer=20)
        prompts, tokenized_prompts, compound_prompts_text, _, _ = self.prompt_learner(cls_id=None)
        text_features = self.model.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text
        ).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        args = SimpleNamespace(image_size=config.image_size)
        self.preprocess, _ = get_transform(args)

    def infer(self, image_path: str) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        img = self.preprocess(image)
        image_tensor = img.reshape(1, 3, self.config.image_size, self.config.image_size).to(self.device)
        with self.torch.no_grad():
            image_features, patch_features = self.model.encode_image(
                image_tensor, list(self.config.features_list), DPAM_layer=20
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_probs = image_features @ self.text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            image_score = float(text_probs[:, 0, 1].item())
            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= self.config.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = self.AnomalyCLIP_lib.compute_similarity(patch_feature, self.text_features[0])
                    similarity_map = self.AnomalyCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :], self.config.image_size
                    )
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            stacked = self.torch.stack(anomaly_map_list).sum(dim=0)
            smoothed = self.torch.stack(
                [self.torch.from_numpy(self.gaussian_filter(i, sigma=self.config.sigma)) for i in stacked.cpu()],
                dim=0,
            )
        return {
            "score": image_score,
            "anomaly_map": smoothed[0].numpy(),
            "mode": "real",
        }


class DemoDefectEngine:
    def infer(self, image_path: str) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        blur = np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32)
        detail = np.abs(gray - blur)
        grad_y, grad_x = np.gradient(gray)
        gradient = np.hypot(grad_x, grad_y).astype(np.float32)
        chroma = np.std(rgb.astype(np.float32), axis=2)
        heat = (
            0.45 * _normalize_map(detail.astype(np.float32))
            + 0.35 * _normalize_map(gradient.astype(np.float32))
            + 0.2 * _normalize_map(chroma.astype(np.float32))
        )
        heat_image = Image.fromarray(np.uint8(np.clip(heat * 255, 0, 255)))
        heat = np.asarray(heat_image.filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32) / 255.0
        score = float(np.clip(np.percentile(heat, 96), 0.0, 1.0))
        return {
            "score": score,
            "anomaly_map": heat,
            "mode": "demo",
            "image_rgb": rgb,
        }


class OpenVocabularyDefectSystem:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()
        self.upload_dir = _ensure_dir(base_dir / "uploads")
        self.output_dir = _ensure_dir(base_dir / "static" / "generated")
        self.data_dir = _ensure_dir(base_dir / "data")
        self.history_path = self.data_dir / "history.json"
        checkpoint_path = os.environ.get("ANOMALYCLIP_CHECKPOINT")
        self.config = DetectionConfig(checkpoint_path=checkpoint_path)
        try:
            self.engine = RealAnomalyClipEngine(self.config)
            self.runtime_mode = "real"
        except Exception:
            self.engine = DemoDefectEngine()
            self.runtime_mode = "demo"

    def analyze(self, image_file, defect_terms: List[str], scene_name: str) -> Dict[str, Any]:
        inspection_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        ext = Path(image_file.filename or "upload.png").suffix or ".png"
        original_name = _slugify(Path(image_file.filename or "inspection").stem)
        upload_path = self.upload_dir / f"{inspection_id}-{original_name}{ext}"
        image_file.save(upload_path)

        raw_result = self.engine.infer(str(upload_path))
        original = np.asarray(Image.open(upload_path).convert("RGB"), dtype=np.uint8)
        heat = _normalize_map(raw_result["anomaly_map"].astype(np.float32))
        overlay = self._build_overlay(original, heat)
        boxes = self._extract_regions(heat, original.shape[:2])
        matched_terms, unknown_flag, explanation = self._match_terms(defect_terms, heat, boxes, float(raw_result["score"]))

        overlay_name = f"{inspection_id}-overlay.png"
        overlay_path = self.output_dir / overlay_name
        Image.fromarray(overlay).save(overlay_path)

        record = {
            "id": inspection_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scene_name": scene_name or "默认产线",
            "image_name": image_file.filename or upload_path.name,
            "mode": "真实模型" if raw_result["mode"] == "real" else "演示模式",
            "score": round(float(raw_result["score"]), 3),
            "decision": _score_to_label(float(raw_result["score"])),
            "matched_terms": matched_terms,
            "unknown_flag": unknown_flag,
        }
        self._append_history(record)

        return {
            "record": record,
            "original_image": f"/uploads/{upload_path.name}",
            "overlay_image": f"/generated/{overlay_name}",
            "terms": defect_terms,
            "heat_peak": round(float(np.max(heat)), 3),
            "heat_mean": round(float(np.mean(heat)), 3),
            "regions": boxes,
            "explanation": explanation,
            "history": self.load_history(),
            "runtime_mode": self.runtime_mode,
        }

    def load_history(self) -> List[Dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return json.loads(self.history_path.read_text(encoding="utf-8"))

    def _append_history(self, record: Dict[str, Any]) -> None:
        history = self.load_history()
        history.insert(0, record)
        self.history_path.write_text(json.dumps(history[:12], ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_overlay(self, image_bgr: np.ndarray, heat: np.ndarray) -> np.ndarray:
        color = np.zeros((heat.shape[0], heat.shape[1], 3), dtype=np.float32)
        color[:, :, 0] = np.clip(1.6 * heat - 0.1, 0.0, 1.0)
        color[:, :, 1] = np.clip(1.3 - np.abs(heat - 0.45) * 3.0, 0.0, 1.0)
        color[:, :, 2] = np.clip(1.0 - 1.5 * heat, 0.0, 1.0)
        blended = image_bgr.astype(np.float32) * 0.58 + color * 255.0 * 0.42
        return np.uint8(np.clip(blended, 0, 255))

    def _extract_regions(self, heat: np.ndarray, image_shape: tuple) -> List[Dict[str, Any]]:
        threshold = max(0.45, float(np.percentile(heat, 90)))
        binary = heat >= threshold
        visited = np.zeros_like(binary, dtype=bool)
        regions = []
        height, width = image_shape
        for y in range(height):
            for x in range(width):
                if not binary[y, x] or visited[y, x]:
                    continue
                queue = [(x, y)]
                visited[y, x] = True
                xs = []
                ys = []
                while queue:
                    cx, cy = queue.pop()
                    xs.append(cx)
                    ys.append(cy)
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if 0 <= nx < width and 0 <= ny < height and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((nx, ny))
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                w = x1 - x0 + 1
                h = y1 - y0 + 1
                area_ratio = float((w * h) / max(width * height, 1))
                if area_ratio < 0.003:
                    continue
                regions.append(
                    {
                        "x": int(x0),
                        "y": int(y0),
                        "w": int(w),
                        "h": int(h),
                        "area_ratio": round(area_ratio, 3),
                    }
                )
        regions.sort(key=lambda item: item["area_ratio"], reverse=True)
        return regions[:3]

    def _match_terms(
        self,
        defect_terms: List[str],
        heat: np.ndarray,
        regions: List[Dict[str, Any]],
        score: float,
    ) -> tuple[List[str], bool, str]:
        if not defect_terms:
            defect_terms = ["划痕", "裂纹", "污渍", "孔洞"]

        if score < 0.2:
            return [], False, "当前图像整体异常响应较低，系统判定为正常样本或低风险样本。"

        spread = float(np.mean(heat > 0.55))
        elongated = any(max(region["w"], region["h"]) / max(min(region["w"], region["h"]), 1) > 3 for region in regions)
        compact = any(0.005 < region["area_ratio"] < 0.03 for region in regions)
        matched = []
        for term in defect_terms:
            lower = term.lower()
            if any(key in lower for key in ["scratch", "划痕", "裂纹", "crack"]) and elongated:
                matched.append(term)
            elif any(key in lower for key in ["spot", "污", "脏", "stain", "污染"]) and spread > 0.08:
                matched.append(term)
            elif any(key in lower for key in ["hole", "孔", "缺口", "凹坑"]) and compact:
                matched.append(term)

        unknown_flag = bool(np.max(heat) > 0.82 and not matched)
        if regions:
            explanation = f"系统在图像中定位到 {len(regions)} 个候选异常区域，热力峰值为 {float(np.max(heat)):.2f}。"
        else:
            explanation = "当前图像未形成稳定异常区域，整体响应较分散。"
        if unknown_flag:
            explanation += " 现有词汇库未能准确覆盖该模式，建议作为未知缺陷加入复核队列。"
        elif matched:
            explanation += f" 结合开放词汇库，当前更接近 {', '.join(matched)}。"
        elif score >= 0.45:
            matched = defect_terms[:1]
            explanation += f" 当前存在异常响应，但尚缺少更精确的语义区分，暂以 {matched[0]} 进入复核流程。"
        return matched, unknown_flag, explanation
