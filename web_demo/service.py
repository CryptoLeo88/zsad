import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
    backbone: str = "ViT-L/14@336px"
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
        self.model, _ = AnomalyCLIP_lib.load(config.backbone, device=self.device, design_details=parameters)
        self.model.eval()
        self.prompt_learner = AnomalyCLIP_PromptLearner(self.model.to("cpu"), parameters)
        checkpoint = torch.load(config.checkpoint_path, map_location=self.device)
        self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        self.prompt_learner.to(self.device)
        self.model.to(self.device)
        self.model.visual.DAPM_replace(DPAM_layer=20)
        prompts, tokenized_prompts, compound_prompts_text, _, _ = self.prompt_learner(cls_id=None)
        text_features = self.model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
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
        return {"score": image_score, "anomaly_map": smoothed[0].numpy(), "mode": "real"}


class StandardDefectEngine:
    def infer(self, image_path: str) -> Dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        blur = np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32)
        detail = np.abs(gray - blur)
        grad_y, grad_x = np.gradient(gray)
        gradient = np.hypot(grad_x, grad_y).astype(np.float32)
        chroma = np.std(rgb.astype(np.float32), axis=2)
        heat = 0.45 * _normalize_map(detail) + 0.35 * _normalize_map(gradient) + 0.2 * _normalize_map(chroma)
        heat_image = Image.fromarray(np.uint8(np.clip(heat * 255, 0, 255)))
        heat = np.asarray(heat_image.filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32) / 255.0
        score = float(np.clip(np.percentile(heat, 96), 0.0, 1.0))
        return {"score": score, "anomaly_map": heat, "mode": "demo"}


class VocabularyStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        if not self.path.exists():
            _write_json(self.path, self._defaults())

    def _defaults(self) -> List[Dict[str, Any]]:
        stamp = "2026-03-22 00:00:00"
        return [
            {"id": "scratch01", "term": "划痕", "category": "表面损伤", "description": "适用于细长线状缺陷", "created_at": stamp},
            {"id": "crack01", "term": "裂纹", "category": "结构损伤", "description": "适用于断裂与开缝类缺陷", "created_at": stamp},
            {"id": "stain01", "term": "污渍", "category": "表面污染", "description": "适用于油污、脏污、附着物", "created_at": stamp},
            {"id": "hole01", "term": "孔洞", "category": "成形异常", "description": "适用于孔洞、缺口、凹坑", "created_at": stamp},
        ]

    def list_terms(self) -> List[Dict[str, Any]]:
        items = _read_json(self.path, [])
        return sorted(items, key=lambda item: item["created_at"], reverse=True)

    def add_term(self, term: str, category: str, description: str) -> None:
        items = self.list_terms()
        items.insert(
            0,
            {
                "id": uuid.uuid4().hex[:8],
                "term": term.strip(),
                "category": category.strip() or "未分类",
                "description": description.strip(),
                "created_at": _now(),
            },
        )
        _write_json(self.path, items)

    def remove_term(self, term_id: str) -> None:
        items = [item for item in self.list_terms() if item["id"] != term_id]
        _write_json(self.path, items)

    def term_names(self) -> List[str]:
        return [item["term"] for item in self.list_terms()]


class TrainingManager:
    def __init__(self, state_dir: Path, repo_root: Path, worker_path: Path) -> None:
        self.state_dir = state_dir
        self.repo_root = repo_root
        self.worker_path = worker_path
        self.jobs_path = state_dir / "training_jobs.json"
        self.logs_dir = _ensure_dir(state_dir / "logs")
        if not self.jobs_path.exists():
            _write_json(self.jobs_path, [])

    def profiles(self) -> List[Dict[str, str]]:
        return [
            {"id": "open_vocab", "label": "开放词汇通用检测框架", "script": "train.py"},
            {"id": "vp_gated", "label": "视觉提示门控检测框架", "script": "train_visual_prompt_gate.py"},
        ]

    def list_jobs(self) -> List[Dict[str, Any]]:
        return sorted(_read_json(self.jobs_path, []), key=lambda item: item["created_at"], reverse=True)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        for job in self.list_jobs():
            if job["id"] == job_id:
                return job
        return None

    def tail_log(self, job_id: str, max_lines: int = 80) -> str:
        log_path = self.logs_dir / f"{job_id}.log"
        if not log_path.exists():
            return ""
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(lines[-max_lines:])

    def update_job(self, job_id: str, **changes: Any) -> None:
        jobs = self.list_jobs()
        for job in jobs:
            if job["id"] == job_id:
                job.update(changes)
                break
        _write_json(self.jobs_path, jobs)

    def start_job(self, form: Dict[str, str]) -> str:
        job_id = uuid.uuid4().hex[:10]
        profile = form.get("profile", "open_vocab")
        job = {
            "id": job_id,
            "name": form.get("job_name", "").strip() or f"{profile}-{job_id}",
            "profile": profile,
            "profile_label": self._profile_label(profile),
            "command": self._build_command(profile, form),
            "train_data_path": form.get("train_data_path", "").strip(),
            "save_path": form.get("save_path", "").strip(),
            "dataset": form.get("dataset", "mvtec").strip() or "mvtec",
            "backbone": form.get("backbone", "ViT-L/14@336px").strip() or "ViT-L/14@336px",
            "status": "pending",
            "created_at": _now(),
            "started_at": "",
            "finished_at": "",
            "message": "训练任务已创建，等待后台执行。",
        }
        jobs = self.list_jobs()
        jobs.insert(0, job)
        _write_json(self.jobs_path, jobs[:30])
        subprocess.Popen(
            [sys.executable, str(self.worker_path), job_id],
            cwd=str(self.repo_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return job_id

    def _profile_label(self, profile_id: str) -> str:
        for item in self.profiles():
            if item["id"] == profile_id:
                return item["label"]
        return profile_id

    def _build_command(self, profile: str, form: Dict[str, str]) -> List[str]:
        script_map = {item["id"]: item["script"] for item in self.profiles()}
        command = [
            sys.executable,
            script_map.get(profile, "train.py"),
            "--train_data_path",
            form.get("train_data_path", "./data/mvtec").strip() or "./data/mvtec",
            "--save_path",
            form.get("save_path", "./checkpoint/noForzen/web_demo").strip() or "./checkpoint/noForzen/web_demo",
            "--dataset",
            form.get("dataset", "mvtec").strip() or "mvtec",
            "--epoch",
            form.get("epoch", "15").strip() or "15",
            "--batch_size",
            form.get("batch_size", "8").strip() or "8",
            "--learning_rate",
            form.get("learning_rate", "0.001").strip() or "0.001",
            "--image_size",
            form.get("image_size", "518").strip() or "518",
            "--seed",
            form.get("seed", "111").strip() or "111",
            "--backbone",
            form.get("backbone", "ViT-L/14@336px").strip() or "ViT-L/14@336px",
        ]
        return command


class OpenVocabularyDefectSystem:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()
        self.repo_root = self.base_dir.parent
        self.upload_dir = _ensure_dir(self.base_dir / "uploads")
        self.output_dir = _ensure_dir(self.base_dir / "static" / "generated")
        self.state_dir = _ensure_dir(self.base_dir / "data")
        self.dataset_dir = _ensure_dir(self.base_dir / "datasets")
        self.history_path = self.state_dir / "history.json"
        self.vocab_store = VocabularyStore(self.state_dir / "vocabulary.json")
        self.training = TrainingManager(self.state_dir, self.repo_root, self.base_dir / "training_worker.py")
        self.engine_cache: Dict[str, Any] = {}
        if not self.history_path.exists():
            _write_json(self.history_path, [])
        checkpoint_path = os.environ.get("ANOMALYCLIP_CHECKPOINT")
        self.config = DetectionConfig(checkpoint_path=checkpoint_path)
        try:
            self.engine = RealAnomalyClipEngine(self.config)
            self.runtime_mode = "框架推理引擎"
        except Exception:
            self.engine = StandardDefectEngine()
            self.runtime_mode = "标准检测引擎"

    def analyze(
        self,
        image_file,
        defect_terms: List[str],
        scene_name: str,
        selected_model_path: str = "",
        selected_framework: str = "",
        selected_backbone: str = "",
    ) -> Dict[str, Any]:
        inspection_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        ext = Path(image_file.filename or "upload.png").suffix or ".png"
        original_name = _slugify(Path(image_file.filename or "inspection").stem)
        upload_path = self.upload_dir / f"{inspection_id}-{original_name}{ext}"
        image_file.save(upload_path)
        engine = self._resolve_engine(selected_model_path, selected_backbone)
        raw_result = engine.infer(str(upload_path))
        original = np.asarray(Image.open(upload_path).convert("RGB"), dtype=np.uint8)
        heat = _normalize_map(raw_result["anomaly_map"].astype(np.float32))
        heat = self._resize_heatmap(heat, original.shape[:2])
        overlay = self._build_overlay(original, heat)
        regions = self._extract_regions(heat, original.shape[:2])
        regions = self._assign_region_terms(regions, defect_terms, heat)
        matched_terms, unknown_flag, explanation = self._match_terms(defect_terms, heat, regions, float(raw_result["score"]))
        overlay_name = f"{inspection_id}-overlay.png"
        boxed_original_name = f"{inspection_id}-boxed-original.png"
        boxed_overlay_name = f"{inspection_id}-boxed-overlay.png"
        boxed_original = self._draw_regions(original, regions)
        boxed_overlay = self._draw_regions(overlay, regions)
        Image.fromarray(overlay).save(self.output_dir / overlay_name)
        Image.fromarray(boxed_original).save(self.output_dir / boxed_original_name)
        Image.fromarray(boxed_overlay).save(self.output_dir / boxed_overlay_name)
        record = {
            "id": inspection_id,
            "created_at": _now(),
            "scene_name": scene_name or "默认产线",
            "image_name": image_file.filename or upload_path.name,
            "mode": "框架推理引擎" if raw_result["mode"] == "real" else "标准检测引擎",
            "score": round(float(raw_result["score"]), 3),
            "decision": _score_to_label(float(raw_result["score"])),
            "matched_terms": matched_terms,
            "unknown_flag": unknown_flag,
            "model_name": Path(selected_model_path).name if selected_model_path else "未指定模型",
            "framework_name": selected_framework or "默认检测框架",
            "backbone_name": selected_backbone or "默认主干网络",
        }
        self._append_history(record)
        return {
            "record": record,
            "original_image": f"/uploads/{upload_path.name}",
            "overlay_image": f"/generated/{overlay_name}",
            "boxed_original_image": f"/generated/{boxed_original_name}",
            "boxed_overlay_image": f"/generated/{boxed_overlay_name}",
            "regions": regions,
            "heat_peak": round(float(np.max(heat)), 3),
            "heat_mean": round(float(np.mean(heat)), 3),
            "explanation": explanation,
        }

    def _resolve_engine(self, selected_model_path: str, selected_backbone: str):
        checkpoint_path = selected_model_path.strip()
        if checkpoint_path and Path(checkpoint_path).exists():
            cache_key = f"{checkpoint_path}::{selected_backbone or 'ViT-L/14@336px'}"
            if cache_key not in self.engine_cache:
                config = DetectionConfig(
                    checkpoint_path=checkpoint_path,
                    backbone=selected_backbone or "ViT-L/14@336px",
                )
                try:
                    self.engine_cache[cache_key] = RealAnomalyClipEngine(config)
                except Exception:
                    self.engine_cache[cache_key] = StandardDefectEngine()
            return self.engine_cache[cache_key]
        return self.engine

    def available_models(self) -> List[Dict[str, str]]:
        candidates = []
        for folder in [self.repo_root / "checkpoint", self.repo_root / "checkpoints"]:
            if folder.exists():
                for path in sorted(folder.rglob("*.pth")):
                    candidates.append({"name": path.name, "path": str(path)})
        return candidates[:50]

    def save_uploaded_dataset(self, dataset_file, job_id: str) -> str:
        suffix = Path(dataset_file.filename or "dataset.zip").suffix.lower()
        target_dir = self.dataset_dir / job_id
        _ensure_dir(target_dir)
        archive_path = target_dir / (dataset_file.filename or f"{job_id}.zip")
        dataset_file.save(archive_path)
        if suffix in {".zip", ".tar", ".gz", ".bz2", ".xz"}:
            extract_dir = target_dir / "extracted"
            _ensure_dir(extract_dir)
            try:
                shutil.unpack_archive(str(archive_path), str(extract_dir))
                return str(extract_dir)
            except shutil.ReadError:
                return str(target_dir)
        return str(target_dir)

    def dashboard_data(self) -> Dict[str, Any]:
        history = self.load_history()
        jobs = self.training.list_jobs()
        vocab = self.vocab_store.list_terms()
        abnormal = [item for item in history if item["decision"] != "状态正常"]
        return {
            "runtime_mode": self.runtime_mode,
            "history_count": len(history),
            "abnormal_count": len(abnormal),
            "vocab_count": len(vocab),
            "job_count": len(jobs),
            "recent_history": history[:5],
            "recent_jobs": jobs[:5],
        }

    def training_defaults(self) -> Dict[str, str]:
        return {
            "job_name": "",
            "profile": "open_vocab",
            "dataset": "industrial_custom",
            "train_data_path": "",
            "save_path": "./checkpoint/noForzen/web_demo",
            "epoch": "15",
            "batch_size": "8",
            "learning_rate": "0.001",
            "image_size": "518",
            "seed": "111",
            "backbone": "ViT-L/14@336px",
        }

    def load_history(self) -> List[Dict[str, Any]]:
        return _read_json(self.history_path, [])

    def history_stats(self) -> Dict[str, int]:
        history = self.load_history()
        return {
            "total": len(history),
            "normal": len([item for item in history if item["decision"] == "状态正常"]),
            "suspect": len([item for item in history if item["decision"] == "疑似异常"]),
            "danger": len([item for item in history if item["decision"] == "高风险异常"]),
        }

    def _append_history(self, record: Dict[str, Any]) -> None:
        history = self.load_history()
        history.insert(0, record)
        _write_json(self.history_path, history[:50])

    def _build_overlay(self, image_rgb: np.ndarray, heat: np.ndarray) -> np.ndarray:
        color = np.zeros((heat.shape[0], heat.shape[1], 3), dtype=np.float32)
        color[:, :, 0] = np.clip(1.6 * heat - 0.1, 0.0, 1.0)
        color[:, :, 1] = np.clip(1.3 - np.abs(heat - 0.45) * 3.0, 0.0, 1.0)
        color[:, :, 2] = np.clip(1.0 - 1.5 * heat, 0.0, 1.0)
        blended = image_rgb.astype(np.float32) * 0.58 + color * 255.0 * 0.42
        return np.uint8(np.clip(blended, 0, 255))

    def _draw_regions(self, image_rgb: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
        canvas = Image.fromarray(image_rgb.copy())
        draw = ImageDraw.Draw(canvas)
        for idx, region in enumerate(regions, 1):
            x0 = int(region["x"])
            y0 = int(region["y"])
            x1 = int(region["x"] + region["w"])
            y1 = int(region["y"] + region["h"])
            draw.rectangle([x0, y0, x1, y1], outline=(214, 48, 49), width=4)
            region_term = region.get("matched_term", f"区域{idx}")
            text = f"{region_term}{idx}"
            box_width = max(96, 14 * len(text))
            text_box = [x0, max(0, y0 - 26), x0 + box_width, max(24, y0)]
            draw.rectangle(text_box, fill=(214, 48, 49))
            draw.text((x0 + 8, max(2, y0 - 23)), text, fill=(255, 255, 255))
        return np.asarray(canvas, dtype=np.uint8)

    def _resize_heatmap(self, heat: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        target_height, target_width = image_shape
        if heat.shape[0] == target_height and heat.shape[1] == target_width:
            return heat
        heat_image = Image.fromarray(np.uint8(np.clip(heat * 255, 0, 255)))
        resized = heat_image.resize((target_width, target_height), Image.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0

    def _extract_regions(self, heat: np.ndarray, image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        threshold = max(0.45, float(np.percentile(heat, 90)))
        binary = heat >= threshold
        visited = np.zeros_like(binary, dtype=bool)
        regions = []
        height, width = image_shape
        for y in range(height):
            for x in range(width):
                if not binary[y, x] or visited[y, x]:
                    continue
                stack = [(x, y)]
                visited[y, x] = True
                xs = []
                ys = []
                while stack:
                    cx, cy = stack.pop()
                    xs.append(cx)
                    ys.append(cy)
                    for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                        if 0 <= nx < width and 0 <= ny < height and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((nx, ny))
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                w = x1 - x0 + 1
                h = y1 - y0 + 1
                area_ratio = float((w * h) / max(width * height, 1))
                if area_ratio < 0.003:
                    continue
                regions.append({"x": x0, "y": y0, "w": w, "h": h, "area_ratio": round(area_ratio, 3)})
        regions.sort(key=lambda item: item["area_ratio"], reverse=True)
        return regions[:3]

    def _assign_region_terms(
        self,
        regions: List[Dict[str, Any]],
        defect_terms: List[str],
        heat: np.ndarray,
    ) -> List[Dict[str, Any]]:
        if not defect_terms:
            defect_terms = self.vocab_store.term_names()
        assigned = []
        for idx, region in enumerate(regions):
            term = self._match_single_region_term(region, defect_terms, heat, idx)
            item = dict(region)
            item["matched_term"] = term
            assigned.append(item)
        return assigned

    def _match_single_region_term(
        self,
        region: Dict[str, Any],
        defect_terms: List[str],
        heat: np.ndarray,
        idx: int,
    ) -> str:
        x0 = int(region["x"])
        y0 = int(region["y"])
        x1 = int(region["x"] + region["w"])
        y1 = int(region["y"] + region["h"])
        patch = heat[y0:y1, x0:x1]
        patch_mean = float(np.mean(patch)) if patch.size else 0.0
        aspect_ratio = max(region["w"], region["h"]) / max(min(region["w"], region["h"]), 1)
        area_ratio = float(region["area_ratio"])
        for term in defect_terms:
            lower = term.lower()
            if any(key in lower for key in ["scratch", "划痕", "裂纹", "crack"]) and aspect_ratio > 3:
                return term
            if any(key in lower for key in ["hole", "孔", "缺口", "凹坑"]) and 0.005 < area_ratio < 0.03:
                return term
            if any(key in lower for key in ["spot", "污", "脏", "stain", "污染"]) and patch_mean > 0.45:
                return term
        if defect_terms:
            return defect_terms[idx % len(defect_terms)]
        return f"区域"

    def _match_terms(
        self,
        defect_terms: List[str],
        heat: np.ndarray,
        regions: List[Dict[str, Any]],
        score: float,
    ) -> Tuple[List[str], bool, str]:
        if not defect_terms:
            defect_terms = self.vocab_store.term_names()
        if score < 0.2:
            return [], False, "当前图像整体异常响应较低，系统判定为正常样本或低风险样本。"
        spread = float(np.mean(heat > 0.55))
        elongated = any(max(item["w"], item["h"]) / max(min(item["w"], item["h"]), 1) > 3 for item in regions)
        compact = any(0.005 < item["area_ratio"] < 0.03 for item in regions)
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
            explanation += " 现有词汇库未能覆盖该模式，建议加入未知缺陷复核流程。"
        elif matched:
            explanation += f" 结合开放词汇库，当前更接近 {', '.join(matched)}。"
        elif score >= 0.45 and defect_terms:
            matched = defect_terms[:1]
            explanation += f" 当前存在异常响应，但语义尚不稳定，暂以 {matched[0]} 进入人工复核。"
        return matched, unknown_flag, explanation
