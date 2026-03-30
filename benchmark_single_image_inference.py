from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
from PIL import Image

RUNTIME_IMPORT_ERROR = None

try:
    import AnomalyCLIP_lib
    import torch
    from scipy.ndimage import gaussian_filter
    from HSF import HybridSemanticFusion
    from adaptive_fusion import AdaptiveFeatureFusion
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform
    from visual_prompt import VisualPromptAligner
except ModuleNotFoundError as exc:
    AnomalyCLIP_lib = None
    torch = None
    HybridSemanticFusion = None
    AdaptiveFeatureFusion = None
    AnomalyCLIP_PromptLearner = None
    get_transform = None
    VisualPromptAligner = None
    RUNTIME_IMPORT_ERROR = exc


def setup_seed(seed: int) -> None:
    if torch is None:
        raise RuntimeError("Missing runtime dependency: torch")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _synchronize(device: torch.device) -> None:
    if torch is None:
        raise RuntimeError("Missing runtime dependency: torch")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _compute_stats(samples: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(samples), 3),
        "std_ms": round(statistics.pstdev(samples), 3) if len(samples) > 1 else 0.0,
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
        "median_ms": round(statistics.median(samples), 3),
    }


class OpenVocabRunner:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.device = device
        self.gaussian_sigma = args.sigma
        self.feature_map_layer = args.feature_map_layer
        self.features_list = args.features_list
        self.image_size = args.image_size

        model_args = {
            "Prompt_length": args.n_ctx,
            "learnabel_text_embedding_depth": args.depth,
            "learnabel_text_embedding_length": args.t_n_ctx,
        }
        self.model, _ = AnomalyCLIP_lib.load(args.backbone, device=str(device), design_details=model_args)
        self.model.eval()

        self.prompt_learner = AnomalyCLIP_PromptLearner(self.model.to("cpu"), model_args)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        self.prompt_learner.to(device)
        self.model.to(device)
        self.model.visual.DAPM_replace(DPAM_layer=20)

        prompts, tokenized_prompts, compound_prompts_text, _, _ = self.prompt_learner(cls_id=None)
        text_features = self.model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        preprocess_args = SimpleNamespace(image_size=args.image_size)
        self.preprocess, _ = get_transform(preprocess_args)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).reshape(1, 3, self.image_size, self.image_size)
        return tensor.to(self.device)

    def forward(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            image_features, patch_features = self.model.encode_image(image_tensor, self.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_probs = image_features @ self.text_features.permute(0, 2, 1)
            text_probs = (text_probs / 0.07).softmax(-1)
            image_score = float(text_probs[:, 0, 1].item())

            anomaly_map_list = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= self.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, self.text_features[0])
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], self.image_size)
                    anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                    anomaly_map_list.append(anomaly_map)

            stacked = torch.stack(anomaly_map_list).sum(dim=0)
            smoothed = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=self.gaussian_sigma)) for i in stacked.cpu()],
                dim=0,
            )
        return {"score": image_score, "anomaly_map_shape": list(smoothed[0].shape)}


class VisualPromptGateRunner:
    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.device = device
        self.gaussian_sigma = args.sigma
        self.feature_map_layer = args.feature_map_layer
        self.features_list = args.features_list
        self.image_size = args.image_size

        model_args = {
            "Prompt_length": args.n_ctx,
            "learnabel_text_embedding_depth": args.depth,
            "learnabel_text_embedding_length": args.t_n_ctx,
        }
        self.model, _ = AnomalyCLIP_lib.load(args.backbone, device=str(device), design_details=model_args)
        self.model.eval()

        self.prompt_learner = AnomalyCLIP_PromptLearner(self.model.to("cpu"), model_args)
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        self.prompt_learner.load_state_dict(checkpoint["prompt_learner"])
        self.prompt_learner.to(device)
        self.model.to(device)
        self.model.visual.DAPM_replace(DPAM_layer=20)

        embed_dim = self.model.text_projection.shape[1]
        self.visual_prompt = VisualPromptAligner(
            embed_dim=embed_dim,
            hidden_dim=args.visual_prompt_hidden_dim,
            dropout=args.visual_prompt_dropout,
        ).to(device)
        self.visual_prompt.load_state_dict(checkpoint["visual_prompt"])
        self.visual_prompt.eval()

        self.adaptive_fusion = AdaptiveFeatureFusion(
            embed_dim=embed_dim,
            hidden_dim=args.gate_hidden_dim,
            dropout=args.gate_dropout,
        ).to(device)
        self.adaptive_fusion.load_state_dict(checkpoint["adaptive_fusion"])
        self.adaptive_fusion.eval()

        self.hsf = HybridSemanticFusion(20)
        prompts, tokenized_prompts, compound_prompts_text, _, _ = self.prompt_learner(cls_id=None)
        text_features = self.model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
        text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        preprocess_args = SimpleNamespace(image_size=args.image_size)
        self.preprocess, _ = get_transform(preprocess_args)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(image).reshape(1, 3, self.image_size, self.image_size)
        return tensor.to(self.device)

    def forward(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            image_features, patch_features = self.model.encode_image(image_tensor, self.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            aligned_visual_features, _, _ = self.visual_prompt(image_features, self.text_features)
            gated_visual_features, _ = self.adaptive_fusion.fuse_visual_prompt(image_features, aligned_visual_features)

            anomaly_map_list = []
            anomaly_maps = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= self.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, anomaly_map = AnomalyCLIP_lib.compute_similarity(patch_feature, self.text_features[0])
                    anomaly_maps.append(anomaly_map)
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], self.image_size)
                    anomaly_map_list.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)

            clustered_feature = self.hsf.forward(patch_features, anomaly_maps)
            final_image_feature, _ = self.adaptive_fusion.fuse_cluster_feature(gated_visual_features, clustered_feature)

            text_probs = final_image_feature.unsqueeze(1) @ self.text_features.permute(0, 2, 1)
            text_probs = torch.softmax(100.0 * text_probs.squeeze(1), dim=-1)
            image_score = float(text_probs[:, 1].item())

            anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
            anomaly_map = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=self.gaussian_sigma)) for i in anomaly_map.detach().cpu()],
                dim=0,
            )
        return {"score": image_score, "anomaly_map_shape": list(anomaly_map[0].shape)}


def build_runner(args: argparse.Namespace, device: torch.device):
    if RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Benchmark runtime dependencies are missing. "
            "Please install the project inference environment first."
        ) from RUNTIME_IMPORT_ERROR
    if args.framework == "open_vocab":
        return OpenVocabRunner(args, device)
    if args.framework == "vp_gated":
        return VisualPromptGateRunner(args, device)
    raise ValueError("Unsupported framework: {0}".format(args.framework))


def resolve_device(device_arg: str) -> torch.device:
    if torch is None:
        raise RuntimeError("Missing runtime dependency: torch")
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but a CUDA device was requested.")
    if device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available, but an MPS device was requested.")
    return device


def benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    setup_seed(args.seed)
    device = resolve_device(args.device)

    load_start = time.perf_counter()
    runner = build_runner(args, device)
    _synchronize(device)
    load_ms = (time.perf_counter() - load_start) * 1000.0

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    image_path = str(Path(args.image_path).expanduser().resolve())
    warmup_tensor = runner.preprocess_image(image_path)
    for _ in range(args.warmup):
        _ = runner.forward(warmup_tensor)
        _synchronize(device)

    preprocess_times = []
    inference_times = []
    end_to_end_times = []
    last_output: Dict[str, Any] = {}

    for _ in range(args.repeat):
        preprocess_start = time.perf_counter()
        image_tensor = runner.preprocess_image(image_path)
        _synchronize(device)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        inference_start = time.perf_counter()
        last_output = runner.forward(image_tensor)
        _synchronize(device)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        preprocess_times.append(preprocess_ms)
        inference_times.append(inference_ms)
        end_to_end_times.append(preprocess_ms + inference_ms)

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0), 2)

    result = {
        "framework": args.framework,
        "backbone": args.backbone,
        "checkpoint_path": str(Path(args.checkpoint_path).expanduser().resolve()),
        "image_path": image_path,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "Apple Silicon MPS"
            if device.type == "mps"
            else "CPU"
        ),
        "image_size": args.image_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "model_load_ms": round(load_ms, 3),
        "preprocess_stats": _compute_stats(preprocess_times),
        "inference_stats": _compute_stats(inference_times),
        "end_to_end_stats": _compute_stats(end_to_end_times),
        "peak_memory_mb": peak_memory_mb,
        "last_score": round(float(last_output.get("score", 0.0)), 6),
        "anomaly_map_shape": last_output.get("anomaly_map_shape"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Single image inference benchmark", add_help=True)
    parser.add_argument("--image_path", type=str, required=True, help="path to the input image")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to the checkpoint")
    parser.add_argument(
        "--framework",
        type=str,
        default="open_vocab",
        choices=["open_vocab", "vp_gated"],
        help="inference framework",
    )
    parser.add_argument("--backbone", type=str, default="ViT-L/14@336px", help="CLIP backbone")
    parser.add_argument("--device", type=str, default="auto", help="device, e.g. auto, cuda:0 or cpu")
    parser.add_argument("--image_size", type=int, default=518, help="input image size")
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup runs")
    parser.add_argument("--repeat", type=int, default=50, help="number of measured runs")
    parser.add_argument("--depth", type=int, default=9, help="prompt depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="prompt length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="learnable text embedding length")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian sigma")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="feature layers")
    parser.add_argument(
        "--feature_map_layer",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="layers used for anomaly map aggregation",
    )
    parser.add_argument("--visual_prompt_hidden_dim", type=int, default=0, help="hidden dim of visual prompt mlp")
    parser.add_argument("--visual_prompt_dropout", type=float, default=0.1, help="dropout in visual prompt mlp")
    parser.add_argument("--gate_hidden_dim", type=int, default=0, help="hidden dim of adaptive gate")
    parser.add_argument("--gate_dropout", type=float, default=0.1, help="dropout in adaptive gate")
    parser.add_argument("--output_json", type=str, default="", help="optional path to save benchmark results as json")
    args = parser.parse_args()

    result = benchmark(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
