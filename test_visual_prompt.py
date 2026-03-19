import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F

from HSF import HybridSemanticFusion
from prompt_ensemble import AnomalyCLIP_PromptLearner
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
from visual_prompt import VisualPromptAligner

import random
import numpy as np
from tabulate import tabulate
from utils import get_transform
from metrics import image_level_metrics, pixel_level_metrics
from scipy.ndimage import gaussian_filter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def test(args):
    logger = get_logger(args.save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    anomalyclip_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=anomalyclip_parameters)
    model.eval()

    preprocess, target_transform = get_transform(args)
    test_data = Dataset(
        root=args.data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
    )
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.obj_list

    results = {}
    for obj in obj_list:
        results[obj] = {
            "gt_sp": [],
            "pr_sp": [],
            "imgs_masks": [],
            "anomaly_maps": [],
        }

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), anomalyclip_parameters)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    visual_prompt = VisualPromptAligner(
        embed_dim=model.text_projection.shape[1],
        hidden_dim=args.visual_prompt_hidden_dim,
        dropout=args.visual_prompt_dropout,
    ).to(device)
    visual_prompt.load_state_dict(checkpoint["visual_prompt"])
    visual_prompt.eval()

    hsf = HybridSemanticFusion(20)

    prompts, tokenized_prompts, compound_prompts_text, _, _ = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for items in tqdm(test_dataloader):
        image = items["img"].to(device)
        cls_name = items["cls_name"]
        gt_mask = items["img_mask"]
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results[cls_name[0]]["imgs_masks"].append(gt_mask)
        results[cls_name[0]]["gt_sp"].extend(items["anomaly"].detach().cpu())

        with torch.no_grad():
            image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            enhanced_image_features, _, _ = visual_prompt(image_features, text_features)

            anomaly_map_list = []
            anomaly_maps = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, anomaly_map = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    anomaly_maps.append(anomaly_map)
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                    anomaly_map_list.append((similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0)

            clustered_feature = hsf.forward(patch_features, anomaly_maps)
            cur_image_feature = args.cluster_alpha * clustered_feature + (1 - args.cluster_alpha) * enhanced_image_features
            cur_image_feature = F.normalize(cur_image_feature, dim=1)

            text_probs = cur_image_feature.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = torch.softmax(100.0 * text_probs.squeeze(1), dim=-1)
            text_probs = text_probs[:, 1]

            anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
            anomaly_map = torch.stack(
                [torch.from_numpy(gaussian_filter(i, sigma=args.sigma)) for i in anomaly_map.detach().cpu()],
                dim=0,
            )

            results[cls_name[0]]["pr_sp"].extend(text_probs.detach().cpu())
            results[cls_name[0]]["anomaly_maps"].append(anomaly_map)

    table_ls = []
    image_auroc_list = []
    image_ap_list = []
    pixel_auroc_list = []
    pixel_aupro_list = []
    for obj in obj_list:
        table = [obj]
        results[obj]["imgs_masks"] = torch.cat(results[obj]["imgs_masks"])
        results[obj]["anomaly_maps"] = torch.cat(results[obj]["anomaly_maps"]).detach().cpu().numpy()
        if args.metrics == "image-level":
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.extend([str(np.round(image_auroc * 100, 1)), str(np.round(image_ap * 100, 1))])
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        elif args.metrics == "pixel-level":
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            table.extend([str(np.round(pixel_auroc * 100, 1)), str(np.round(pixel_aupro * 100, 1))])
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        else:
            pixel_auroc = pixel_level_metrics(results, obj, "pixel-auroc")
            pixel_aupro = pixel_level_metrics(results, obj, "pixel-aupro")
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.extend([
                str(np.round(pixel_auroc * 100, 1)),
                str(np.round(pixel_aupro * 100, 1)),
                str(np.round(image_auroc * 100, 1)),
                str(np.round(image_ap * 100, 1)),
            ])
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
            pixel_auroc_list.append(pixel_auroc)
            pixel_aupro_list.append(pixel_aupro)
        table_ls.append(table)

    if args.metrics == "image-level":
        table_ls.append(["mean", str(np.round(np.mean(image_auroc_list) * 100, 1)), str(np.round(np.mean(image_ap_list) * 100, 1))])
        table = tabulate(table_ls, headers=["objects", "image_auroc", "image_ap"], tablefmt="pipe")
    elif args.metrics == "pixel-level":
        table_ls.append(["mean", str(np.round(np.mean(pixel_auroc_list) * 100, 1)), str(np.round(np.mean(pixel_aupro_list) * 100, 1))])
        table = tabulate(table_ls, headers=["objects", "pixel_auroc", "pixel_aupro"], tablefmt="pipe")
    else:
        table_ls.append([
            "mean",
            str(np.round(np.mean(pixel_auroc_list) * 100, 1)),
            str(np.round(np.mean(pixel_aupro_list) * 100, 1)),
            str(np.round(np.mean(image_auroc_list) * 100, 1)),
            str(np.round(np.mean(image_ap_list) * 100, 1)),
        ])
        table = tabulate(table_ls, headers=["objects", "pixel_auroc", "pixel_aupro", "image_auroc", "image_ap"], tablefmt="pipe")
    logger.info("\n%s", table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnomalyCLIP-VisualPrompt-Test", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data/DAGM", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default="./results/", help="path to save results")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to checkpoint")
    parser.add_argument("--dataset", type=str, default="DAGM")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="prompt depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="prompt length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="learnable text embedding length")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3], help="feature map layer")
    parser.add_argument("--metrics", type=str, default="image-pixel-level")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4, help="gaussian sigma")
    parser.add_argument("--visual_prompt_hidden_dim", type=int, default=0, help="hidden dim of visual prompt mlp")
    parser.add_argument("--visual_prompt_dropout", type=float, default=0.1, help="dropout in visual prompt mlp")
    parser.add_argument("--cluster_alpha", type=float, default=0.2, help="weight for clustered feature fusion")
    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
