import AnomalyCLIP_lib
import torch
import argparse
import torch.nn.functional as F

from HSF import HybridSemanticFusion
from prompt_ensemble import AnomalyCLIP_PromptLearner
from loss import FocalLoss, BinaryDiceLoss, KD_Loss
from dataset import Dataset
from logger import get_logger
from tqdm import tqdm
from visual_prompt import VisualPromptAligner
from adaptive_fusion import AdaptiveFeatureFusion
import numpy as np
import os

import random
from utils import get_transform
import warnings

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
    logger = get_logger(args.save_path)
    preprocess, target_transform = get_transform(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    anomalyclip_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx
    }

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=anomalyclip_parameters)
    model.eval()

    hsf = HybridSemanticFusion(20)
    train_data = Dataset(
        root=args.train_data_path,
        transform=preprocess,
        target_transform=target_transform,
        dataset_name=args.dataset,
    )
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), anomalyclip_parameters)
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    embed_dim = model.text_projection.shape[1]
    visual_prompt = VisualPromptAligner(
        embed_dim=embed_dim,
        hidden_dim=args.visual_prompt_hidden_dim,
        dropout=args.visual_prompt_dropout,
    ).to(device)
    adaptive_fusion = AdaptiveFeatureFusion(
        embed_dim=embed_dim,
        hidden_dim=args.gate_hidden_dim,
        dropout=args.gate_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(prompt_learner.parameters()) +
        list(visual_prompt.parameters()) +
        list(adaptive_fusion.parameters()),
        lr=args.learning_rate,
        betas=(0.5, 0.999),
    )

    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_kd = KD_Loss()

    model.eval()
    prompt_learner.train()
    visual_prompt.train()
    adaptive_fusion.train()

    for epoch in tqdm(range(args.epoch)):
        model.eval()
        prompt_learner.train()
        visual_prompt.train()
        adaptive_fusion.train()
        loss_list = []
        image_loss_list = []
        vp_loss_list = []
        gate_reg_list = []

        for items in tqdm(train_dataloader):
            image = items["img"].to(device)
            label = items["anomaly"]
            gt = items["img_mask"].squeeze().to(device)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0

            with torch.no_grad():
                image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            prompts, tokenized_prompts, compound_prompts_text, prompts_pos, prompts_neg = prompt_learner(cls_id=None)
            text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
            text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            aligned_visual_features, visual_prompt_tokens, _ = visual_prompt(image_features, text_features)
            gated_visual_features, visual_gate = adaptive_fusion.fuse_visual_prompt(
                image_features, aligned_visual_features
            )

            text_probs = gated_visual_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
            text_probs = text_probs[:, 0, ...] / 0.07
            image_loss = F.cross_entropy(text_probs.squeeze(), label.long().to(device))
            image_loss_list.append(image_loss.item())

            similarity_map_list = []
            anomaly_maps = []
            for idx, patch_feature in enumerate(patch_features):
                if idx >= args.feature_map_layer[0]:
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity, anomaly_map = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                    anomaly_maps.append(anomaly_map)
                    similarity_map = AnomalyCLIP_lib.get_similarity_map(
                        similarity[:, 1:, :],
                        args.image_size
                    ).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)

            clustered_feature = hsf.forward(patch_features, anomaly_maps)
            final_image_feature, cluster_gate = adaptive_fusion.fuse_cluster_feature(
                gated_visual_features, clustered_feature
            )

            anomaly_score = (100.0 * final_image_feature.unsqueeze(1) @ text_features.permute(0, 2, 1))
            anomaly_score = anomaly_score.squeeze(1)
            anomaly_score = torch.softmax(anomaly_score, dim=1)

            loss = 0

            is_anomaly = items["anomaly"].to(device)
            is_anomaly[is_anomaly > 0.5] = 1
            is_anomaly[is_anomaly <= 0.5] = 0

            classification_loss = loss_focal(anomaly_score, is_anomaly.unsqueeze(1))
            loss += classification_loss
            for similarity_map in similarity_map_list:
                loss += loss_focal(similarity_map, gt)
                loss += loss_dice(similarity_map[:, 1, :, :], gt)
                loss += loss_dice(similarity_map[:, 0, :, :], 1 - gt)

            optimizer.zero_grad()
            if "llm_embedding" in items:
                loss += loss_kd(items["llm_embedding"].to(device), prompts_pos, prompts_neg)

            vp_target = visual_prompt_tokens.detach() / visual_prompt_tokens.detach().norm(dim=-1, keepdim=True)
            vp_loss = F.mse_loss(gated_visual_features, vp_target)
            vp_loss_list.append(vp_loss.item())

            # Prevent the two gates from collapsing too early to all-0 or all-1.
            gate_reg = ((visual_gate.mean() - 0.5) ** 2 + (cluster_gate.mean() - 0.5) ** 2)
            gate_reg_list.append(gate_reg.item())

            total_loss = loss + image_loss
            total_loss += args.visual_prompt_loss_weight * vp_loss
            total_loss += args.gate_regularization_weight * gate_reg

            total_loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if (epoch + 1) % args.print_freq == 0:
            logger.info(
                "epoch [{}/{}], loss:{:.4f}, image_loss:{:.4f}, vp_loss:{:.4f}, gate_reg:{:.4f}".format(
                    epoch + 1,
                    args.epoch,
                    np.mean(loss_list),
                    np.mean(image_loss_list),
                    np.mean(vp_loss_list) if vp_loss_list else 0.0,
                    np.mean(gate_reg_list) if gate_reg_list else 0.0,
                )
            )

        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(args.save_path, "epoch_" + str(epoch + 1) + ".pth")
            torch.save(
                {
                    "prompt_learner": prompt_learner.state_dict(),
                    "visual_prompt": visual_prompt.state_dict(),
                    "adaptive_fusion": adaptive_fusion.state_dict(),
                },
                ckp_path,
            )


if __name__ == "__main__":
    print(torch.version.__version__)

    parser = argparse.ArgumentParser("AnomalyCLIP-VisualPrompt-GatedFusion", add_help=True)
    parser.add_argument("--train_data_path", type=str, default="./data/mvtec", help="train dataset path")
    parser.add_argument("--save_path", type=str, default="./checkpoint/noForzen/hsf_visual_prompt_gate", help="path to save results")
    parser.add_argument("--dataset", type=str, default="mvtec", help="train dataset name")
    parser.add_argument("--depth", type=int, default=9, help="the depth of learnable text prompt")
    parser.add_argument("--n_ctx", type=int, default=12, help="the length of learnable text prompt")
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
    parser.add_argument("--visual_prompt_hidden_dim", type=int, default=0, help="hidden dim of visual prompt mlp, 0 means 2x embed dim")
    parser.add_argument("--visual_prompt_dropout", type=float, default=0.1, help="dropout in visual prompt mlp")
    parser.add_argument("--visual_prompt_loss_weight", type=float, default=0.1, help="weight for visual prompt alignment loss")
    parser.add_argument("--gate_hidden_dim", type=int, default=0, help="hidden dim of adaptive gate, 0 means embed dim")
    parser.add_argument("--gate_dropout", type=float, default=0.1, help="dropout in adaptive gate")
    parser.add_argument("--gate_regularization_weight", type=float, default=0.01, help="regularization weight for gate stability")
    args = parser.parse_args()
    setup_seed(args.seed)
    train(args)
