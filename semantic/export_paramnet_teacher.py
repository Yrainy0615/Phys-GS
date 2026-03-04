import argparse
import glob
import os

import torch

from material_param_dataset import MaterialDatasetConfig, MaterialParamDataset
from train_paramnet_konly import MaterialEmbedding, ParamNet


def _pick_best_ckpt(train_dir: str) -> str:
    best_list = sorted(glob.glob(os.path.join(train_dir, "best_*.pth")))
    if not best_list:
        raise FileNotFoundError(f"No best_*.pth found in {train_dir}")
    return best_list[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="data/different_types")
    parser.add_argument("--sem_cache_dir", type=str, default="semantic/cache")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument(
        "--experiments_optimization_dir",
        type=str,
        default="experiments_optimization",
    )
    parser.add_argument(
        "--case_to_material",
        type=str,
        default="semantic/case_to_material_different_types.json",
    )
    parser.add_argument(
        "--paramnet_ckpt",
        type=str,
        required=True,
        help="Path to pretrained paramnet_konly checkpoint",
    )
    parser.add_argument(
        "--out_experiments_dir",
        type=str,
        default="experiments_k_teacher",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_experiments_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset_cfg = MaterialDatasetConfig(
        base_path=args.base_path,
        sem_cache_dir=args.sem_cache_dir,
        experiments_dir=args.experiments_dir,
        experiments_optimization_dir=args.experiments_optimization_dir,
        case_to_material_path=args.case_to_material,
    )
    dataset = MaterialParamDataset(dataset_cfg)

    ckpt = torch.load(args.paramnet_ckpt, map_location="cpu")
    mat_emb = MaterialEmbedding(
        num_classes=int(ckpt["num_material_classes"]),
        emb_dim=int(ckpt["emb_dim"]),
    ).to(device)
    param_net = ParamNet(
        input_dim=int(ckpt["input_dim"]),
        hidden_dims=list(ckpt["hidden_dims"]),
    ).to(device)
    mat_emb.load_state_dict(ckpt["material_embedding"])
    param_net.load_state_dict(ckpt["param_net"])
    mat_emb.eval()
    param_net.eval()

    with torch.no_grad():
        for sample in dataset:
            case_name = sample["case_name"]
            z_geo = sample["z_geo"].to(device)
            z_sem = sample["z_sem"].to(device)
            material_id = sample["material_id"].to(device)
            num_object_springs = z_geo.shape[0]

            e_class = mat_emb(material_id, num_object_springs)
            edge_cond = torch.cat([e_class, z_geo, z_sem], dim=1)
            pred_logk = param_net(edge_cond)
            pred_k_object = torch.exp(pred_logk).view(-1).cpu()

            src_best = _pick_best_ckpt(os.path.join(args.experiments_dir, case_name, "train"))
            src_ckpt = torch.load(src_best, map_location="cpu")
            src_spring = src_ckpt["spring_Y"].float().view(-1)
            src_num_object = int(src_ckpt.get("num_object_springs", pred_k_object.numel()))

            if src_num_object != pred_k_object.numel():
                raise ValueError(
                    f"{case_name}: predicted object springs ({pred_k_object.numel()}) "
                    f"!= source checkpoint object springs ({src_num_object})"
                )

            out_spring = src_spring.clone()
            out_spring[:src_num_object] = pred_k_object

            out_ckpt = dict(src_ckpt)
            out_ckpt["epoch"] = int(ckpt["epoch"])
            out_ckpt["num_object_springs"] = src_num_object
            out_ckpt["spring_Y"] = out_spring
            out_ckpt["source_teacher_ckpt"] = src_best
            out_ckpt["source_paramnet_ckpt"] = args.paramnet_ckpt

            out_train_dir = os.path.join(args.out_experiments_dir, case_name, "train")
            os.makedirs(out_train_dir, exist_ok=True)
            out_path = os.path.join(out_train_dir, os.path.basename(src_best))
            torch.save(out_ckpt, out_path)
            print(
                f"[saved] {out_path} | object_springs={src_num_object} "
                f"all_springs={out_spring.numel()}"
            )


if __name__ == "__main__":
    main()
