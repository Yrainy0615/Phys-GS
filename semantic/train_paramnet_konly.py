import argparse
import json
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from material_param_dataset import MaterialDatasetConfig, create_dataloader


def load_num_material_classes(mapping_path: str) -> int:
    with open(mapping_path, "r") as f:
        raw = json.load(f)

    if "case_to_material" in raw:
        case_to_material = raw["case_to_material"]
        class_to_id = raw.get("class_to_id", {})
    else:
        case_to_material = raw
        class_to_id = {}

    example_value = next(iter(case_to_material.values()))
    if isinstance(example_value, str):
        if class_to_id:
            return max(int(v) for v in class_to_id.values()) + 1
        return len(set(case_to_material.values()))
    return max(int(v) for v in case_to_material.values()) + 1


class MaterialEmbedding(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int):
        super().__init__()
        self.table = nn.Embedding(num_embeddings=num_classes, embedding_dim=emb_dim)

    def forward(self, material_id: torch.Tensor, num_edges: int) -> torch.Tensor:
        emb = self.table(material_id.view(1)).view(1, -1)
        return emb.expand(num_edges, -1)


class ParamNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        dims = [input_dim] + hidden_dims + [1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


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
    parser.add_argument("--case_to_material", type=str, default="semantic/case_to_material_different_types.json")
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--use_knn_topology", action="store_true")
    parser.add_argument("--object_knn", type=int, default=20)
    parser.add_argument("--object_radius", type=float, default=0.02)
    parser.add_argument("--object_max_neighbours", type=int, default=30)

    parser.add_argument("--emb_dim", type=int, default=16)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 128])
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    num_material_classes = load_num_material_classes(args.case_to_material)

    dataset_cfg = MaterialDatasetConfig(
        base_path=args.base_path,
        sem_cache_dir=args.sem_cache_dir,
        experiments_dir=args.experiments_dir,
        experiments_optimization_dir=args.experiments_optimization_dir,
        case_to_material_path=args.case_to_material,
        use_knn_topology=args.use_knn_topology,
        object_knn=args.object_knn,
        object_radius=args.object_radius,
        object_max_neighbours=args.object_max_neighbours,
    )
    dataset, loader = create_dataloader(
        cfg=dataset_cfg,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    z_geo_dim = dataset[0]["z_geo"].shape[1]
    z_sem_dim = dataset[0]["z_sem"].shape[1]
    input_dim = args.emb_dim + z_geo_dim + z_sem_dim

    mat_emb = MaterialEmbedding(num_classes=num_material_classes, emb_dim=args.emb_dim).to(
        device
    )
    param_net = ParamNet(input_dim=input_dim, hidden_dims=args.hidden_dims).to(device)
    optimizer = torch.optim.Adam(
        list(mat_emb.parameters()) + list(param_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        mat_emb.train()
        param_net.train()
        epoch_loss = 0.0
        epoch_graphs = 0

        for batch in loader:
            optimizer.zero_grad()
            graph_losses = []

            for i in range(len(batch["case_name"])):
                z_geo = batch["z_geo"][i].to(device)
                z_sem = batch["z_sem"][i].to(device)
                teacher_logk = batch["teacher_logk"][i].to(device)
                material_id = batch["material_id"][i].to(device)

                e_class = mat_emb(material_id, z_geo.shape[0])
                edge_cond = torch.cat([e_class, z_geo, z_sem], dim=1)
                pred_logk = param_net(edge_cond)

                loss_k = F.l1_loss(pred_logk, teacher_logk)
                graph_losses.append(loss_k)

            if len(graph_losses) == 0:
                continue

            loss = torch.stack(graph_losses).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(graph_losses)
            epoch_graphs += len(graph_losses)

        mean_loss = epoch_loss / max(epoch_graphs, 1)
        print(f"[epoch {epoch:04d}] loss_k={mean_loss:.6f}")

        ckpt = {
            "epoch": epoch,
            "material_embedding": mat_emb.state_dict(),
            "param_net": param_net.state_dict(),
            "num_material_classes": num_material_classes,
            "emb_dim": args.emb_dim,
            "hidden_dims": args.hidden_dims,
            "z_geo_dim": z_geo_dim,
            "z_sem_dim": z_sem_dim,
            "input_dim": input_dim,
        }
        torch.save(
            ckpt,
            os.path.join(args.save_dir, f"paramnet_konly_epoch_{epoch:04d}.pth"),
        )


if __name__ == "__main__":
    main()
