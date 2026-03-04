import argparse
import glob
import json
import os
import pickle
import random
import sys
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warp as wp
from tqdm import tqdm

from material_param_dataset import MaterialDatasetConfig, create_dataloader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg


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


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_case_cfg(base_path: str, case_name: str, experiments_optimization_dir: str) -> None:
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    optimal_path = os.path.join(experiments_optimization_dir, case_name, "optimal_params.pkl")
    if os.path.exists(optimal_path):
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

    with open(os.path.join(base_path, case_name, "metadata.json"), "r") as f:
        data = json.load(f)
    with open(os.path.join(base_path, case_name, "calibrate.pkl"), "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c2w) for c2w in c2ws])
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = os.path.join(base_path, case_name, "color")


class CaseRuntime:
    def __init__(
        self,
        base_path: str,
        case_name: str,
        experiments_optimization_dir: str,
        train_frame: int,
        device: str,
    ):
        _load_case_cfg(base_path, case_name, experiments_optimization_dir)
        self.case_name = case_name
        self.case_dir = os.path.join(base_path, case_name)
        self.train_frame = int(train_frame)
        self.device = device
        self.mask_cache: Dict[int, torch.Tensor] = {}

        self.trainer = InvPhyTrainerWarp(
            data_path=os.path.join(base_path, case_name, "final_data.pkl"),
            base_dir=os.path.join("semantic", "runtime", case_name),
            train_frame=self.train_frame,
            pure_inference_mode=True,
            device=device,
        )
        self.sim = self.trainer.simulator
        self.num_object_springs = self.trainer.num_object_springs
        self.num_original_points = self.trainer.num_original_points

    def load_union_mask_cam0(self, frame_idx: int, width: int, height: int) -> torch.Tensor:
        if frame_idx in self.mask_cache:
            return self.mask_cache[frame_idx]

        mask_dir = os.path.join(self.case_dir, "mask", "0")
        frame_name = f"{frame_idx}.png"
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*", frame_name)))
        if not mask_paths:
            mask = torch.zeros((1, 1, height, width), dtype=torch.float32, device=self.device)
            self.mask_cache[frame_idx] = mask
            return mask

        union_mask = None
        for mask_path in mask_paths:
            img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
            cur = (img > 0).astype(np.float32)
            union_mask = cur if union_mask is None else np.maximum(union_mask, cur)

        if union_mask is None:
            union_mask = np.zeros((height, width), dtype=np.float32)
        mask = torch.from_numpy(union_mask)[None, None].to(self.device)
        self.mask_cache[frame_idx] = mask
        return mask


def point_mask_render_loss(
    points_world: torch.Tensor,
    mask: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    n = points_world.shape[0]
    ones = torch.ones((n, 1), dtype=points_world.dtype, device=points_world.device)
    pts_h = torch.cat([points_world, ones], dim=1)
    cam = pts_h @ w2c.t()
    z = cam[:, 2].clamp(min=1e-6)
    u = K[0, 0] * (cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (cam[:, 1] / z) + K[1, 2]

    u_norm = (u / max(width - 1, 1)) * 2.0 - 1.0
    v_norm = (v / max(height - 1, 1)) * 2.0 - 1.0
    valid = (z > 1e-6) & (u_norm >= -1.0) & (u_norm <= 1.0) & (v_norm >= -1.0) & (v_norm <= 1.0)
    if valid.sum().item() == 0:
        return torch.zeros((), device=points_world.device, dtype=points_world.dtype)

    grid = torch.stack([u_norm, v_norm], dim=1)[None, :, None, :]
    sampled = F.grid_sample(
        mask,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0, :, 0]
    return (1.0 - sampled[valid]).abs().mean()


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
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train_mode",
        type=str,
        default="observation",
        choices=["teacher", "observation"],
    )

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
    parser.add_argument("--lambda_render", type=float, default=1.0)
    parser.add_argument("--lambda_track", type=float, default=1.0)
    parser.add_argument("--lambda_geo", type=float, default=1.0)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join("semantic", "runtime"), exist_ok=True)
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

    runtimes: Dict[str, CaseRuntime] = {}

    for epoch in range(args.epochs):
        mat_emb.train()
        param_net.train()
        epoch_loss = 0.0
        epoch_render = 0.0
        epoch_track = 0.0
        epoch_geo = 0.0
        epoch_graphs = 0

        if args.train_mode == "teacher":
            epoch_total_steps = max(len(dataset), 1)
        else:
            epoch_total_steps = 0
            for sample in dataset:
                train_frame_val = int(sample["train_frame"].item())
                epoch_total_steps += max(train_frame_val - 1, 1)
            epoch_total_steps = max(epoch_total_steps, 1)
        pbar = tqdm(
            total=epoch_total_steps,
            desc=f"epoch {epoch:04d}",
            dynamic_ncols=True,
        )

        for batch in loader:
            for i in range(len(batch["case_name"])):
                z_geo = batch["z_geo"][i].to(device)
                z_sem = batch["z_sem"][i].to(device)
                material_id = batch["material_id"][i].to(device)

                e_class = mat_emb(material_id, z_geo.shape[0])
                edge_cond = torch.cat([e_class, z_geo, z_sem], dim=1)
                pred_logk = param_net(edge_cond)

                if args.train_mode == "teacher":
                    teacher_logk = batch["teacher_logk"][i].to(device)
                    optimizer.zero_grad()
                    loss_k = F.l1_loss(pred_logk, teacher_logk)
                    loss_k.backward()
                    optimizer.step()
                    epoch_loss += float(loss_k.item())
                    epoch_graphs += 1
                    pbar.update(1)
                    pbar.set_postfix(loss_k=f"{loss_k.item():.4f}")
                    continue

                case_name = batch["case_name"][i]
                if case_name not in runtimes:
                    train_frame_init = int(batch["train_frame"][i].item())
                    print(f"[runtime:init] case={case_name} train_frame={train_frame_init}")
                    runtimes[case_name] = CaseRuntime(
                        base_path=args.base_path,
                        case_name=case_name,
                        experiments_optimization_dir=args.experiments_optimization_dir,
                        train_frame=train_frame_init,
                        device=args.device,
                    )
                runtime = runtimes[case_name]
                sim = runtime.sim

                cfg.chamfer_weight = float(args.lambda_geo)
                cfg.track_weight = float(args.lambda_track)
                cfg.acc_weight = 0.0

                base_spring_y = batch["base_spring_y"][i].to(device).view(-1)
                base_logk = torch.log(base_spring_y.clamp_min(1e-8))
                num_object_springs = runtime.num_object_springs
                if pred_logk.shape[0] != num_object_springs:
                    raise ValueError(
                        f"{case_name}: pred springs mismatch {pred_logk.shape[0]} vs {num_object_springs}"
                    )
                full_logk = torch.cat([pred_logk.view(-1), base_logk[num_object_springs:]], dim=0)
                if full_logk.numel() != sim.n_springs:
                    raise ValueError(
                        f"{case_name}: full spring size mismatch {full_logk.numel()} vs {sim.n_springs}"
                    )

                sim.set_spring_Y(full_logk.detach())
                sim.set_collide(
                    batch["collide_elas"][i].to(device).view(-1),
                    batch["collide_fric"][i].to(device).view(-1),
                )
                sim.set_collide_object(
                    batch["collide_object_elas"][i].to(device).view(-1),
                    batch["collide_object_fric"][i].to(device).view(-1),
                )
                sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
                sim.set_acc_count(False)

                train_frame = int(batch["train_frame"][i].item())
                K = batch["cam0_intrinsics"][i].to(device)
                w2c = batch["cam0_w2c"][i].to(device)
                wh = batch["wh"][i].to(device)
                width = int(wh[0].item())
                height = int(wh[1].item())

                grad_accum = torch.zeros_like(pred_logk)
                graph_geo = 0.0
                graph_track = 0.0
                graph_render = 0.0
                steps = 0

                for frame_idx in range(1, train_frame):
                    sim.set_controller_target(frame_idx)
                    if sim.object_collision_flag:
                        sim.update_collision_graph()

                    with sim.tape:
                        sim.step()
                        sim.calculate_loss()
                    sim.tape.backward(sim.loss)

                    grad_full = wp.to_torch(sim.wp_spring_Y.grad, requires_grad=False).detach()
                    grad_accum = grad_accum + grad_full[:num_object_springs].view_as(pred_logk)

                    track_val = wp.to_torch(sim.track_loss, requires_grad=False).item()
                    geo_val = wp.to_torch(sim.chamfer_loss, requires_grad=False).item()
                    graph_track += track_val
                    graph_geo += geo_val

                    pred_points = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False)
                    pred_points = pred_points[: runtime.num_original_points]
                    gt_mask = runtime.load_union_mask_cam0(frame_idx, width=width, height=height)
                    render_val = point_mask_render_loss(pred_points, gt_mask, K, w2c, width, height)
                    graph_render += float(render_val.item())

                    sim.tape.reset()
                    sim.clear_loss()
                    sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)
                    steps += 1
                    pbar.update(1)

                if steps == 0:
                    pbar.update(1)
                    continue

                mean_track = graph_track / steps
                mean_geo = graph_geo / steps
                mean_render = graph_render / steps

                # Render term is non-differentiable in this current bridge; use it as a detached scale.
                render_scale = 1.0 + float(args.lambda_render) * min(mean_render, 10.0)
                optimizer.zero_grad()
                (pred_logk * 0.0 + 1.0).backward((grad_accum / float(steps)) * render_scale)
                optimizer.step()

                total = (
                    float(args.lambda_track) * mean_track
                    + float(args.lambda_geo) * mean_geo
                    + float(args.lambda_render) * mean_render
                )
                epoch_loss += total
                epoch_track += mean_track
                epoch_geo += mean_geo
                epoch_render += mean_render
                epoch_graphs += 1
                pbar.set_postfix(
                    total=f"{total:.4f}",
                    track=f"{mean_track:.4f}",
                    geo=f"{mean_geo:.4f}",
                    render=f"{mean_render:.4f}",
                )

        pbar.close()

        mean_loss = epoch_loss / max(epoch_graphs, 1)
        if args.train_mode == "teacher":
            print(f"[epoch {epoch:04d}] loss_k={mean_loss:.6f}")
        else:
            mean_track = epoch_track / max(epoch_graphs, 1)
            mean_geo = epoch_geo / max(epoch_graphs, 1)
            mean_render = epoch_render / max(epoch_graphs, 1)
            print(
                f"[epoch {epoch:04d}] total={mean_loss:.6f} "
                f"track={mean_track:.6f} geo={mean_geo:.6f} render={mean_render:.6f}"
            )

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
        ckpt["train_mode"] = args.train_mode
        ckpt["lambda_render"] = args.lambda_render
        ckpt["lambda_track"] = args.lambda_track
        ckpt["lambda_geo"] = args.lambda_geo
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            torch.save(
                ckpt,
                os.path.join(args.save_dir, f"paramnet_render_latest.pth"),
            )


if __name__ == "__main__":
    main()
