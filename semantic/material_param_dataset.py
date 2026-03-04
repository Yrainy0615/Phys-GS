import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import DataLoader, Dataset


def _load_structure_points(final_data_path: str) -> np.ndarray:
    with open(final_data_path, "rb") as f:
        data = pickle.load(f)
    object_points = data["object_points"][0]
    other_surface_points = data["surface_points"]
    interior_points = data["interior_points"]
    structure_points = np.concatenate(
        [object_points, other_surface_points, interior_points], axis=0
    )
    return structure_points.astype(np.float32)


def _build_object_edges_open3d(
    points: np.ndarray,
    use_knn_topology: bool,
    object_knn: int,
    object_radius: float,
    object_max_neighbours: int,
) -> np.ndarray:
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(object_pcd)

    spring_flags = np.zeros((len(points), len(points)), dtype=np.uint8)
    springs: List[List[int]] = []
    for i in range(len(points)):
        if use_knn_topology:
            _, idx, _ = pcd_tree.search_knn_vector_3d(points[i], object_knn + 1)
        else:
            _, idx, _ = pcd_tree.search_hybrid_vector_3d(
                points[i], object_radius, object_max_neighbours
            )
        idx = idx[1:]
        for j in idx:
            rest_length = np.linalg.norm(points[i] - points[j])
            if (
                spring_flags[i, j] == 0
                and spring_flags[j, i] == 0
                and rest_length > 1e-4
            ):
                spring_flags[i, j] = 1
                spring_flags[j, i] = 1
                springs.append([i, j])
    return np.asarray(springs, dtype=np.int64)


def _geo_features(points: np.ndarray, edges: np.ndarray, density_k: int = 16) -> np.ndarray:
    n = points.shape[0]
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    k = min(density_k, max(n - 1, 1))
    nn = np.partition(d2, kth=k - 1, axis=1)[:, :k]
    local_density = 1.0 / (np.sqrt(nn.mean(axis=1)) + 1e-8)

    degree = np.zeros((n,), dtype=np.float32)
    for i, j in edges:
        degree[i] += 1.0
        degree[j] += 1.0

    e_i = edges[:, 0]
    e_j = edges[:, 1]
    vec = points[e_j] - points[e_i]
    l0 = np.linalg.norm(vec, axis=1, keepdims=True).clip(min=1e-8)
    direction = vec / l0

    feat = np.concatenate(
        [
            l0,
            direction,
            degree[e_i, None],
            degree[e_j, None],
            local_density[e_i, None],
            local_density[e_j, None],
        ],
        axis=1,
    )
    return feat.astype(np.float32)


def _edge_sem_from_node(node_sem: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return (node_sem[edges[:, 0]] + node_sem[edges[:, 1]]) * 0.5


def _pick_best_ckpt(train_dir: str) -> str:
    best_list = sorted(glob.glob(os.path.join(train_dir, "best_*.pth")))
    if not best_list:
        raise FileNotFoundError(f"No best_*.pth found in {train_dir}")
    return best_list[-1]


@dataclass
class MaterialDatasetConfig:
    base_path: str
    sem_cache_dir: str
    experiments_dir: str
    experiments_optimization_dir: str
    case_to_material_path: str
    use_knn_topology: bool = False
    object_knn: int = 20
    object_radius: float = 0.02
    object_max_neighbours: int = 30


class MaterialParamDataset(Dataset):
    def __init__(self, cfg: MaterialDatasetConfig):
        self.cfg = cfg
        self.case_to_material, self.class_to_id = self._load_case_to_material(
            cfg.case_to_material_path
        )
        self.case_names = sorted(self.case_to_material.keys())
        self.samples: List[Dict[str, torch.Tensor]] = []
        for case_name in self.case_names:
            self.samples.append(self._build_sample(case_name))

    @staticmethod
    def _load_case_to_material(mapping_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
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
            if not class_to_id:
                class_names = sorted(set(case_to_material.values()))
                class_to_id = {name: idx for idx, name in enumerate(class_names)}
            case_to_material_id = {
                case: int(class_to_id[label]) for case, label in case_to_material.items()
            }
        else:
            case_to_material_id = {
                case: int(label) for case, label in case_to_material.items()
            }
            if not class_to_id:
                class_to_id = {
                    str(idx): idx for idx in sorted(set(case_to_material_id.values()))
                }
        return case_to_material_id, class_to_id

    def _build_sample(self, case_name: str) -> Dict[str, torch.Tensor]:
        case_dir = os.path.join(self.cfg.base_path, case_name)
        points = _load_structure_points(
            os.path.join(case_dir, "final_data.pkl")
        )
        topology_cfg = self._load_topology_from_optimization(case_name)
        edges = _build_object_edges_open3d(
            points=points,
            use_knn_topology=topology_cfg["use_knn_topology"],
            object_knn=topology_cfg["object_knn"],
            object_radius=topology_cfg["object_radius"],
            object_max_neighbours=topology_cfg["object_max_neighbours"],
        )
        z_geo = _geo_features(points, edges)

        sem_path = os.path.join(self.cfg.sem_cache_dir, f"{case_name}_node_sem.npz")
        node_sem = np.load(sem_path)["node_sem"].astype(np.float32)
        if node_sem.shape[0] != points.shape[0]:
            raise ValueError(
                f"node_sem point count mismatch for {case_name}: "
                f"{node_sem.shape[0]} vs {points.shape[0]}"
            )
        z_sem = _edge_sem_from_node(node_sem, edges)

        ckpt = torch.load(
            _pick_best_ckpt(os.path.join(self.cfg.experiments_dir, case_name, "train")),
            map_location="cpu",
        )
        num_object_springs = int(ckpt.get("num_object_springs", ckpt["spring_Y"].numel()))
        teacher_k = ckpt["spring_Y"].float().view(-1, 1)[:num_object_springs]
        base_spring_y = ckpt["spring_Y"].float().view(-1)
        if teacher_k.shape[0] != edges.shape[0]:
            raise ValueError(
                f"teacher_k size mismatch for {case_name}: {teacher_k.shape[0]} vs {edges.shape[0]} "
                f"(topology mismatch with first-order run; "
                f"use_knn_topology={topology_cfg['use_knn_topology']}, "
                f"object_knn={topology_cfg['object_knn']}, "
                f"object_radius={topology_cfg['object_radius']}, "
                f"object_max_neighbours={topology_cfg['object_max_neighbours']})"
            )

        split_path = os.path.join(case_dir, "split.json")
        with open(split_path, "r") as f:
            split = json.load(f)
        train_frame = int(split["train"][1])

        metadata_path = os.path.join(case_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        intrinsics = np.asarray(metadata["intrinsics"], dtype=np.float32)
        with open(os.path.join(case_dir, "calibrate.pkl"), "rb") as f:
            c2ws = np.asarray(pickle.load(f), dtype=np.float32)
        w2cs = np.linalg.inv(c2ws).astype(np.float32)
        wh = np.asarray(metadata["WH"], dtype=np.int64)

        return {
            "case_name": case_name,
            "case_dir": case_dir,
            "final_data_path": os.path.join(case_dir, "final_data.pkl"),
            "train_frame": torch.tensor(train_frame, dtype=torch.long),
            "cam0_intrinsics": torch.from_numpy(intrinsics[0]).float(),
            "cam0_w2c": torch.from_numpy(w2cs[0]).float(),
            "wh": torch.from_numpy(wh).long(),
            "edges": torch.from_numpy(edges).long(),
            "z_geo": torch.from_numpy(z_geo).float(),
            "z_sem": torch.from_numpy(z_sem).float(),
            "teacher_logk": teacher_k.log(),
            "base_spring_y": base_spring_y,
            "collide_elas": ckpt["collide_elas"].float().view(1),
            "collide_fric": ckpt["collide_fric"].float().view(1),
            "collide_object_elas": ckpt["collide_object_elas"].float().view(1),
            "collide_object_fric": ckpt["collide_object_fric"].float().view(1),
            "material_id": torch.tensor(int(self.case_to_material[case_name]), dtype=torch.long),
        }

    def _load_topology_from_optimization(self, case_name: str) -> Dict[str, float]:
        cfg = {
            "use_knn_topology": bool(self.cfg.use_knn_topology),
            "object_knn": int(self.cfg.object_knn),
            "object_radius": float(self.cfg.object_radius),
            "object_max_neighbours": int(self.cfg.object_max_neighbours),
        }
        optimal_path = os.path.join(
            self.cfg.experiments_optimization_dir, case_name, "optimal_params.pkl"
        )
        if not os.path.exists(optimal_path):
            return cfg

        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)

        if "use_knn_topology" in optimal_params:
            cfg["use_knn_topology"] = bool(optimal_params["use_knn_topology"])
        if "object_knn" in optimal_params:
            cfg["object_knn"] = int(optimal_params["object_knn"])
        if "object_radius" in optimal_params:
            cfg["object_radius"] = float(optimal_params["object_radius"])
        if "object_max_neighbours" in optimal_params:
            cfg["object_max_neighbours"] = int(optimal_params["object_max_neighbours"])
        return cfg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_graph_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
    out: Dict[str, List[torch.Tensor]] = {
        "case_name": [],
        "case_dir": [],
        "final_data_path": [],
        "train_frame": [],
        "cam0_intrinsics": [],
        "cam0_w2c": [],
        "wh": [],
        "edges": [],
        "z_geo": [],
        "z_sem": [],
        "teacher_logk": [],
        "base_spring_y": [],
        "collide_elas": [],
        "collide_fric": [],
        "collide_object_elas": [],
        "collide_object_fric": [],
        "material_id": [],
    }
    for sample in batch:
        out["case_name"].append(sample["case_name"])
        out["case_dir"].append(sample["case_dir"])
        out["final_data_path"].append(sample["final_data_path"])
        out["train_frame"].append(sample["train_frame"])
        out["cam0_intrinsics"].append(sample["cam0_intrinsics"])
        out["cam0_w2c"].append(sample["cam0_w2c"])
        out["wh"].append(sample["wh"])
        out["edges"].append(sample["edges"])
        out["z_geo"].append(sample["z_geo"])
        out["z_sem"].append(sample["z_sem"])
        out["teacher_logk"].append(sample["teacher_logk"])
        out["base_spring_y"].append(sample["base_spring_y"])
        out["collide_elas"].append(sample["collide_elas"])
        out["collide_fric"].append(sample["collide_fric"])
        out["collide_object_elas"].append(sample["collide_object_elas"])
        out["collide_object_fric"].append(sample["collide_object_fric"])
        out["material_id"].append(sample["material_id"])
    return out


def create_dataloader(
    cfg: MaterialDatasetConfig,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> Tuple[MaterialParamDataset, DataLoader]:
    dataset = MaterialParamDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_graph_batch,
    )
    return dataset, loader
