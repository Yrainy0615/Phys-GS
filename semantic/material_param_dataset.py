import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
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


def _knn_graph(points: np.ndarray, k: int) -> np.ndarray:
    n = points.shape[0]
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, kth=min(k, n - 1), axis=1)[:, :k]
    edges = set()
    for i in range(n):
        for j in nn_idx[i]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b:
                edges.add((a, b))
    return np.array(sorted(edges), dtype=np.int64)


def _radius_graph(points: np.ndarray, radius: float, max_neighbors: int) -> np.ndarray:
    n = points.shape[0]
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    r2 = radius * radius
    edges = set()
    for i in range(n):
        cand = np.where(d2[i] <= r2)[0]
        cand = cand[cand != i]
        if cand.size > max_neighbors:
            order = np.argsort(d2[i, cand])
            cand = cand[order[:max_neighbors]]
        for j in cand:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b:
                edges.add((a, b))
    return np.array(sorted(edges), dtype=np.int64)


def _build_edges(
    points: np.ndarray,
    use_knn: bool,
    object_knn: int,
    radius: float,
    max_neighbors: int,
) -> np.ndarray:
    if use_knn:
        return _knn_graph(points, object_knn)
    return _radius_graph(points, radius, max_neighbors)


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
            sample = self._build_sample(case_name)
            self.samples.append(sample)

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
            case_to_material_id = {case: int(label) for case, label in case_to_material.items()}
            if not class_to_id:
                class_to_id = {
                    str(idx): idx for idx in sorted(set(case_to_material_id.values()))
                }

        return case_to_material_id, class_to_id

    def _build_sample(self, case_name: str) -> Dict[str, torch.Tensor]:
        final_data_path = os.path.join(self.cfg.base_path, case_name, "final_data.pkl")
        points = _load_structure_points(final_data_path)
        edges = _build_edges(
            points=points,
            use_knn=self.cfg.use_knn_topology,
            object_knn=self.cfg.object_knn,
            radius=self.cfg.object_radius,
            max_neighbors=self.cfg.object_max_neighbours,
        )
        z_geo = _geo_features(points, edges)

        sem_path = os.path.join(self.cfg.sem_cache_dir, f"{case_name}_node_sem.npz")
        sem_npz = np.load(sem_path)
        node_sem = sem_npz["node_sem"].astype(np.float32)
        if node_sem.shape[0] != points.shape[0]:
            raise ValueError(
                f"node_sem point count mismatch for {case_name}: "
                f"{node_sem.shape[0]} vs {points.shape[0]}"
            )
        z_sem = _edge_sem_from_node(node_sem, edges)

        train_dir = os.path.join(self.cfg.experiments_dir, case_name, "train")
        ckpt_path = _pick_best_ckpt(train_dir)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        teacher_k = ckpt["spring_Y"].float().view(-1, 1)
        if teacher_k.shape[0] != edges.shape[0]:
            raise ValueError(
                f"teacher_k size mismatch for {case_name}: "
                f"{teacher_k.shape[0]} vs {edges.shape[0]}"
            )

        material_id = int(self.case_to_material[case_name])
        return {
            "case_name": case_name,
            "edges": torch.from_numpy(edges).long(),
            "z_geo": torch.from_numpy(z_geo).float(),
            "z_sem": torch.from_numpy(z_sem).float(),
            "teacher_logk": teacher_k.log(),
            "material_id": torch.tensor(material_id, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_graph_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
    out: Dict[str, List[torch.Tensor]] = {
        "case_name": [],
        "edges": [],
        "z_geo": [],
        "z_sem": [],
        "teacher_logk": [],
        "material_id": [],
    }
    for sample in batch:
        out["case_name"].append(sample["case_name"])
        out["edges"].append(sample["edges"])
        out["z_geo"].append(sample["z_geo"])
        out["z_sem"].append(sample["z_sem"])
        out["teacher_logk"].append(sample["teacher_logk"])
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
import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
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


def _knn_graph(points: np.ndarray, k: int) -> np.ndarray:
    n = points.shape[0]
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, kth=min(k, n - 1), axis=1)[:, :k]
    edges = set()
    for i in range(n):
        for j in nn_idx[i]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b:
                edges.add((a, b))
    return np.array(sorted(edges), dtype=np.int64)


def _radius_graph(points: np.ndarray, radius: float, max_neighbors: int) -> np.ndarray:
    n = points.shape[0]
    d2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    r2 = radius * radius
    edges = set()
    for i in range(n):
        cand = np.where(d2[i] <= r2)[0]
        cand = cand[cand != i]
        if cand.size > max_neighbors:
            order = np.argsort(d2[i, cand])
            cand = cand[order[:max_neighbors]]
        for j in cand:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if a != b:
                edges.add((a, b))
    return np.array(sorted(edges), dtype=np.int64)


def _build_edges(points: np.ndarray, use_knn: bool, object_knn: int, radius: float, max_neighbors: int) -> np.ndarray:
    if use_knn:
        return _knn_graph(points, object_knn)
    return _radius_graph(points, radius, max_neighbors)


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
            sample = self._build_sample(case_name)
            self.samples.append(sample)

    @staticmethod
    def _load_case_to_material(mapping_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
        with open(mapping_path, "r") as f:
            raw = json.load(f)

        # Supports:
        # 1) {"caseA": 0, "caseB": 1}
        # 2) {"caseA": "cloth", ...}
        # 3) {"class_to_id": {...}, "case_to_material": {...}}
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
                class_to_id = {str(idx): idx for idx in sorted(set(case_to_material_id.values()))}

        return case_to_material_id, class_to_id

    def _build_sample(self, case_name: str) -> Dict[str, torch.Tensor]:
        final_data_path = os.path.join(self.cfg.base_path, case_name, "final_data.pkl")
        points = _load_structure_points(final_data_path)
        edges = _build_edges(
            points=points,
            use_knn=self.cfg.use_knn_topology,
            object_knn=self.cfg.object_knn,
            radius=self.cfg.object_radius,
            max_neighbors=self.cfg.object_max_neighbours,
        )
        z_geo = _geo_features(points, edges)

        sem_path = os.path.join(self.cfg.sem_cache_dir, f"{case_name}_node_sem.npz")
        sem_npz = np.load(sem_path)
        node_sem = sem_npz["node_sem"].astype(np.float32)
        if node_sem.shape[0] != points.shape[0]:
            raise ValueError(
                f"node_sem point count mismatch for {case_name}: "
                f"{node_sem.shape[0]} vs {points.shape[0]}"
            )
        z_sem = _edge_sem_from_node(node_sem, edges)

        train_dir = os.path.join(self.cfg.experiments_dir, case_name, "train")
        ckpt_path = _pick_best_ckpt(train_dir)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        teacher_k = ckpt["spring_Y"].float().view(-1, 1)
        if teacher_k.shape[0] != edges.shape[0]:
            raise ValueError(
                f"teacher_k size mismatch for {case_name}: "
                f"{teacher_k.shape[0]} vs {edges.shape[0]}"
            )

        material_id = int(self.case_to_material[case_name])
        return {
            "case_name": case_name,
            "edges": torch.from_numpy(edges).long(),
            "z_geo": torch.from_numpy(z_geo).float(),
            "z_sem": torch.from_numpy(z_sem).float(),
            "teacher_logk": teacher_k.log(),
            "material_id": torch.tensor(material_id, dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_graph_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, List[torch.Tensor]]:
    out: Dict[str, List[torch.Tensor]] = {
        "case_name": [],
        "edges": [],
        "z_geo": [],
        "z_sem": [],
        "teacher_logk": [],
        "material_id": [],
    }
    for sample in batch:
        out["case_name"].append(sample["case_name"])
        out["edges"].append(sample["edges"])
        out["z_geo"].append(sample["z_geo"])
        out["z_sem"].append(sample["z_sem"])
        out["teacher_logk"].append(sample["teacher_logk"])
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
