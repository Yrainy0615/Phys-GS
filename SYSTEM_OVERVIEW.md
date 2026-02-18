# PhysTwin System Overview

> Reference: [PhysTwin Paper (arXiv:2503.17973)](https://arxiv.org/pdf/2503.17973) – *Physics-Informed Reconstruction and Simulation of Deformable Objects from Videos*

This document describes the training pipeline, data flow, and objectives of each stage in the PhysTwin framework.

---

## 1. Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          PhysTwin Training Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  Raw RGBD Videos  →  Data Processing  →  Zero-Order Opt  →  First-Order Opt  →  ...   │
│       (3 cams)         (script_process)   (script_optimize)   (script_train)          │
│                                                                                       │
│  ...  →  Inference  →  Gaussian Training  →  Interactive Playground                   │
│           (script_inference)   (gs_run.sh)      (ready for use)                        │
│                                                                                       │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**Output Artifacts:**

| Stage | Output | Used By |
|-------|--------|---------|
| Data Processing | `./data/different_types/{case}/final_data.pkl` | All optimization & inference |
| Zero-Order Opt | `./experiments_optimization/{case}/optimal_params.pkl` | First-order opt, inference, playground |
| First-Order Opt | `./experiments/{case}/train/best_*.pth` | Inference, playground |
| Inference | Simulated trajectories in `./experiments/{case}/` | Gaussian LBS, evaluation |
| Gaussian Training | `./gaussian_output/{case}/` | Interactive playground (rendering) |

---

## 2. Data Requirements & Processing

**Input:** Raw RGBD videos (3 RealSense-D455 cameras), each case with:
- `color/`, `depth/`, `calibrate.pkl`, `metadata.json`

**Data Processing** (run before training):

```bash
python script_process_data.py      # → final_data.pkl, shape prior, masks, tracking
python export_gaussian_data.py     # → ./data/gaussian_data/{case}/ for first-frame Gaussians
python export_video_human_mask.py  # (optional) for evaluation masks
```

**What `final_data.pkl` contains:**
- Object point clouds (from depth + TRELLIS shape prior)
- Controller (hand) trajectories (Grounded-SAM2 + CoTracker3)
- Visibility masks per frame
- 3D-lifted tracking from CoTracker3

---

## 3. Stage-by-Stage Training Objectives

### 3.1 Zero-Order Optimization (CMA-ES)

**Script:** `script_optimize.py` → `optimize_cma.py`

**Objective:** Optimize **non-differentiable** sparse parameters so simulated trajectories best match observed geometry and motion.

**Optimized parameters (12 scalar / int values):**

| Parameter | Role |
|-----------|------|
| `global_spring_Y` | Homogeneous spring stiffness (elasticity) |
| `object_radius` | Object topology: connection radius for object-object springs |
| `object_max_neighbours` | Max neighbours per object node |
| `controller_radius` | Controller–object spring connection radius |
| `controller_max_neighbours` | Max neighbours per controller node |
| `collide_elas`, `collide_fric` | Table/collider elasticity & friction |
| `collide_object_elas`, `collide_object_fric` | Object self-collision elasticity & friction |
| `collision_dist` | Collision distance threshold |
| `drag_damping`, `dashpot_damping` | Damping coefficients |

**Loss function** (per frame, averaged over `train_frame`):

$$L = C_{\text{geometry}} + C_{\text{motion}} + C_{\text{acc}}$$

- **\(C_{\text{geometry}}\)** – Single-direction Chamfer distance between predicted mass positions and observed partial point cloud
- **\(C_{\text{motion}}\)** – Tracking error between predicted positions and CoTracker3 pseudo-GT
- **\(C_{\text{acc}}\)** – Acceleration regularization for plausible dynamics

**Paper alignment:** *Sec. 4.2.1 “Sparse-to-Dense Optimization”* – zero-order sampling for non-differentiable topology and sparse physical parameters.

**Output:** `experiments_optimization/{case_name}/optimal_params.pkl`

---

### 3.2 First-Order Optimization (Gradient Descent)

**Script:** `script_train.py` → `train_warp.py`

**Objective:** Refine **differentiable** parameters using the topology and sparse physics from zero-order optimization. Uses a differentiable Warp-based spring-mass simulator.

**Optimized parameters:**
- Per-spring stiffness `spring_Y` (dense)
- Collision parameters: `collide_elas`, `collide_fric`, `collide_object_elas`, `collide_object_fric`

**Loss:** Same as zero-order: Chamfer + tracking + acceleration.

**Paper alignment:** *Sec. 4.2.1* – first-order gradient descent for dense spring stiffness and collision parameters.

**Output:** `experiments/{case_name}/train/best_*.pth`

---

### 3.3 Inference

**Script:** `script_inference.py` → `inference_warp.py`

**Objective:** Run the trained physics model on the full video sequence to obtain:
- Simulated mass-node trajectories
- Simulated velocities

**Inputs:**
- `experiments_optimization/{case}/optimal_params.pkl`
- `experiments/{case}/train/best_*.pth`
- `./data/different_types/{case}/final_data.pkl`

**Output:** Simulated states saved under `experiments/{case}/` for use in Gaussian deformation (LBS) and evaluation.

---

### 3.4 Gaussian Splatting (Appearance)

**Script:** `gs_run.sh` → `gs_train.py` (3D Gaussian Splatting)

**Objective:** Train **static** 3D Gaussians on the **first frame** of the multi-view sequence to model appearance.

**Input:** `./data/gaussian_data/{scene}/` (RGB, masks, depth from 3 views for frame 0)

**Loss:**
- L1 + D-SSIM (photometric)
- Depth supervision
- Segmentation mask (foreground)

**Output:** `gaussian_output/{scene}/{exp_name}/` – trained 3D Gaussians for rendering.

**Paper alignment:** *Sec. 4.2.2 “Appearance Optimization”* – static Gaussians at \(t=0\), then deformed via LBS driven by physics simulation.

---

## 4. End-to-End Flow

```text
1. Data Processing
   Raw RGBD → final_data.pkl + gaussian_data/

2. Zero-Order (CMA-ES)
   final_data.pkl + shape prior → optimal_params.pkl
   Minimizes: Chamfer + Track + Acc (non-diff topology + sparse physics)

3. First-Order (Adam)
   optimal_params.pkl + final_data.pkl → best_*.pth
   Minimizes: Chamfer + Track + Acc (dense spring stiffness + collision)

4. Inference
   optimal_params + best_*.pth + final_data.pkl → simulated trajectories

5. Gaussian Training
   gaussian_data/ (frame 0) → gaussian_output/

6. Interactive Playground
   Load: optimal_params, best_*.pth, gaussian_output, final_data
   Physics sim drives LBS-deformed Gaussians for real-time rendering
```

---

## 5. Paper Formulation Mapping

**Optimization problem** (Eq. 2 in paper):

$$\min_{\alpha, \mathcal{G}_0, \theta} \sum_{t,i} C(\hat{\mathbf{O}}_{t,i}, \mathbf{O}_{t,i})$$

- **Physics stage (Eq. 3):**  
  \(\min_{\alpha, \mathcal{G}_0} \sum_t (C_{\text{geometry}} + C_{\text{motion}})\)

  - Implemented as: Zero-order (sparse) → First-order (dense), with \(C_{\text{acc}}\) as regularization.

- **Appearance stage (Eq. 4):**  
  \(\min_\theta \sum_{t,i} C_{\text{render}}(\hat{\mathbf{I}}_{i,t}, \mathbf{I}_{i,t})\)

  - Implemented as: `gs_train.py` on first frame only.

---

## 6. Running a Subset of Cases

Edit the scripts to restrict which cases are processed.

**`script_optimize.py` / `script_train.py`:**

```python
# Change this line:
dir_names = glob.glob(f"{base_path}/*")

# To filter specific cases, e.g.:
target_cases = ["double_stretch_sloth", "single_push_rope_1"]
dir_names = [f"{base_path}/{c}" for c in target_cases]
```

**`gs_run.sh`:**

```bash
# Uncomment and edit the scenes array:
scenes=("double_stretch_sloth" "single_push_rope_1")
```

---

## 7. Quick Reference: Commands

| Step | Command | Duration (approx.) |
|------|---------|-------------------|
| Zero-Order | `python script_optimize.py` | ~12 min/case |
| First-Order | `python script_train.py` | ~5 min/case |
| Inference | `python script_inference.py` | - |
| Gaussian | `bash gs_run.sh` | Depends on iterations |
| Playground | `python interactive_playground.py --case_name X --n_ctrl_parts 2` | - |
