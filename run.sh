# 1) extract semantic cache
python semantic/extract_dino_semantic_features.py \
  --base_path /home/yang/Projects/Phys-GS/data/different_types \
  --cases $(python - <<'PY'
import json
p='/home/yang/Projects/Phys-GS/semantic/case_to_material_different_types.json'
d=json.load(open(p))
print(" ".join(sorted(d["case_to_material"].keys())))
PY
) \
  --output_dir /home/yang/Projects/Phys-GS/semantic/cache \
  --device cuda \
  --frame_idx 0

# 2) train paramnet (k-only)
python semantic/train_paramnet_konly.py \
  --base_path /home/yang/Projects/Phys-GS/data/different_types \
  --sem_cache_dir /home/yang/Projects/Phys-GS/semantic/cache \
  --experiments_dir /home/yang/Projects/Phys-GS/experiments \
  --case_to_material /home/yang/Projects/Phys-GS/semantic/case_to_material_different_types.json \
  --save_dir /home/yang/Projects/Phys-GS/semantic/checkpoints \
  --batch_size 2 \
  --epochs 100 \
  --device cuda