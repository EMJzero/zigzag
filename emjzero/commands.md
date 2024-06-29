### Run TPU
python3.11 main_salsa.py --mapping inputs/mapping/edge_tpu_like.yaml --model emjzero/KQV_gemm_layer.yaml --accelerator inputs/hardware/edge_tpu_like.yaml

### Run Eyeriss
python3.11 main.py --mapping inputs/mapping/eyeriss_like.yaml --model emjzero/KQV_gemm_layer.yaml --accelerator inputs/hardware/eyeriss_like.yaml

### Visualize Mapping
python3.11 visualization.py
#### With Automatic Substitution
python3.11 visualization.py | sed 's/ C / E /g' | sed 's/ K / D /g' | sed 's/ OY / L /g'