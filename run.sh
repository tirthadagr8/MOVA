export CUDA_VISIBLE_DEVICES=0
export NCCL_DISABLE=1
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
source ~/venv/bin/activate
python scripts/inference_single.py \
    --cp_size 1 \
  --ckpt_path /media/tirthadagr8/DRIVE2/models/mova/model \
  --cp_size 1 \
  --height 352 \
  --width 640 \
  --prompt "" \
  --ref_path /home/tirthadagr8/Desktop/TI2I/Flux2/output/flux2_output.png \
  --output_path ./data/samples/single_person.mp4 \
  --offload cpu \
  --attn_type sage_auto \
  --num_inference_steps 20 \
  # --num_frames 41
