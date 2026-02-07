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
  --prompt "Create a AI-powered sensory ad for PC Chandra Jewellers' gold butterfly earrings. \n\n0-0.5s: EXTREME MACRO SHOT - Earring vibrates at 40Hz (imperceptible), biometric sensors translate model's heartbeat into cobalt-blue diamond pulses. ASMR whisper: 'Feel the rhythm.'\n\n0.5-2.5s: 360° ZERO-G ROTATION - Light bends around wings like liquid gold. Diamonds refract light into holographic butterflies that fly into model's hair. Sound: Crystal singing bowl + sub-bass heartbeat.\n\n2.5-4.5s: SKIN-TO-GOLD SYNCH - Model touches earlobe; thermal AI triggers gold wings warming from ivory to molten gold. Diamonds melt into liquid light along edges then re-solidify as new clusters. Sound: Sizzling silk + diamond 'crack'.\n\n4.5-6s: BREATH-ANIMATED TEXT - Model's exhaled breath condenses into micro-diamond particles that assemble 'PC Chandra Jewellers' in 3D gold filigree. Text floats upward with golden vapor trail, shatters on inhale, reforms on exhale.\n\n6-7s: SILENT IMPLOSION - Freeze frame: Earring glowing with trapped breath-light. All sound cuts. Final frame: 'PC Chandra' etched into earring's shadow on neck. \n\nTECH SPECS: 4K 120fps slow-mo, buttery color grading, tactile audio mapping. No stock footage - all generated from CAD model + biometric data. First 0.5s must stop scrollers." \
  --ref_path /home/tirthadagr8/Desktop/TI2I/Flux2/output/flux2_output.png \
  --output_path ./data/samples/single_person.mp4 \
  --offload cpu \
  --attn_type sage_auto \
  --num_inference_steps 20 \
  # --num_frames 41
