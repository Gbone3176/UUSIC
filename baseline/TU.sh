CUDA_VISIBLE_DEVICES=0

######## train TU v1 ########
# torchrun \
#   --nproc_per_node=1 \
#   --master_port=29501 \
#   omni_train_TU.py \
#   --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
#   --img_size 224 \
#   --batch_size 128 \
#   --base_lr 1e-4 \
#   --max_epochs 400 \
#   --gpu 0 \
#   --seed 42 \
#   --output_dir exp_out/trail_debug_35 \


# ######## test TU v1 ########
# torchrun \
#     --nproc_per_node=1 \
#     --master_port=2345 \
#     omni_test_TU_tta.py \
#     --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
#     --output_dir test_out/test_trail_debug_35 \
#     --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_35/best_model_314_0.7968.pth \
#     --batch_size 1 \
#     --img_size 224 \
#     --is_saveout \

######## train TU v2 ########
# torchrun \
#   --nproc_per_node=1 \
#   --master_port=12345 \
#   omni_train_TU_v2.py \
#   --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
#   --img_size 224 \
#   --batch_size 128 \
#   --base_lr 1e-4 \
#   --max_epochs 720 \
#   --gpu 0 \
#   --seed 1206 \
#   --output_dir exp_out/trail_debug_42 \
#   --prompt \
  # --pretrain_ckpt /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_39-3/best_model_752_0.7735.pth


####### train TU v3 运行一些特殊修改########
# torchrun \
#   --nproc_per_node=1 \
#   --master_port=23451 \
#   omni_train_TU_v3.py \
#   --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
#   --img_size 224 \
#   --batch_size 64 \
#   --base_lr 1e-4 \
#   --max_epochs 500 \
#   --gpu 0 \
#   --seed 23 \
#   --output_dir exp_out/trail_debug_42-1 \
#   --prompt \
#   --pretrain_ckpt /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_42/best_model_119_0.7943.pth

# ######## test TU v2 ########
# torchrun \
#   --nproc_per_node=1 \
#   --master_port=2345 \
#   omni_test_TU_v2.py \
#   --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
#   --output_dir test_out/test_final_cur \
#   --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_39-3/best_model_752_0.7735.pth \
#   --batch_size 1 \
#   --img_size 224 \
#   --is_saveout \
#   --prompt \


# ######## test TU v2 tta ########
torchrun \
  --nproc_per_node=1 \
  --master_port=2345 \
  omni_test_TU_tta.py \
  --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
  --output_dir test_out/test_final_next \
  --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_42/best_model_641_0.8198.pth \
  --batch_size 1 \
  --img_size 224 \
  --is_saveout \
  --prompt \
  --use_tta \