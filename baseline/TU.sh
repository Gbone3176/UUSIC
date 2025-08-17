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
#   omni_train_TU_v4.py \
#   --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
#   --img_size 224 \
#   --batch_size 128 \
#   --base_lr 1e-4 \
#   --max_epochs 800 \
#   --gpu 0 \
#   --seed 1206 \
#   --output_dir exp_out/trail_debug_44 \
#   --prompt \
#   --pretrain_ckpt /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_42/best_model_119_0.7943.pth


####### train TU v4 魔改network #######
# torchrun \
#   --nproc_per_node=1 \
#   --master_port=1234 \
#   omni_train_TU_v4.py \
#   --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
#   --img_size 224 \
#   --batch_size 128 \
#   --base_lr 1e-3 \
#   --max_epochs 800 \
#   --gpu 0 \
#   --seed 1206 \
#   --output_dir exp_out/trail_debug_44-2-3 \
#   --prompt \
#   --pretrain_ckpt /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_44/best_model_129_0.5491.pth

# ######## test TU v2 ########
# torchrun \
#   --nproc_per_node=1 \
#   --master_port=23145 \
#   omni_test_TU_v2.py \
#   --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
#   --output_dir test_out/test_36 \
#   --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_36/best_model_208_0.8276.pth \
#   --batch_size 1 \
#   --img_size 224 \
#   --is_saveout \
#   --prompt \


# ######## test TU v2 tta ########
# torchrun \
#   --nproc_per_node=1 \
#   --master_port=21345 \
#   omni_test_TU_tta.py \
#   --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
#   --output_dir test_out/test_44-2-3 \
#   --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_44-2-3/best_model_367_0.8287.pth \
#   --batch_size 1 \
#   --img_size 224 \
#   --is_saveout \
#   --prompt \
#   --use_tta \


# ######## test TU v2 tta ########
torchrun \
  --nproc_per_node=1 \
  --master_port=21345 \
  omni_test_TU_tta_45-1.py \
  --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
  --output_dir test_out/test_44-2-3_572_0.8393 \
  --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_44-2-3/best_model_572_0.8393.pth \
  --batch_size 1 \
  --img_size 224 \
  --is_saveout \
  --prompt \
  --use_tta \

# ######## test TU tta V45-1-2 ########
# torchrun \
#   --nproc_per_node=1 \
#   --master_port=21345 \
#   omni_test_TU_tta_45-1-3.py \
#   --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
#   --output_dir test_out/test_45-1-3 \
#   --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_44-2-3/best_model_367_0.8287.pth \
#   --batch_size 1 \
#   --img_size 224 \
#   --is_saveout \
#   --prompt \
#   --use_tta \
#   --ms_scales_thyroid 1.0 \
#   --ms_scales_default 1.0 \
