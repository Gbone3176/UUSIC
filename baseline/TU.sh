CUDA_VISIBLE_DEVICES=0

######## train TU ########
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


# ######## test TU ########
torchrun \
    --nproc_per_node=1 \
    --master_port=2345 \
    omni_test_TU.py \
    --root_path /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data \
    --output_dir test_out/test_trail_debug_35 \
    --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_35/best_model_250_0.7912.pth \
    --batch_size 1 \
    --img_size 224 \
    --is_saveout \

