export CUDA_VISIBLE_DEVICES=0,1
######## train PG ########
torchrun \
    --nproc_per_node=2 \
    --master_port=12345 \
    omni_train_PG_decoders.py \
    --output_dir=exp_out/train_debug_22 \
    --prompt \
    --base_lr=0.00001 \
    --batch_size=128 \
    --max_epochs=300 \
    --resume=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/train_debug_22/latest_111.pth \
    --root_path=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data/ \
    
# ######## test PG ########
# torchrun \
#     --nproc_per_node=1 \
#     --master_port=2345 \
#     omni_test_PG_decoders.py \
#     --root_path ./data \
#     --output_dir exp_out/test_trail_debug_22 \
#     --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_22/epoch40_performance_0.274.pth \
#     --batch_size 1 \
#     --img_size 224 \
#     --cfg configs/swin_tiny_patch4_window7_224_lite-PG.yaml \
#     --prompt \
#     --is_saveout \


######## train PG(Frozen Encoder) ########
# torchrun \
#     --nproc_per_node=1 \
#     --master_port=12345 \
#     omni_train_PG_decoders.py \
#     --output_dir=exp_out/trail_debug_22 \
#     --prompt \
#     --base_lr=0.00001 \
#     --batch_size=128 \
#     --max_epochs=400 \
#     --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/pretrained_ckpt/perceptguide.pth \
#     --root_path=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data/