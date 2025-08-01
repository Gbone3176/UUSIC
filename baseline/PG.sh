export CUDA_VISIBLE_DEVICES=0
######## train PG ########
# torchrun \
#     --nproc_per_node=1 \
#     --master_port=12345 \
#     omni_train_PG_decoders.py \
#     --output_dir=exp_out/trail_debug_18 \
#     --prompt \
#     --base_lr=0.001 \
#     --batch_size=128 \
#     --max_epochs=200 \
#     --root_path=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data/ \
    
######## test PG ########
torchrun \
    --nproc_per_node=1 \
    --master_port=2345 \
    omni_test_PG_decoders.py \
    --root_path ./data \
    --output_dir exp_out/trail_debug_17—294epoch \
    --resume /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_16/best_model_294_0.4725.pth \
    --batch_size 1 \
    --img_size 224 \
    --cfg configs/swin_tiny_patch4_window7_224_lite-PG.yaml \
    --prompt \
    --is_saveout \


