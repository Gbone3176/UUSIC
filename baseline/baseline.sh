export CUDA_VISIBLE_DEVICES=0
######## train PG ########
# torchrun \
#     --nproc_per_node=1 \
#     --master_port=12345 \
#     omni_train_PG.py \
#     --output_dir=exp_out/trail_debug_32-1 \
#     --prompt \
#     --base_lr=0.00001 \
#     --batch_size=128 \
#     --max_epochs=300 \
#     --root_path=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data/ \
#     --pretrain_ckpt=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_32/best_model_340_0.7503.pth \
    # --resume=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_32/best_model_340_0.7503.pth \
    
######## test PG ########
torchrun \
    --nproc_per_node=1 \
    --master_port=2345 \
    omni_test_PG.py \
    --output_dir=exp_out/test_trail_debug_32-1 \
    --prompt \
    --base_lr=0.00001 \
    --batch_size=1 \
    --max_epochs=400 \
    --root_path=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data/ \
    --resume=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_32-1/best_model_19_0.7812.pth \


######## test ########
# torchrun \
#     --nproc_per_node=2 \
#     --master_port=12345 \
#     omni_test.py \
#     --output_dir=exp_out/cfff_debug_7 \
#     --prompt \
#     --base_lr=0.00001 \
#     --batch_size=128 \
#     --max_epochs=400 \
#     --root_path=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data/ \
#     --resume=/cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/cfff_debug_7/best_model_266_0.3994.pth \

# 或者保持旧命令（添加--no-python参数减少警告）
# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node=2 \
#     --master_port=12345 \
#     omni_train.py \
#     --output_dir=exp_out/trial_uusic_1 \
#     --prompt \
#     --base_lr=0.003 \
#     --batch_size=128


############# test the model #############

# torchrun \
#     --nproc_per_node=1 \
#     --master_port=2345 \
#     omni_test.py \
#     --output_dir=exp_out/trial_uusic_9 \
#     --base_lr=0.00001 \
#     --prompt \


############# generate the submission file #############

# python model.py