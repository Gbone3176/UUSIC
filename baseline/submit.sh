# python model_TU.py \
#   --input_dir /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data/Val/ \
#   --data_list /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data/private_val_for_participants.json \
#   --output_dir /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/sample_result_submission_mc\
#   --checkpoint /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_36/best_model_208_0.8276.pth \
#   --vit_type R50-ViT-B_16 \
#   --img_size 224 \
#   --device cuda

python model_TU_tta.py \
  --input_dir /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data/Val/ \
  --data_list /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/data/private_val_for_participants.json \
  --output_dir /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/sample_result_submission_tta_debug_39-3_752_0.7735\
  --checkpoint /cpfs01/projects-HDD/cfff-906dc71fafda_HDD/gbw_21307130160/challenge-main/baseline/exp_out/trail_debug_39-3/best_model_752_0.7735.pth \
  --vit_type R50-ViT-B_16 \
  --img_size 224 \
  --device cuda \
  --use_prompts \
  --use_tta \