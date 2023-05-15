# accelerate launch --mixed_precision="fp16" evaluate_wse.py --task coco_order --batchsize 16 --sampling_steps 250 --img_retrieval
# tasks:
# ['pets', 'imagecode', 'imagecode_video', 'clevr', 'svo_verb', 'svo_subj', 'svo_obj', 'flickr30k_text']
# python3 evaluate_clip_baseline.py --task pets --subset
# python3 evaluate_clip_baseline.py --task imagecode
# python3 evaluate_clip_baseline.py --task imagecode_video
# python3 evaluate_clip_baseline.py --task svo_verb
# python3 evaluate_clip_baseline.py --task svo_subj
# python3 evaluate_clip_baseline.py --task svo_obj
# python3 evaluate_clip_baseline.py --task clevr
# python3 evaluate_clip_baseline.py --task flickr30k_text
# python3 evaluate_clip_baseline.py --task winoground
# python3 evaluate_clip_baseline.py --task vg_attribution
# python3 evaluate_clip_baseline.py --task coco_order
# python3 evaluate_clip_baseline.py --task flickr30k_order
# accelerate launch --mixed_precision="fp16" evaluate_wse.py --task svo_obj --batchsize 16 --sampling_steps 250 --subset
# accelerate launch --mixed_precision="fp16" evaluate_wse.py --task svo_subj --batchsize 16 --sampling_steps 250 --subset
# accelerate launch --mixed_precision="fp16" evaluate_wse.py --task svo_verb --batchsize 16 --sampling_steps 250 --subset
# python3 evaluate_clip_baseline.py --task clevr
# accelerate launch --mixed_precision="fp16" evaluate_wse.py --task clevr --batchsize 16 --sampling_steps 250
# accelerate launch --mixed_precision="fp16" evaluate_wse.py --task clevr --batchsize 16 --sampling_steps 250 --img_retrieval
# accelerate launch evaluate_wse.py --task clevr --batchsize 8 --sampling_steps 10
# accelerate launch evaluate_wse.py --task coco_order --batchsize 8 --sampling_steps 10
# accelerate launch evaluate_wse.py --task flickr30k --batchsize 8 --sampling_steps 10 --img_retrieval
# accelerate launch evaluate_wse.py --task flickr30k_order --batchsize 8 --sampling_steps 10
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval
# accelerate launch evaluate_wse.py --task svo_obj --batchsize 8 --sampling_steps 10 --img_retrieval
# accelerate launch evaluate_wse.py --task svo_subj --batchsize 8 --sampling_steps 10 --img_retrieval
# accelerate launch evaluate_wse.py --task svo_verb --batchsize 8 --sampling_steps 10 --img_retrieval
# accelerate launch evaluate_wse.py --task vg_attribution --batchsize 8 --sampling_steps 10 --img_retrieval

# accelerate launch evaluate_wse.py --task clevr --batchsize 8 --sampling_steps 10 --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task coco_order --batchsize 8 --sampling_steps 10 --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task flickr30k --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task flickr30k_order --batchsize 8 --sampling_steps 10 --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task svo_obj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task svo_subj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task svo_verb --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task vg_attribution --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/

# CUDA_VISIBLE_DEVICES=7 accelerate launch --task flickr30k_text --batchsize 8 --sampling_steps 10
# CUDA_VISIBLE_DEVICES=7 accelerate launch --task flickr30k_text --batchsize 8 --sampling_steps 10 --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# CUDA_VISIBLE_DEVICES=7 accelerate launch --task winoground --batchsize 8 --sampling_steps 10 --img_retrieval
# CUDA_VISIBLE_DEVICES=7 accelerate launch --task winoground --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# CUDA_VISIBLE_DEVICES=7 accelerate launch --task vg_relation --batchsize 8 --sampling_steps 10 --subset
# CUDA_VISIBLE_DEVICES=7 accelerate launch --task vg_relation --batchsize 8 --sampling_steps 10 --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/ --subset
# accelerate launch evaluate_wse.py --task imagecode_video --batchsize 8 --sampling_steps 10 --img_retrieval
# accelerate launch evaluate_wse.py --task imagecode_video --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task pets --batchsize 8 --sampling_steps 10 --lora_dir vanilla_finetuning_lora_savingmodel/checkpoint-4000/ --subset

# accelerate launch evaluate_wse.py --task clevr --batchsize 8 --sampling_steps 10 --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task coco_order --batchsize 8 --sampling_steps 10 --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task flickr30k --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task flickr30k_order --batchsize 8 --sampling_steps 10 --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task svo_obj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task svo_subj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task svo_verb --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task vg_attribution --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task flickr30k_text --batchsize 8 --sampling_steps 10 --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task winoground --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task vg_relation --batchsize 8 --sampling_steps 10 --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task imagecode_video --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/
# accelerate launch evaluate_wse.py --task pets --batchsize 8 --sampling_steps 10 --lora_dir relativistic_finetuning_lora_savingmodel/checkpoint-4000/ --subset

# accelerate launch evaluate_wse.py --task clevr --batchsize 8 --sampling_steps 10 --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task coco_order --batchsize 8 --sampling_steps 10 --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task flickr30k --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task flickr30k_order --batchsize 8 --sampling_steps 10 --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task svo_obj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task svo_subj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task svo_verb --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task vg_attribution --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task flickr30k_text --batchsize 8 --sampling_steps 10 --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task winoground --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task vg_relation --batchsize 8 --sampling_steps 10 --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task imagecode_video --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/
# accelerate launch evaluate_wse.py --task pets --batchsize 8 --sampling_steps 10 --lora_dir unhinged_hard_neg_coco_finetuning_lora_savingmodel_lr1e-4/checkpoint-200/ --subset


# accelerate launch evaluate_wse.py --task clevr --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task coco_order --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k_order --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task imagecode_video --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_obj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_subj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_verb --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task vg_attribution --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k_text --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task winoground --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task vg_relation --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task imagecode_video --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task pets --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000 --subset

# accelerate launch evaluate_wse.py --task clevr --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k_text --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k_order --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task coco_order --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task imagecode_video --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_obj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_subj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_verb --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task vg_attribution --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task winoground --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task vg_relation --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task pets --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000 --subset

# accelerate launch evaluate_wse.py --task clevr --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k_text --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k_order --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task coco_order --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task imagecode_video --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_obj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_subj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_verb --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task vg_attribution --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task winoground --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task vg_relation --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task pets --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000 --subset

# accelerate launch evaluate_wse.py --task clevr --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k_text --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task flickr30k_order --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task coco_order --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task imagecode --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task imagecode_video --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_obj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_subj --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task svo_verb --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task vg_attribution --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task winoground --batchsize 8 --sampling_steps 10 --img_retrieval --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task vg_relation --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000
# accelerate launch evaluate_wse.py --task pets --batchsize 8 --sampling_steps 10 --lora_dir vanilla_coco_finetuning_lora_savingmodel_lr1e-4_LONGER/checkpoint-3000 --subset

accelerate launch evaluate_wse.py --task flickr30k_text --batchsize 8 --sampling_steps 250 --subset --gray_baseline
accelerate launch evaluate_wse.py --task flickr30k_order --batchsize 8 --sampling_steps 250 --subset --gray_baseline
accelerate launch evaluate_wse.py --task coco_order --batchsize 8 --sampling_steps 250 --subset --gray_baseline
accelerate launch evaluate_wse.py --task vg_attribution --batchsize 8 --sampling_steps 250 --subset --gray_baseline
accelerate launch evaluate_wse.py --task vg_relation --batchsize 8 --sampling_steps 250 --subset --gray_baseline
accelerate launch evaluate_wse.py --task clevr --batchsize 8 --sampling_steps 250 --gray_baseline
accelerate launch evaluate_wse.py --task pets --batchsize 8 --sampling_steps 250 --subset --gray_baseline
accelerate launch evaluate_wse.py --task winoground --batchsize 8 --sampling_steps 250 --gray_baseline