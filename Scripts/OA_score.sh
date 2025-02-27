cd "Your ViSA path"/Codes
export PYTHONPATH=$PYTHONPATH:"Your SAM2.1 path, e.g.:Grounded-SAM-2-main"
export CUDA_VISIBLE_DEVICES="0"

export model_path="Your grounded sam model path, e.g.: grounding-dino-base"
export tag_path="Your tag file path, e.g.: ram_tag_list_1k8.txt"
export data_path="Your data path"
export image_path="Your image path, e.g.: LLaVA-OneVision/llava-onevision"
export oa_save_path="Your grounding result save path"
export save_path="Your save path"

python visual_quantification/OA_score.py \
--model_path $model_path \
--tag_path $tag_path \
--data_path $data_path \
--image_path $image_path \
--save_path $oa_save_path

python visual_quantification/OA_score_postprocess.py --data_path $oa_save_path --save_path $save_path
