cd "Your ViSA path"/Codes
export PYTHONPATH=$PYTHONPATH:"Your SAM2.1 path, e.g.:sam2-main"
export CUDA_VISIBLE_DEVICES="0"

export model_path="Your sam model path, e.g.: sam2.1_hiera_large.pt"
export data_path="Your data path"
export image_path="Your image path, e.g.: LLaVA-OneVision/llava-onevision"
export save_path="Your save path"

python visual_quantification/SC_score.py \
--model_path $model_path \
--data_path $data_path \
--image_path $image_path \
--save_path $save_path

