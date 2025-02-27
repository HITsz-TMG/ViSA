cd {Your ViSA path}/Codes
export CUDA_VISIBLE_DEVICES="0"

exprot prior_token_num=10

# Qwen2-VL-72B
export model_path="path of Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4"
export vllm="Qwen2VL"
export data_path="Your data path"
export image_path="Your image path, e.g.: LLaVA-OneVision/llava-onevision"
export save_path1="Your save path"

python text_quantification/PT_IM_score.py \
--model_path $model_path \
--vllm $vllm \
--data_path $data_path \
--image_path $image_path \
--save_path $save_path1 \
--prior_token_num $prior_token_num

# InternVL2_5-78B
export model_path="path of OpenGVLab/InternVL2_5-78B-AWQ"
export vllm="InternVL2_5_mpo"
export data_path="Your data path"
export image_path="Your image path, e.g.: LLaVA-OneVision/llava-onevision"
export save_path2="Your save path"

python text_quantification/PT_IM_score.py \
--model_path $model_path \
--vllm $vllm \
--data_path $data_path \
--image_path $image_path \
--save_path $save_path2 \
--prior_token_num $prior_token_num


# InternVL2_5-78B-MPO
export model_path="path of OpenGVLab/InternVL2_5-78B-MPO-AWQ"
export vllm="InternVL2_5_mpo"
export data_path="Your data path"
export image_path="Your image path, e.g.: LLaVA-OneVision/llava-onevision"
export save_path3="Your save path"

python text_quantification/PT_IM_score.py \
--model_path $model_path \
--vllm $vllm \
--data_path $data_path \
--image_path $image_path \
--save_path $save_path3 \
--prior_token_num $prior_token_num


# llava on
export model_path="lmms-lab/llava-onevision-qwen2-72b-ov-chat"
export vllm="Llava"
export data_path="Your data path"
export image_path="Your image path, e.g.: LLaVA-OneVision/llava-onevision"
export save_path4="Your save path"

python text_quantification/PT_IM_score.py \
--model_path $model_path \
--vllm $vllm \
--data_path $data_path \
--image_path $image_path \
--save_path $save_path4 \
--prior_token_num $prior_token_num


# shapley aggregation

python visual_quantification/Shapley.py \
--data_path $save_path1 $save_path2 $save_path3 $save_path4 \
--save_path "Your final save path for PT score" \
--score_key "PT score"

python visual_quantification/Shapley.py \
--data_path $save_path1 $save_path2 $save_path3 $save_path4 \
--save_path "Your final save path for IM score" \
--score_key "IM score"


