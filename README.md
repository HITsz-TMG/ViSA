# Picking the Cream of the Crop: Visual-Centric Data Selection with Collaborative Agents

🚀 **Welcome to the repo of ViSA!**

ViSA (**Vi**sual-Centric Data **S**election with Collaborative **A**gents) is an open-source project designed to enhance visual data selection through collaborative agents.

[![Paper](https://img.shields.io/badge/Paper-arxiv-yellow)](https://arxiv.org/abs/2502.19917)

- [x] Model Release
- [x] Data Release
- [x] Code Release

# ⚡️ Installation

To ensure smooth integration with external dependencies, we recommend setting up separate virtual environments for different components of the project.

## Setting up the VLLM Environment


```shell
conda create -n vllm python=3.11
conda activate vllm
pip install -r vllm_requirements.txt
```

**Note**: Due to existing bugs in the current VLLM `main` branch when using Qwen2-VL, we recommend using the VLLM `dev` branch instead.

```shell
conda create -n qwen_vllm python=3.11
conda activate qwen_vllm
pip install -r qwen_vllm_requirements.txt
```

## Setting up the SAM2 Environment

```shell
conda create -n sam python=3.11
conda activate sam
pip install -r sam_requirements.txt
```

## Setting up the Training Environment

We provide a simple training environment for running experiments. However, we also encourage the use of more efficient training frameworks like LLama-Factory.

```
pip install -r training_requirements.txt
```

# 🌈 Quick Start

## 📥 Model Download

We use the following large vision-language models as visual agents. Please manually download them before running the experiments:

- [Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4)
- [OpenGVLab/InternVL2_5-78B-AWQ](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4)
- [OpenGVLab/InternVL2_5-78B-MPO-AWQ](https://huggingface.co/OpenGVLab/InternVL2_5-78B-MPO-AWQ)
- [llava-hf/llava-onevision-qwen2-72b-ov-chat-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-chat-hf)


## 🔗 Repo Download

We rely on the following open-source projects. Please install them according to their official guidelines:

- [SAM2](https://github.com/facebookresearch/sam2)
- [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-SAM-2)

```bash
conda activate sam

# install sam2
git clone https://github.com/facebookresearch/sam2.git && cd sam2-main
pip install -e .

# install grounded-sam2
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git && cd Grounded-SAM-2-main
pip install -e .
pip install --no-build-isolation -e grounding_dino
```

## 🚀 Running Experiments

We provide five reference scripts for data selection. Before running them, please ensure that all necessary parameters (e.g., model paths, save directories) are correctly specified.

### Segmentation Complexity Score (SC Score) 

```bash
conda activate sam
bash Scrpit/SC_score.sh
```

### Object Alignment Score (OA Score)

```bash
conda activate sam
bash Scrpit/OA_score.sh
```

### Diversity Perspective Score (DP Score)

```bash
conda activate vllm # dev_vllm for qwen
bash Scrpit/DP_score.sh
```

### Prior Token Perplexity Score (PT Score) & Image-Text Mutual Information Score (IM Score)

```bash
conda activate vllm # dev_vllm for qwen
bash Scrpit/PT_IM_score.sh
```

# 🗝️ Dataset

You can download our dataset here. We provide two versions of the data: [ViSA-LlavaOV-80K](https://huggingface.co/datasets/foggyforest/ViSA_LlavaOV_80K) and [ViSA-LlavaOV-700K](https://huggingface.co/datasets/foggyforest/ViSA_LlavaOV_700K).

The 80K dataset can be used for small-scale multimodal model alignment or replicating the experiments in our paper, while the 700K dataset is suitable for large-scale multimodal model alignment.

Due to capacity limitations for new accounts on Huggingface, we are temporarily unable to upload data containing images. To obtain the image data, please download the original [Llava-OneVision](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data) dataset.


# 💫 Models

Our visual-semantic alignment models based on Qwen2-VL-2B architecture are available for academic research:

- [Qwen2-VL-2B-ViSA-80K](https://huggingface.co/foggyforest/Qwen2-VL-2B-ViSA-80K): Trained on ViSA-LlavaOV-80K dataset, specifically calibrated for reproducing experimental results in our publication.
- [Qwen2-VL-2B-Instruction-ViSA-700K](https://huggingface.co/foggyforest/Qwen2-VL-2B-Instruction-ViSA-700K): Enhanced through ViSA-LlavaOV-700K training, demonstrating superior multi-modal reasoning compared to its base instruction model.

(WIP) We will publish the detailed evaluation soon 

# 📢 Stay Connected

For any questions, issues, or contributions, feel free to open an issue or submit a pull request.

# Citation

If you find our model/code/paper helpful, please consider citing our papers 📝 and staring us ⭐️！

```bib
@article{liu2025picking,
  title={Picking the Cream of the Crop: Visual-Centric Data Selection with Collaborative Agents},
  author={Liu, Zhenyu and Li, Yunxin and Hu, Baotian and Luo, Wenhan and Wang, Yaowei and Zhang, Min},
  journal={arXiv preprint arXiv:2502.19917},
  year={2025}
}
```


