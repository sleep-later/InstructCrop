# InstructCrop

Official code release: InstructCrop — Teaching Multimodal Large Language Models to Crop Aesthetic Images

## 1. Installation
1. Install dependencies:
```sh
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
or set up the same environment as LLaVA: https://github.com/haotian-liu/LLaVA

3. Build rod_align (required for anchor/grid utilities):
Refer to: https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch
The repository contains build scripts under `rod_align/` (e.g., `make.sh`, `setup.py`).

4. Download base model for `--base_model_path`:
You need to obtain a compatible base LLaVA model listed in the LLaVA MODEL_ZOO:
https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md

Example version used in this project: LLaVA-1.5-7B (Hugging Face):
https://huggingface.co/liuhaotian/llava-v1.5-7b

Place the downloaded base model path into `--base_model_path` when running `inference.py`.

## 2. Cropping model
Setup:
- Compile and install the `rod_align` dependencies.
- Place downloaded model weights in an accessible path for inference scripts.

Architecture:
- Base Model (LLaVA + LoRA): multimodal backbone handling image + prompt.
- Cropping Model (Stage 1): generates initial candidate boxes.
- Instruct Model (Stage 2): refines boxes for high precision.

Key files:
- Inference entry: `inference.py`
- Model implementations: `model/text_baseline.py` (contains LICAForCausalLM, Cropping_LICA, Instruct_Model)
- LLaVA utilities: `model/llava/`
- rod_align tools: `rod_align/`

## 3. Download Weights
Download model weights from:
https://drive.google.com/drive/folders/1XGZnW_GaFaz8yL4fn7DNjGl-kSS-eMFs?usp=drive_link

## 4. Inference (single image)
Run from project root:
```sh
python inference.py \
  --base_model_path <path_to_base_llava_model> \
  --vision_tower_path openai/clip-vit-large-patch14 \
  --stage1_lora_path <path_to_stage1_lora> \
  --stage1_ckpt_path <stage1_ckpt.pth> \
  --stage2_ckpt_path <stage2_ckpt.pth> \
  --image_path <image_or_url> \
  --output_dir ./inference_results \
  --device cuda \
  --precision bf16 \
  --input_size 256 \
  --out_dim 256
```

Outputs:
- Visualization: ./inference_results/<image_basename>_vis.jpg
- Text explanation and coordinates: ./inference_results/<image_basename>_res.txt

Brief pipeline:
1. Load tokenizer and base LLaVA model, apply Stage1 LoRA.
2. Load Stage1 and Stage2 checkpoints.
3. Preprocess input image for CLIP and model inputs.
4. Generate explanatory text and extract multi-modal features.
5. Stage1 proposes candidate boxes; Stage2 refines to final crop.
6. Map coordinates to original image, save visualization and text.

## 5. Acknowledgements
This project leverages and builds upon:
- LLaVA: https://github.com/haotian-liu/LLaVA
- Grid-Anchor-based-Image-Cropping-Pytorch (rod_align): https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch

## 6. Quick references
- Inference script: `inference.py`  
- Model code: `model/text_baseline.py`  
- LLaVA tools: `model/llava/`  
- rod_align: `rod_align/`
