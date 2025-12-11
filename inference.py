import os
import sys
import argparse
import torch
import numpy as np
import cv2
import requests
from io import BytesIO
from PIL import Image
import transformers
from transformers import CLIPImageProcessor
from torchvision import transforms
from peft import PeftModel

from model.text_baseline import LICAForCausalLM, Cropping_LICA, Instruct_Model
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Script")
    
    # 模型路径参数
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base LLaVA model")
    parser.add_argument("--vision_tower_path", type=str, default="openai/clip-vit-large-patch14", help="Path to CLIP vision tower")
    parser.add_argument("--stage1_lora_path", type=str, required=True, help="Path to Stage 1 LoRA adapter")
    parser.add_argument("--stage1_ckpt_path", type=str, required=True, help="Path to Stage 1 checkpoint (.pth)")
    parser.add_argument("--stage2_ckpt_path", type=str, required=True, help="Path to Stage 2 checkpoint (.pth)")
    
    # 输入输出参数
    parser.add_argument("--image_path", type=str, required=True, help="Path or URL to the input image")
    parser.add_argument("--prompt", type=str, default="Please output optimal cropping and explain why.focusing on subject prominence, balance, and harmony within the image. Keep it concise.", help="User prompt")
    parser.add_argument("--output_dir", type=str, default="./inference_results", help="Directory to save results")
    
    # 模型配置参数
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Model precision")
    parser.add_argument("--input_size", type=int, default=256, help="Input image size for resizing")
    parser.add_argument("--out_dim", type=int, default=256, help="Output dimension for cropping model")
    
    return parser.parse_args()

def get_dtype(precision):
    if precision == "bf16": return torch.bfloat16
    if precision == "fp16": return torch.half
    return torch.float32

def load_image(image_path):
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image

def setup_models(args, dtype, device):
    print(">>> Loading Tokenizer and Base Model...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model_path, model_max_length=2048, padding_side="right", use_fast=False
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[CRP]", special_tokens=True)
    crop_token_idx = tokenizer("[CRP]", add_special_tokens=False).input_ids[0]
    
    use_mm_start_end = True 
    if use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    model_args = {
        "train_mask_decoder": True,
        "out_dim": args.out_dim,
        "vision_tower": args.vision_tower_path,
        "use_mm_start_end": use_mm_start_end,
        "crop_token_idx": crop_token_idx
    }
    
    model = LICAForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=dtype, low_cpu_mem_usage=True, **model_args
    )
    model.resize_token_embeddings(len(tokenizer))
    model.get_model().initialize_vision_modules(model.get_model().config)
    
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=dtype, device=device)
    model.get_model().mm_projector.to(dtype=dtype, device=device)
    model.get_model().initialize_lisa_modules(model.get_model().config, dtype, device)
    
    print(f"    Applying LoRA from {args.stage1_lora_path}...")
    model = PeftModel.from_pretrained(model, args.stage1_lora_path)
    model.to(dtype=dtype, device=device).eval()

    print(">>> Loading Cropping Modules...")
    # Stage 1 Model
    model_cropping = Cropping_LICA(dtype, device)
    ckpt_s1 = torch.load(args.stage1_ckpt_path, map_location='cpu')
    s1_state = ckpt_s1.get('model_cropping_state_dict', ckpt_s1)
    model_cropping.load_state_dict(s1_state, strict=False)
    model_cropping.to(dtype=dtype, device=device).eval()
    
    # Stage 2 Model (Instruct Model)
    instruct_model = Instruct_Model(dtype, device)
    ckpt_s2 = torch.load(args.stage2_ckpt_path, map_location='cpu')
    if 'instruct_model_state_dict' in ckpt_s2: s2_state = ckpt_s2['instruct_model_state_dict']
    elif 'model_state_dict' in ckpt_s2: s2_state = ckpt_s2['model_state_dict']
    else: s2_state = ckpt_s2
    instruct_model.load_state_dict(s2_state, strict=False)
    instruct_model.to(dtype=dtype, device=device).eval()

    return tokenizer, model, model_cropping, instruct_model

def preprocess_for_inference(image, input_size, dtype, device):
    """
    Standard preprocessing pipeline: Resize -> ToTensor -> Normalize
    """
    im_width, im_height = image.size
    
    # 1. Resize
    w, h = input_size, input_size
    resized_image = image.resize((w, h), Image.Resampling.LANCZOS)
    
    # 2. Transform
    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    im_tensor = image_transformer(resized_image)
    images_tensor = im_tensor.unsqueeze(0).to(device, dtype=dtype) # [1, 3, 256, 256]
    
    return images_tensor, (w, h)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype = get_dtype(args.precision)

    # 1. Setup Models
    tokenizer, model, model_cropping, instruct_model = setup_models(args, dtype, device)
    clip_processor = CLIPImageProcessor.from_pretrained(args.vision_tower_path)

    # 2. Process Image
    print(f">>> Processing Image: {args.image_path}")
    raw_image = load_image(args.image_path)
    orig_w, orig_h = raw_image.size
    
    # Base Model Input (Resized to args.input_size, e.g., 256)
    images_tensor, (im_w, im_h) = preprocess_for_inference(raw_image, args.input_size, dtype, device)
    
    # CLIP Input (For text generation)
    images_clip_tensor = clip_processor(raw_image, return_tensors='pt')['pixel_values'].to(device, dtype=dtype)

    # 3. Construct Prompt
    qs = DEFAULT_IMAGE_TOKEN + "\n" + args.prompt
    # Assuming use_mm_start_end is True as in setup
    qs = DEFAULT_IM_START_TOKEN + qs + DEFAULT_IM_END_TOKEN
    qs = qs + "It is [CRP]."

    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_str = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_str, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

    # 4. Inference Pipeline
    print(">>> Running Inference...")
    with torch.no_grad():
        # A. Text Generation
        print("    [1/3] Generating Text...")
        generated = model.generate(
            inputs=input_ids,
            images=images_clip_tensor,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        
        output_ids = generated.sequences
        valid_ids = output_ids[0][output_ids[0] != -200] # Filter out padding if necessary
        text_output = tokenizer.decode(valid_ids, skip_special_tokens=True).strip()
        print(f"    --> Text Output: {text_output[:100]}...")

        # B. Feature Extraction
        print("    [2/3] Extracting Features...")
        resize_list = [[orig_w, orig_h]]
        input_dict = {
            "input_ids": input_ids,
            "images": images_tensor,
            "images_clip": images_clip_tensor,
            "attention_masks": torch.ones_like(input_ids).long().to(device),
            "offset": torch.tensor([0, 1]).long().to(device),
            "resize_list": resize_list,
            "inference": True
        }
        
        text_feature = model(**input_dict)
        if len(text_feature.shape) == 2: text_feature = text_feature.unsqueeze(1)
        elif len(text_feature.shape) == 1: text_feature = text_feature.unsqueeze(0).unsqueeze(0)

        # C. Box Prediction
        print("    [3/3] Predicting Optimal Crop...")
        dummy_crop = torch.zeros((1, 4)).to(device, dtype=dtype)
        
        stage1_boxes = model_cropping(images_tensor, text_feature, dummy_crop, inference=True, stage=2)
        final_boxes = instruct_model(images_tensor, stage1_boxes, dummy_crop, text_feature, inference=True)

    # 5. Post-processing & Visualization
    print(">>> Post-processing Coordinates...")
    crop = final_boxes.clone().float() # [1, 4]

    # Coordinate Transformation: Tensor Size -> Original Image Size
    # Formula: crop / tensor_size * original_size
    crop[:, 0::2] = crop[:, 0::2] / im_w * orig_w
    crop[:, 1::2] = crop[:, 1::2] / im_h * orig_h

    # Clip to image boundaries
    crop[:, 0::2] = torch.clamp(crop[:, 0::2], min=0, max=orig_w)
    crop[:, 1::2] = torch.clamp(crop[:, 1::2], min=0, max=orig_h)

    # Convert to numpy (int) for visualization
    s2_coords = crop[0].detach().cpu().numpy().astype(int)
    print(f"    --> Final Coords: {s2_coords}")

    # Visualization
    img_cv = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_cv, (s2_coords[0], s2_coords[1]), (s2_coords[2], s2_coords[3]), (0, 0, 255), 3)
    
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    out_vis_path = os.path.join(args.output_dir, f"{base_name}_vis.jpg")
    out_txt_path = os.path.join(args.output_dir, f"{base_name}_res.txt")
    
    cv2.imwrite(out_vis_path, img_cv)
    
    with open(out_txt_path, "w") as f:
        f.write(f"Prompt: {args.prompt}\n\nOutput Explanation: {text_output}\n\nOptimal Crop Coords (xyxy): {s2_coords.tolist()}")
    
    print(f"\n>>> Success! Visualization saved to: {out_vis_path}")

if __name__ == "__main__":
    main()
