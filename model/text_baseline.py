from typing import List
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
import timm
from .text_baseline_utils import CropTransformer, CropRegression, CropScore, PositionalEncoding
import math

def custom_smooth_l1_loss(pred, target, beta=1.0):
    diff = torch.abs(pred - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()

# modified from torchvision to also return the union
def masked_loss(loss, n):
    mask = torch.ones_like(loss)
    mask[-n:] = 1e-10
    loss = (loss*mask).sum()/(mask.sum())
    return loss


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        scale=1000,  # 100000.0,
        eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def build_resnet(layers, pretrained=False):
    assert layers in [18, 34, 50, 101], f'layers must be one of [18, 34, 50, 101], while layers = {layers}'
    if layers == 18:
        resnet = models.resnet18(pretrained)
    elif layers == 34:
        resnet = models.resnet34(pretrained)
    elif layers == 50:
        resnet = models.resnet50(pretrained)
    else:
        resnet = models.resnet101(pretrained)
    modules = list(resnet.children())[:-2]
    resnet = nn.Sequential(*modules)
    return resnet

class LicaMetaModel:
    def __init__(
            self,
            config,
            **kwargs,
    ):
        super(LicaMetaModel, self).__init__(config)

        self.config = config
        self.config.out_dim = kwargs["out_dim"]

    def initialize_lisa_modules(self, config,dtype,device):

        torch_dtype = dtype
        device = device
    
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        in_dim = config.hidden_size
        out_dim = config.out_dim

        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            # nn.Dropout(0.5),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs = self.text_hidden_fcs.to(dtype=torch_dtype, device=device)
        self.text_hidden_fcs.train()
        self.text_hidden_fcs.apply(init_weights)

        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

class LicaModel(LicaMetaModel, LlavaLlamaModel):
    def __init__(
            self,
            config,
            **kwargs,
    ):
        super(LicaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LICAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
            self,
            config,
            **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.crop_loss_weight = kwargs.pop("crop_loss_weight", None)
            # self.com_loss_weight = kwargs.pop("com_loss_weight", None)
        else:
            config.mm_vision_tower = config.vision_tower

        self.crop_token_idx = kwargs.pop("crop_token_idx")

        super().__init__(config)

        self.model = LicaModel(config, **kwargs)
        self.post_init()

    def get_cropping_embs(self, im):
        return self.model.visual_model(im)

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
            self,
            images: torch.FloatTensor,
            images_clip: torch.FloatTensor,
            input_ids: torch.LongTensor,
            labels: torch.LongTensor,
            attention_masks: torch.LongTensor,
            offset: torch.LongTensor,
            resize_list: List[tuple],
            inference: bool = False,
            crop=None,
            **kwargs,
    ):
        batch_size = images.shape[0]
            
        crop_token_mask = input_ids[:, 1:] == self.crop_token_idx
        crop_token_mask = torch.cat(
            [
                crop_token_mask,
                torch.zeros((crop_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
        # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
        crop_token_mask = torch.cat(
            [torch.zeros((crop_token_mask.shape[0], 255)).bool().cuda(), crop_token_mask],
            dim=1,
        )

        if inference:
            # epoch = kwargs["epoch"]
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            output_list = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = output_list

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states
        
        
        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[crop_token_mask] 
        crop_token_counts = crop_token_mask.int().sum(-1)  # [bs, ]

        crop_token_offset = crop_token_counts.cumsum(-1)
        
        crop_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), crop_token_offset], dim=0
        )

        crop_token_offset = crop_token_offset[offset]  # for definiting the samples in the batch

        pred_embeddings_ = []
        for i in range(len(crop_token_offset) - 1):
            start_i, end_i = crop_token_offset[i], crop_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
        pred_embeddings = pred_embeddings_
        pred_embeddings = torch.stack(pred_embeddings, dim=1)
        if inference:
            return pred_embeddings
        model_output = output
        output = model_output.logits

        ce_loss = model_output.loss

        ce_loss = ce_loss * self.ce_loss_weight

        return pred_embeddings, ce_loss
    
from torchvision.ops import box_iou

def stable_iou_calculator(boxes1, boxes2):
    return box_iou(boxes1, boxes2)

# Helper Module for Positional Encoding
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int, max_h: int = 32, max_w: int = 32):
        super().__init__()
        self.h_embed = nn.Embedding(max_h, channels // 2)
        self.w_embed = nn.Embedding(max_w, channels // 2)
        
    def forward(self, boxes_norm, grid_size):
        """
        Args:
            boxes_norm (Tensor): Normalized boxes (cx, cy, w, h), shape [B*M, 4].
            grid_size (int): The assumed grid size of the feature map (e.g., 32).
        
        Returns:
            Tensor: Positional embedding, shape [B*M, C, 1, 1].
        """
        # Quantize box centers to a grid
        h_coords = (boxes_norm[:, 1] * (grid_size - 1)).round().long()
        w_coords = (boxes_norm[:, 0] * (grid_size - 1)).round().long()
        
        # Get embeddings
        h_pos = self.h_embed(h_coords) # [B*M, C/2]
        w_pos = self.w_embed(w_coords) # [B*M, C/2]
        
        # Concatenate and reshape to match feature map C dimension
        pos_encoding = torch.cat([h_pos, w_pos], dim=1) # [B*M, C]
        return pos_encoding.unsqueeze(-1).unsqueeze(-1) # [B*M, C, 1, 1]

class Instruct_Model(nn.Module):
    def __init__(self, torch_dtype,device,feat_dim=32):
        super(Instruct_Model,self).__init__()   
        self.score = CropScore(img_size=(256, 256), feat_dim=32).to(dtype=torch_dtype, device=device)
        self.crop_loss_weight = 20
        self.f1_adapters = nn.Conv2d(256, feat_dim, kernel_size=1).to(dtype=torch_dtype, device=device) 
        self.f2_adapters = nn.Conv2d(512, feat_dim, kernel_size=1).to(dtype=torch_dtype, device=device) 
        self.f3_adapters = nn.Conv2d(1024, feat_dim, kernel_size=1).to(dtype=torch_dtype, device=device)
        self.expert_models = nn.ModuleList()
        model_paths = [
            "/mnt/xiangfei/LISA/model/experts/composition_classify.pth",
            "/mnt/xiangfei/LISA/model/experts/aes_nextvit.pth"]
        
        for path in model_paths:
            expert = ExpertFeatureExtractor(expert_path=path).to(dtype=torch_dtype, device=device)
            self.expert_models.append(expert)
        
         # 创建特征适配器
        self.f1_adapters = nn.ModuleList([
            MultiScaleAdapter(256, feat_dim).to(dtype=torch_dtype, device=device) #3*3
            for _ in model_paths
        ])
        
        self.f2_adapters = nn.ModuleList([
            MultiScaleAdapter(512, feat_dim).to(dtype=torch_dtype, device=device) 
            for _ in model_paths
        ])
        
        self.f3_adapters = nn.ModuleList([
            MultiScaleAdapter(1024, feat_dim).to(dtype=torch_dtype, device=device) 
            for _ in model_paths
        ])
        self.fusion = FeatureFusion(feat_dim)
        
    def forward(self, images, boxes, crop,text_feat,inference=False):
        all_w = []  # 存储每个 expert model 的 w 分数
        B = images.shape
        for i, expert_model in enumerate(self.expert_models):
            with torch.no_grad():
                f = expert_model(images)

            f1 = f[1]
            f2 = f[2]
            f3 = f[3]

            # 应用特征适配器
            f1 = self.f1_adapters[i](f1)
            f2 = self.f2_adapters[i](f2)
            f3 = self.f3_adapters[i](f3)
            # 基于f2进行插值
            f1 = F.interpolate(f1.float(), size=(f2.shape[2], f2.shape[3]),
                            mode='bilinear', align_corners=False).to(f2.dtype)
            f3 = F.interpolate(f3.float(), size=(f2.shape[2], f2.shape[3]),
                            mode='bilinear', align_corners=False).to(f2.dtype)

            # current_f = f1+ f2 + f3 

            current_f = self.fusion(f1, f2, f3)
            current_w = self.score(boxes, current_f,text_feat) 
            # current_w = F.softmax(current_w, dim=1)
            all_w.append(current_w) # 存储当前 expert 的 w
            
        w = torch.mean(torch.stack(all_w), dim=0)
        # w = self.score(boxes,f)
        w = F.softmax(w,dim = 1)
        r = torch.sum(boxes * w.unsqueeze(-1), dim=1)
        crop = crop/256
        
        if inference:
            return r
        loss = custom_smooth_l1_loss(r.squeeze(), crop.squeeze(1))* self.crop_loss_weight
        return r, loss

# 替换简单的特征相加为注意力加权融合
class FeatureFusion(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(feat_dim*3, feat_dim, 1),
            nn.ReLU(),
            nn.Conv2d(feat_dim, 3, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, f1, f2, f3):
        fused = torch.cat([f1, f2, f3], dim=1)
        weights = self.attn(fused)
        return f1*weights[:,0:1] + f2*weights[:,1:2] + f3*weights[:,2:3]
       
class Cropping_LICA(nn.Module):
    def __init__(self, torch_dtype,device,positional_embedding=True,feat_dim=32,text_dim=256):
        super(Cropping_LICA, self).__init__()
        self.crop_regression = CropRegression(anchor_stride=16, img_size=(256, 256)).to(dtype=torch_dtype, device=device)

        # 初始化交叉注意力模块并设置类型和设备
        # 初始化Transformer解码器

        self.crop = CropTransformer(
            anchor_stride= 16, feat_dim=32,text_dim=256,
            spatial_dim=256 // 16).to(dtype=torch_dtype, device=device)
        # self.score = CropScore(img_size=(256, 256), feat_dim=32).to(dtype=torch_dtype, device=device)
        self.crop_loss_weight = 20
        # 文本编码器
        text_enc_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8)
        self.text_enc = nn.TransformerEncoder(text_enc_layer, num_layers=2)
        self.cross_attn = nn.MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        # 位置编码
        spatial_dim=256 // 16
        if positional_embedding:
            self.pe = nn.Embedding(spatial_dim ** 2, feat_dim)
        else:
            self.pe = PositionalEncoding(
                feat_dim, dropout=0.1, max_len=spatial_dim ** 2)
        self.positional_embedding = positional_embedding
        
        # 文本特征投影层，将文本特征投影到与图像特征相同的维度
        self.text_proj = nn.Linear(text_dim, feat_dim)
    def forward(self, images, text_embedding, crop, inference=False,stage=1):
        pred_boxes = []
        batch_size = images.shape[0]
        x = self.crop(text_embedding)
        boxes = self.crop_regression(x)
        if inference:
            if stage==2:
                return boxes
            w = torch.ones(1, 256, 1) / 256
        else:
            w = torch.ones(2, 256, 1) / 256
        w = w.to(images.device, dtype=torch.bfloat16)
        r = torch.sum(boxes * w, dim=1)
        
        for i in range(len(text_embedding.squeeze(0))):
            pred_boxes.append(r[i:i+1])

        if inference:
            return r.unsqueeze(0)

        crop_loss = 0
        for batch_idx in range(len(pred_boxes)):
            crop_box = crop[batch_idx] / 256
            pred_box = pred_boxes[batch_idx]
            per_boxes = boxes[batch_idx]
            if not pred_box.requires_grad:
                print("Warning: pred_box 不需要计算梯度！！！")

            assert (
                    crop_box.shape[0] == pred_box.shape[0]
            ), "gt_mos.shape: {}, pred_mask.shape: {}".format(
                crop_box.shape, pred_box.shape
            )
            if torch.isnan(pred_box).any() or torch.isinf(pred_box).any():
                print("Input contains NaN or Inf")
            # with torch.autograd.detect_anomaly():

            if pred_box.dtype == torch.bfloat16 or crop_box.dtype == torch.bfloat16:
                crop_loss += custom_smooth_l1_loss(pred_box.squeeze(), crop_box)* self.crop_loss_weight
                crop_loss += 0.1*custom_smooth_l1_loss(per_boxes,crop_box.repeat(per_boxes.shape[0],1))* self.crop_loss_weight
            else:
                crop_loss += (torch.nn.SmoothL1Loss(reduction='mean')(pred_box, crop.squeeze(1))) * self.crop_loss_weight

        
        return pred_box, crop_loss


class MultiScaleAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, channel_ratio=[1, 3, 3, 1]):
        super(MultiScaleAdapter, self).__init__()
        
        # 使用自定义通道分配比例
        total_ratio = sum(channel_ratio)
        
        # 计算每个分支的通道数
        branch1x1_channels = out_channels * channel_ratio[0] // total_ratio
        branch3x3_channels = out_channels * channel_ratio[1] // total_ratio
        branch5x5_channels = out_channels * channel_ratio[2] // total_ratio
        branch_pool_channels = out_channels - branch1x1_channels - branch3x3_channels - branch5x5_channels
        
        # 1x1 卷积分支
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch1x1_channels, kernel_size=1),
            nn.BatchNorm2d(branch1x1_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 卷积分支
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch3x3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch3x3_channels),
            nn.ReLU(inplace=True)
        )
        
        # 5x5 卷积分支
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch5x5_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch5x5_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 最大池化分支
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_pool_channels, kernel_size=1),
            nn.BatchNorm2d(branch_pool_channels),
            nn.ReLU(inplace=True)
        )
        
        # 末端1x1卷积融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        
    def forward(self, x):
        # 各分支前向计算
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        
        # 拼接所有分支的输出
        concatenated = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)
        
        # 应用末端融合层
        outputs = self.fusion(concatenated)
        
        # 应用SE注意力（可选）
        # outputs = self.se(outputs)
        
        return outputs

class AesModel(nn.Module):
    def __init__(self):
        super(AesModel, self).__init__()
        self.nextvit = timm.create_model(
            'nextvit_large.bd_in1k',
            pretrained=True,
            features_only=True,
        )
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1))
        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.nextvit(x)[-1]
        gap = self.GAP(x)
        logits = self.fc_layer(gap)
        # 分类层
        return logits
    
class ExpertFeatureExtractor(nn.Module):
    def __init__(self, expert_path):
        super().__init__()
        if "com" in expert_path:
            com = CompositionNextViT()
            state_dict = torch.load(expert_path)
            print('Loading composition cls model:', com.load_state_dict(state_dict, strict=True))
        if "aes" in expert_path:
            com = AesModel()
            state_dict = torch.load(expert_path)
            print('Loading aes model:', com.load_state_dict(state_dict, strict=True))
        self.backbone = com.nextvit
    def forward(self, x):
        return self.backbone(x)

class CompositionNextViT(nn.Module):
    def __init__(self, clip_model_name='nextvit_large.bd_in1k', num_classes=9):
        super(CompositionNextViT, self).__init__()
        self.comp_types = num_classes
        
        self.nextvit = timm.create_model(
            'nextvit_large.bd_in1k',
            pretrained=False,
            features_only=True,
        )
        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1))
        self.fc_layer = nn.Linear(1024, self.comp_types, bias=True)
        # self.fc_layer = nn.Linear(1024, self.comp_types, bias=True)

    def forward(self, x):
        x = self.nextvit(x)[-1]
        gap = self.GAP(x)
        logits = self.fc_layer(gap)
        # 分类层
        return logits

def bbox_iou_xyxy(pred_box, target_box, mode="ciou"):
    if not isinstance(pred_box, torch.Tensor):
        pred_box = torch.tensor(pred_box, dtype=torch.float32)
    if not isinstance(target_box, torch.Tensor):
        target_box = torch.tensor(target_box, dtype=torch.float32)
    
    if len(pred_box.shape) == 1:
        pred_box = pred_box.unsqueeze(0)
    if len(target_box.shape) == 1:
        target_box = target_box.unsqueeze(0)
        
    p_x1, p_y1, p_x2, p_y2 = pred_box[:, 0], pred_box[:, 1], pred_box[:, 2], pred_box[:, 3]
    t_x1, t_y1, t_x2, t_y2 = target_box[:, 0], target_box[:, 1], target_box[:, 2], target_box[:, 3]
    
    p_w, p_h = p_x2 - p_x1, p_y2 - p_y1
    t_w, t_h = t_x2 - t_x1, t_y2 - t_y1
    
    p_cx, p_cy = (p_x1 + p_x2) / 2, (p_y1 + p_y2) / 2
    t_cx, t_cy = (t_x1 + t_x2) / 2, (t_y1 + t_y2) / 2
    
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h
    
    p_area = p_w * p_h
    t_area = t_w * t_h
    
    union_area = p_area + t_area - inter_area
    
    iou = inter_area / (union_area + 1e-7)
    
    if mode == "iou":
        return iou
    
    enclose_x1 = torch.min(p_x1, t_x1)
    enclose_y1 = torch.min(p_y1, t_y1)
    enclose_x2 = torch.max(p_x2, t_x2)
    enclose_y2 = torch.max(p_y2, t_y2)
    
    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
    
    if mode == "giou":
        enclose_area = enclose_w * enclose_h
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
        return giou
    
    center_dist_squared = (p_cx - t_cx)**2 + (p_cy - t_cy)**2
    
    enclose_diagonal_squared = enclose_w**2 + enclose_h**2
    
    if mode == "diou":
        diou = iou - center_dist_squared / (enclose_diagonal_squared + 1e-7)
        return diou
    
    if mode == "ciou":
        p_aspect_ratio = torch.atan(p_w / (p_h + 1e-7))
        t_aspect_ratio = torch.atan(t_w / (t_h + 1e-7))
        
        v = (4 / (math.pi**2)) * ((p_aspect_ratio - t_aspect_ratio)**2)
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-7)
        
        ciou = iou - (center_dist_squared / (enclose_diagonal_squared + 1e-7) + alpha * v)
        return ciou
    
    return iou

def ciou_loss(pred_box, target_box):
    """
    计算CIoU损失
    """
    ciou = bbox_iou_xyxy(pred_box, target_box, mode="ciou")
    return 1 - ciou
