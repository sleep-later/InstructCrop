import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.nn.init as init
import einops
import numpy as np
from torchvision.ops import roi_pool,roi_align
from .rod_align.modules.rod_align import RoDAlignAvg, RoDAlign
import math
from torchvision.ops import box_convert

def generate_anchors(anchor_stride):
    assert anchor_stride <= 16, 'not implement for anchor_stride{} > 16'.format(anchor_stride)
    P_h = np.array([2+i*4 for i in range(16 // anchor_stride)])
    P_w = np.array([2+i*4 for i in range(16 // anchor_stride)])

    num_anchors = len(P_h) * len(P_h)

    # initialize output anchors
    anchors = torch.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = P_w[j]
            anchors[k,0] = P_h[i]
            k += 1
    return anchors

def shift(shape, stride, anchors):
    shift_w = torch.arange(0, shape[0]) * stride
    shift_h = torch.arange(0, shape[1]) * stride
    shift_w, shift_h = torch.meshgrid([shift_w, shift_h])
    shifts  = torch.stack([shift_w, shift_h], dim=-1)  # h,w,2
    # add A anchors (A,2) to
    # shifts (h,w,2) to get
    # shift anchors (A,h,w,2)
    trans_anchors = einops.rearrange(anchors, 'a c -> a 1 1 c')
    trans_shifts  = einops.rearrange(shifts,  'h w c -> 1 h w c')
    all_anchors   = trans_anchors + trans_shifts
    return all_anchors

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

    def forward(self, x):
        '''
        :param x: b,512,H/16,W/16
        :return: b,4. anchor shifts of the best crop
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.output(x)
        return out
    
class PositionalEncoding2D(nn.Module):
    def __init__(self, channels: int, grid_size: int = 32):
        super().__init__()
        # We create learnable embeddings for height and width coordinates
        self.row_embed = nn.Embedding(grid_size, channels // 2)
        self.col_embed = nn.Embedding(grid_size, channels // 2)
        self.grid_size = grid_size

    def forward(self, boxes_norm):
        """
        Args:
            boxes_norm (Tensor): Normalized boxes (xyxy), shape [N, 4].
        
        Returns:
            Tensor: Positional embedding, shape [N, C, 1, 1].
        """
        # Convert xyxy to cxcywh to easily get center coordinates
        boxes_cxcywh = box_convert(boxes_norm, in_fmt='xyxy', out_fmt='cxcywh')
        
        # Quantize box centers to grid coordinates
        # Multiplying by (grid_size - 1) and rounding maps [0, 1] to {0, 1, ..., grid_size-1}
        cx = (boxes_cxcywh[:, 0] * (self.grid_size - 1)).round().long().clamp(0, self.grid_size - 1)
        cy = (boxes_cxcywh[:, 1] * (self.grid_size - 1)).round().long().clamp(0, self.grid_size - 1)
        
        # Get embeddings from coordinates
        x_embed = self.col_embed(cx)  # [N, C/2]
        y_embed = self.row_embed(cy)  # [N, C/2]
        
        # Concatenate and reshape to match feature map C dimension
        pos_encoding = torch.cat([x_embed, y_embed], dim=1)  # [N, C]
        # Reshape to [N, C, 1, 1] to be added to a feature map
        return pos_encoding.unsqueeze(-1).unsqueeze(-1)
    
class CropRegression(nn.Module):
    def __init__(self, anchor_stride, img_size):
        super(CropRegression, self).__init__()
        self.num_anchors = (16 // anchor_stride) ** 2

        anchors = generate_anchors(anchor_stride)
        feat_shape  = (img_size[0] // 16, img_size[1] // 16)
        all_anchors = shift(feat_shape, 16, anchors)
        all_anchors = all_anchors.float().unsqueeze(0)
        # 1,num_anchors,h//16,w//16,2

        all_anchors[..., 0] /= img_size[0]
        all_anchors[..., 1] /= img_size[1]

        self.upscale_factor = max(1, self.num_anchors // 2)
        anchors_x   = F.pixel_shuffle(
            all_anchors[...,0], upscale_factor=self.upscale_factor)
        anchors_y   = F.pixel_shuffle(
            all_anchors[...,1], upscale_factor=self.upscale_factor)
        # 1,h//s,w//s,2 where s=16//anchor_stride

        all_anchors = torch.stack([anchors_x, anchors_y], dim=-1).squeeze(1)
        self.register_buffer('all_anchors', all_anchors)
        
    def forward(self, offsets):
        '''
        :param offsets: b,num_anchors*4,h//16,w//16
        :return: b,4
        '''
        offsets = einops.rearrange(offsets, 'b (n c) h w -> b n h w c',
                                   n=self.num_anchors, c=4)
        coords  = [F.pixel_shuffle(
            offsets[...,i],
            upscale_factor=self.upscale_factor) for i in range(4)]
        # b, h//s, w//s, 4, where s=16//anchor_stride
        offsets = torch.stack(coords, dim=-1).squeeze(1)
        regression = torch.zeros_like(offsets) # b,h,w,4
        regression[...,0::2] = offsets[..., 0::2] + self.all_anchors[...,0:1]
        regression[...,1::2] = offsets[..., 1::2] + self.all_anchors[...,1:2]
        regression = einops.rearrange(regression, 'b h w c -> b (h w) c')
        return regression

def generate_anchors(anchor_stride):
    assert anchor_stride == 16
    # HACK: assume downsample 16x
    P_h = np.array([8])
    P_w = np.array([8])

    # assert anchor_stride <= 16, 'not implement for anchor_stride{} > 16'.format(anchor_stride)
    # P_h = np.array([2+i*4 for i in range(16 // anchor_stride)])
    # P_w = np.array([2+i*4 for i in range(16 // anchor_stride)])

    num_anchors = len(P_h) * len(P_h)

    # initialize output anchors
    anchors = torch.zeros((num_anchors, 2))
    k = 0
    for i in range(len(P_w)):
        for j in range(len(P_h)):
            anchors[k,1] = P_w[j]
            anchors[k,0] = P_h[i]
            k += 1
    return anchors


import torch
import torch.nn as nn
from torchvision.ops import roi_align

class CropScore(nn.Module):
    def __init__(self, img_size, feat_dim, text_dim=256, projection_dim=128):
        """
        Args:
            img_size (tuple): The size of the input image.
            feat_dim (int): The channel dimension of the input feature map 'f'.
            text_dim (int): The dimension of the input text feature.
            projection_dim (int): The dimension for both visual and text features to be projected into.
        """
        super().__init__()
        assert img_size[0] == img_size[1]
        self.img_size = img_size
        
        self.rod_align = RoDAlignAvg(5, 5, 1 / 16) 
        
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(feat_dim * 2, projection_dim, kernel_size=5),
            nn.ReLU(True),
            nn.Flatten(1),
        )
        
        self.film_generator = nn.Sequential(
            nn.Linear(text_dim, projection_dim),
            nn.ReLU(True),
            nn.Linear(projection_dim, 2 * projection_dim) 
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(projection_dim, 1) # 输出最终的匹配分数
        )

    def forward(self, xyxy, f, text_feature):
        """
        Args:
            xyxy (Tensor): Bounding box proposals, shape (B, M, 4).
            f (Tensor): Image feature map.
            text_feature (Tensor): Text prompt features, shape (B, 1, text_dim).
        """
        B, M, _ = xyxy.shape
        xyxy_scaled = xyxy.detach().clamp(min=0, max=1) * self.img_size[0]
        
        x1 = roi_align(
            f.float(), [xyxy_scaled[i].float() for i in range(B)], output_size=(5, 5),
            spatial_scale=f.shape[-1] / self.img_size[0])

        index = torch.arange(B).view(-1, 1, 1).repeat(1, M, 1).to(f.device)
        crop_xyxy_rod = torch.cat(
            [index, xyxy_scaled], dim=-1).reshape(-1, 5).contiguous()
        x2 = self.rod_align(f.float(), crop_xyxy_rod.float())

        image_crop_feature = torch.cat([x1, x2], dim=1)        
        visual_feature = self.visual_encoder(image_crop_feature.to(f.dtype))

 
        text_vector = text_feature.squeeze(1) 
        text_vector_expanded = text_vector.unsqueeze(1).expand(-1, M, -1) 
        text_vector_reshaped = text_vector_expanded.reshape(B * M, -1) 
        
        film_params = self.film_generator(text_vector_reshaped.to(f.dtype))
        gamma, beta = torch.chunk(film_params, 2, dim=-1) 
        
        modulated_feature = gamma * visual_feature + beta
        score = self.classifier(modulated_feature) 
        
        return score.reshape(B, M)

class GatedFeatureFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
                nn.Sigmoid()
            ) for _ in range(3)
        ])
        
        self.transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(3)
        ])
        
    def forward(self, features):
        transformed = [transform(feat) for transform, feat in zip(self.transforms, features)]
        gates = [gate(feat) for gate, feat in zip(self.gates, features)]
        
        # 归一化门控值
        gate_sum = sum(gates) + 1e-8
        normalized_gates = [gate / gate_sum for gate in gates]
        
        # 加权融合
        fused = sum([feat * gate for feat, gate in zip(transformed, normalized_gates)])
        return fused

class DynamicWeightFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 全局上下文编码器
        self.context_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1)  # 3个专家的权重
        )
        
    def forward(self, features):
        combined = torch.cat([f.unsqueeze(1) for f in features], dim=1)  # [B,3,256,16,16]
        weights = self.context_encoder(torch.mean(combined, dim=1))  # [B,3,1,1]
        weights = F.softmax(weights, dim=1)
        
        weighted_sum = torch.sum(combined * weights.unsqueeze(2), dim=1)  # [B,256,16,16]
        return weighted_sum
    
class CropTransformer(nn.Module):
    def __init__(self, anchor_stride, feat_dim, spatial_dim, text_dim,
                 positional_embedding=True):
        super().__init__()
        out_channel = int((16 / anchor_stride) ** 2 * 4)
        
        img_enc_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8)
        self.img_enc = nn.TransformerEncoder(img_enc_layer, num_layers=2)
        
        text_enc_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=8)
        self.text_enc = nn.TransformerEncoder(text_enc_layer, num_layers=2)
        self.cross_attn = nn.MultiheadAttention(feat_dim, num_heads=8, batch_first=True)
        if positional_embedding:
            self.pe = nn.Embedding(spatial_dim ** 2, feat_dim)
        else:
            self.pe = PositionalEncoding(
                feat_dim, dropout=0.1, max_len=spatial_dim ** 2)
        self.positional_embedding = positional_embedding
        
        self.text_proj = nn.Linear(text_dim, feat_dim)
         
        self.gate = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid()
        )
        
        self.crop = nn.Sequential(
            nn.Linear(feat_dim, out_channel),
            # nn.Sigmoid(),
        )
        # self.f_down = nn.Conv2d(256,32, 1)
        self.f_down = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1)
        )
        self.f3_adapter = nn.Conv2d(256, 256, 1)
        self.f4_adapter = nn.Conv2d(512, 256, 1)
        self.f5_adapter = nn.Conv2d(1024, 256, 1)

    def forward(self, text_feat):
        B = text_feat.shape[1]
        text_feat = text_feat.squeeze(0)  # (B, text_dim)
        text_feat = self.text_proj(text_feat)  # (B, feat_dim)
        text_feat = text_feat.unsqueeze(0)  # (1, B, feat_dim)7
        
        text_feat = text_feat.permute(1, 0, 2)  # (B, 1, feat_dim)
        text_feat = text_feat.expand(-1, 16*16, -1)  # (B, H*W, feat_dim)
        text_feat = text_feat.permute(1, 0, 2) # (H*W, B, feat_dim)

        if self.positional_embedding:
            position_ids = torch.arange(16*16, device=text_feat.device)
            position_embeddings = self.pe(position_ids).unsqueeze(1)  # [H*W, 1, feat_dim]
            text_feat = text_feat + position_embeddings
        else:
            text_feat = self.pe(text_feat)
        text_feat = self.text_enc(text_feat)  # (H*W, B, feat_dim)
        text_feat = text_feat.permute(1, 0, 2) # (B, H*W, feat_dim)
        
        D = text_feat.shape[-1]
        x = self.crop(text_feat.reshape(-1, D)).reshape(B, 16, 16, -1).permute(0, 3, 1, 2)
        return x