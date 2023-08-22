"""
Multimodal Adaptive Emotion Transformer with Flexible Modality Inputs on A Novel Dataset with Continuous Labels
The code is modified from https://github.com/microsoft/unilm/tree/master/vlmo
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial

from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))
        
        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        with_mixffn=False,
        layer_scale_init_values=0.1,
        max_text_len=40,
        prompt=False,
        prompt_len=2,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2_text = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_text = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_imag = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_vl = None
        if with_mixffn:
            self.mlp_vl = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.norm2_vl = norm_layer(dim)
        
        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)),requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        self.max_text_len = max_text_len

        self.prompt_len = prompt_len
        self.eeg_prompt = nn.Parameter(torch.zeros((1, prompt_len, dim)), requires_grad=True) if prompt is True else None
        self.eye_prompt = nn.Parameter(torch.zeros((1, prompt_len, dim)), requires_grad=True) if prompt is True else None
        if self.eeg_prompt is not None:
            trunc_normal_(self.eeg_prompt, std=0.02)
        if self.eye_prompt is not None:
            trunc_normal_(self.eye_prompt, std=0.02)


    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):
        if modality_type == "eeg" and self.eeg_prompt is not None:
            eeg_prompt = self.eeg_prompt.expand(x.size()[0], -1, -1) if self.eeg_prompt is not None else None
            x = torch.cat((x, eeg_prompt), dim=1)
        elif modality_type == "eye" and self.eye_prompt is not None:
            eye_prompt = self.eye_prompt.expand(x.size()[0], -1, -1) if self.eye_prompt is not None else None
            x = torch.cat((x, eye_prompt), dim=1)
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask, relative_position_bias=relative_position_bias))

        if modality_type == "eye":
            x = x + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x)))
        elif modality_type == "eeg":
            x = x + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x)))
        else:
            if self.mlp_vl is None:
                x_text = x[:, : self.max_text_len]
                x_imag = x[:, self.max_text_len :]
                x_text = x_text + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x_text)))
                x_imag = x_imag + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag)))
                x = torch.cat([x_text, x_imag], dim=1)
            else:
                x = x + self.drop_path(self.gamma_2 * self.mlp_vl(self.norm2_vl(x)))
        
        if modality_type == "eeg" and self.eeg_prompt is not None:
            x = x[:,:-self.prompt_len,:]
        elif modality_type == "eye" and self.eye_prompt is not None:
            x = x[:,:-self.prompt_len,:]
            
        return x


class MAET(nn.Module):
    def __init__(
        self,
        eeg_dim=310,
        eye_dim=33,
        num_classes=7,
        embed_dim=32,
        depth=3,
        eeg_seq_len=5,
        eye_seq_len=5,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        use_abs_pos_emb=False,
        layer_scale_init_values=0.1,
        mixffn_start_layer_index=2,
        use_mean_pooling=False,
        domain_generalization=False,
        num_domains=19,
        prompt=False,
        prompt_len=2,
    ):
        """
        Args:
            eeg_dim (int, tuple): input eeg feature size
            eye_dim (int, tuple): input eye feature size
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            need_relative_position_embed (bool): enable relative position bias on self-attention
            use_abs_pos_emb (bool): enable abs pos emb
            layer_scale_init_values (float or None): layer scale init values, set None to disable
            mixffn_start_layer_index (int): mixture-ffn start index
            domain_generalization (bool): whether to perform domain adversarial training for domain generalization
            num_domains (int): number of domains
            prompt (bool): whether to enable emotional prompt tuning
            prompt_len (int): number of prompt tokens
        """
        super().__init__()
        self.use_abs_pos_emb = use_abs_pos_emb
        self.prompt = prompt

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.eeg_transform = MultiViewEmbedding(eeg_dim, embed_dim, eeg_seq_len)
        self.eye_transform = MultiViewEmbedding(eye_dim, embed_dim, eye_seq_len)

        self.eeg_seq_len = eeg_seq_len
        self.eye_seq_len = eye_seq_len
        self.num_heads = num_heads
        self.mixffn_start_layer_index = mixffn_start_layer_index
        self.eeg_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eye_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eeg_pos_embed = nn.Parameter(torch.zeros(1, eeg_seq_len + 1, embed_dim)) if not self.use_abs_pos_emb else None
        self.eye_pos_embed = nn.Parameter(torch.zeros(1, eye_seq_len + 1, embed_dim)) if not self.use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.eeg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eye_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_mixffn=(i >= self.mixffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                    max_text_len=eeg_seq_len + 1,
                    prompt=prompt,
                    prompt_len=prompt_len
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fusion = Fusion(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_eeg = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_eye = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.domain_generalization = domain_generalization
        self.domain_classifier = None
        if domain_generalization:
            self.domain_classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_domains)
            )
        
        if self.eeg_pos_embed is not None:
            trunc_normal_(self.eeg_pos_embed, std=0.02)
        if self.eye_pos_embed is not None:
            trunc_normal_(self.eye_pos_embed, std=0.02)
        trunc_normal_(self.eeg_cls_token, std=0.02)
        trunc_normal_(self.eye_cls_token, std=0.02)
        trunc_normal_(self.eeg_type_embed, std=0.02)
        trunc_normal_(self.eye_type_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"eeg_pos_embed", "eye_pos_embed", "eeg_cls_token", "eye_cls_token", "eeg_type_embed", "eye_type_embed"}
    
    def forward_features(self, eeg, eye, alpha):
        if eeg is not None:
            eeg = self.eeg_transform(eeg)
            eeg_cls_tokens = self.eeg_cls_token.expand(eeg.size()[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if eye is not None:
            eye = self.eye_transform(eye)
            eye_cls_tokens = self.eye_cls_token.expand(eye.size()[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        if eye is None:
            x = torch.cat((eeg_cls_tokens, eeg), dim=1)
            modality_type = 'eeg'
            x = x + self.eeg_type_embed.expand(x.size()[0], x.size()[1], -1)
            if self.eeg_pos_embed is not None:
                x = x + self.eeg_pos_embed.expand(x.size()[0], -1, -1)
        elif eeg is None:
            x = torch.cat((eye_cls_tokens, eye), dim=1)
            modality_type = 'eye'
            x = x + self.eye_type_embed.expand(x.size()[0], x.size()[1], -1)
            if self.eye_pos_embed is not None:
                x = x + self.eye_pos_embed.expand(x.size()[0], -1, -1)
        else:
            x = torch.cat((eeg_cls_tokens, eeg, eye_cls_tokens, eye), dim=1)
            modality_type = None
            x = x + torch.cat([self.eeg_type_embed.expand(x.size()[0], self.eeg_seq_len + 1, -1), self.eye_type_embed.expand(x.size()[0], self.eye_seq_len + 1, -1)], dim=1)
            pos_embed = torch.cat([self.eeg_pos_embed.expand(x.size()[0], -1, -1), self.eye_pos_embed.expand(x.size()[0], -1, -1)], dim=1)
            x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, modality_type=modality_type)

        x = self.norm(x)

        domain_output = None
        if self.domain_generalization:
            reverse_x = ReverseLayerF.apply(x[:, 0], alpha)
            domain_output = self.domain_classifier(reverse_x)
        if self.fc_norm is not None:
            # return self.fc_norm(x[:, 1:].mean(1))
            return self.fc_norm(x.mean(1))
        else:
            if modality_type is not None:
                if self.prompt:
                    x = self.head_eeg(x[:, 1:].mean(dim=1)) if modality_type == 'eeg' else self.head_eye(x[:, 1:].mean(dim=1))
                else:
                    x = self.head_eeg(x[:, 1:].mean(dim=1)) if modality_type == 'eeg' else self.head_eye(x[:, 1:].mean(dim=1))
                return x, domain_output
            else:
                eeg_cls, eye_cls = x[:, 0], x[:, self.eeg_seq_len + 1]
                x = self.fusion(eeg_cls, eye_cls)
                x = self.head(x)
                return x, domain_output

    def forward(self, eeg=None, eye=None, alpha_=0):
        x, domain_output = self.forward_features(eeg, eye, alpha_)
        if domain_output is None:
            return x
        return x, domain_output


class MultiViewEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, heads):
        super().__init__()
        self.output_dim = output_dim
        self.heads = heads

        self.transform1 = nn.Linear(input_dim, output_dim)
        self.transform2 = nn.Linear(input_dim, output_dim * heads)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(output_dim)
    
    def forward(self, x):
        B, _ = x.size()
        x1 = self.transform1(x).unsqueeze(1).repeat(1, self.heads, 1)
        x2 = self.sigmoid(self.transform2(x)).reshape(B, self.heads, self.output_dim)

        x = torch.mul(x1, x2)
        x = self.bn(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class Fusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, 1))
        self.softmax = nn.Softmax(1)

    def forward(self, eeg, eye):
        o1 = eeg @ self.weight
        o2 = eye @ self.weight
        o = torch.cat([o1, o2], dim=-1)
        alpha = self.softmax(o)
        eeg = eeg * alpha[:, 0].unsqueeze(1)
        eye = eye * alpha[:, 1].unsqueeze(1)
        out = eeg + eye
        return out


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None