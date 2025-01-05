import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from functools import partial
import json

from model.swin_transformer import SwinTransformer, interpolate_relative_pos_embed
from model.xbert import BertConfig, BertForMaskedLM, BertModel
from model.xroberta import RobertaConfig, RobertaForMaskedLM, RobertaModel
from model.vit import VisionTransformer, interpolate_pos_embed
from model.clip_vit import CLIPVisionTransformer

def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor, handling non-distributed scenarios."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        if dist.is_available() and dist.is_initialized():
            output = [torch.empty_like(tensor) for _ in range(world_size)]
            dist.all_gather(output, tensor)
            ctx.rank = rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, 0)
        else:
            # 非分布式环境下，直接返回原始张量
            ctx.rank = 0
            ctx.batch_size = tensor.shape[0]
            return tensor

    @staticmethod
    def backward(ctx, grad_output):
        if dist.is_available() and dist.is_initialized():
            # 返回与当前进程相关的梯度部分
            return (
                grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
                None,
                None
            )
        else:
            # 非分布式环境下，梯度直接传递
            return grad_output, None, None

allgather = AllGather.apply


def build_vision_encoder(config, load_params=False):
    """
    Args:
        load_params: False when building fine-tuning models
    """
    num_patches = (config['image_res'] // config['patch_size']) ** 2

    if config['use_clip_vit']:  # good performance, but only base model available
        vision_config = read_json(config['vision_config'])
        assert config['patch_size'] == vision_config['patch_size']
        vision_width = vision_config['vision_width']

        vision_encoder = CLIPVisionTransformer(image_size=config['image_res'], patch_size=vision_config['patch_size'],
                                               hidden_size=vision_config['vision_width'],
                                               hidden_act=vision_config['hidden_act'],
                                               num_attention_heads=vision_config['num_attention_heads'],
                                               attention_dropout=vision_config['attention_dropout'],
                                               intermediate_size=vision_config['intermediate_size'],
                                               num_hidden_layers=vision_config['num_hidden_layers'],
                                               local_attn_depth=vision_config['local_attn_depth'])

        if load_params:
            # download from https://huggingface.co/openai/clip-vit-base-patch16/tree/main
            state_dict_orig = torch.load(vision_config['ckpt'], map_location="cpu")
            state_dict = {}
            for k, v in state_dict_orig.items():
                if k.startswith('vision_model.'):
                    k = k[13:]
                    if k.startswith('embeddings.'):
                        k = k[11:]
                        k = k.replace('patch_embedding.weight', 'patch_embed.weight')
                        k = k.replace('position_embedding.weight', 'pos_embed.weight')

                    if k != 'position_ids':
                        state_dict[k] = v

            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed.weight'].unsqueeze(dim=0),
                                                       num_patches=num_patches, num_extra_tokens=1)
            state_dict['pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

    elif config['use_swin']:
        vision_config = read_json(config['vision_config'])
        assert config['image_res'] == vision_config['image_res']
        assert config['patch_size'] == 32
        vision_width = vision_config['vision_width']

        vision_encoder = SwinTransformer(img_size=vision_config['image_res'],
                                         patch_size=4,
                                         in_chans=3,
                                         embed_dim=vision_config['embed_dim'],
                                         depths=vision_config['depths'],
                                         num_heads=vision_config['num_heads'],
                                         window_size=vision_config['window_size'],
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         drop_rate=0.0,
                                         drop_path_rate=0.1,
                                         ape=False,
                                         patch_norm=True,
                                         use_checkpoint=False)

        if load_params:
            # download from https://github.com/microsoft/Swin-Transformer
            state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

            for k in list(state_dict.keys()):
                if 'relative_position_bias_table' in k:
                    dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                    state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                elif ('relative_position_index' in k) or ('attn_mask' in k):
                    del state_dict[k]

    else:  # deit, worse than clip-vit/swin...
        assert config['patch_size'] == 16
        vision_width = 768

        vision_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=config['patch_size'], embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            local_attn_depth=4)

        if load_params:
            # download from https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
            state_dict = torch.load("data/deit_base_patch16_224-b5f2ef4d.pth", map_location="cpu")["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], num_patches=num_patches, num_extra_tokens=1)
            state_dict['pos_embed'] = pos_embed_reshaped

    if load_params:
        print("### Load ViT: ", flush=True)
        msg = vision_encoder.load_state_dict(state_dict, strict=False)
        print("missing_keys: ", msg.missing_keys)
        print("unexpected_keys: ", msg.unexpected_keys)

    return vision_encoder, vision_width


def build_text_encoder(config, vision_width, load_text_params=False, config_text=None):
    init_params = []  # train from scratch with larger lr

    if config_text is None:
        config_text = RobertaConfig.from_json_file(config['text_config']) \
            if config['use_roberta'] else BertConfig.from_json_file(config['text_config'])

    config_text.encoder_width = vision_width

    # for fine-tuning, not load_text_params by default
    assert load_text_params is False

    if config['use_roberta']:
        text_encoder = RobertaModel(config=config_text, add_pooling_layer=False)
    else:
        text_encoder = BertModel(config=config_text, add_pooling_layer=False)

    return text_encoder, init_params


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, input_dim * 2),
        nn.LayerNorm(input_dim * 2),
        nn.GELU(),
        nn.Linear(input_dim * 2, output_dim)
    )



def load_pretrained(ckpt_rpath, config, is_eval=False, load_text=False):
    checkpoint = torch.load(ckpt_rpath, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint

    if is_eval:
        return state_dict

    num_patches = (config['image_res'] // config['patch_size']) ** 2

    print("### Loading pretrained vision encoder", flush=True)
    if config['use_clip_vit']:
        del state_dict['vision_encoder.position_ids']
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed.weight'].unsqueeze(dim=0),
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed.weight'] = pos_embed_reshaped.squeeze(dim=0)

    elif config['use_swin']:

        window_size = read_json(config['vision_config'])['window_size']

        for k in list(state_dict.keys()):
            if 'relative_position_bias_table' in k:
                dst_num_pos = (2 * window_size - 1) ** 2
                state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
            elif ('relative_position_index' in k) or ('attn_mask' in k):
                del state_dict[k]

    else:
        pos_embed_reshaped = interpolate_pos_embed(state_dict['vision_encoder.pos_embed'],
                                                   num_patches=num_patches, num_extra_tokens=1)
        state_dict['vision_encoder.pos_embed'] = pos_embed_reshaped

    if load_text:
        print("### Loading pretrained text encoder", flush=True)
        for key in list(state_dict.keys()):
            if 'text_encoder.' in key:
                if config['use_roberta']:
                    if 'roberta.' in key:
                        encoder_key = key.replace('roberta.', '')
                        state_dict[encoder_key] = state_dict[key]
                        del state_dict[key]

                else:
                    if 'bert.' in key:
                        encoder_key = key.replace('bert.', '')
                        state_dict[encoder_key] = state_dict[key]
                        del state_dict[key]

    return state_dict


class XVLMBase(nn.Module):
    def __init__(self, config=None, load_vision_params=False, load_text_params=False,
                 use_contrastive_loss=False, use_matching_loss=False):
        """
        初始化XVLMBase类

        Args:
            config (dict): 配置字典
            load_vision_params (bool): 是否加载视觉编码器的参数
            load_text_params (bool): 是否加载文本编码器的参数
            use_contrastive_loss (bool): 是否使用对比损失
            use_matching_loss (bool): 是否使用匹配损失
            config_text (dict): 文本编码器的配置字典
        """
        super().__init__()
        self.init_params = []  # 从头开始训练的参数列表，使用较大的学习率

        # 构建视觉编码器，并获取其输出维度
        self.vision_encoder, vision_width = build_vision_encoder(config, load_params=load_vision_params)

        # 构建文本编码器，并获取初始化参数
        self.text_encoder, init_params = build_text_encoder(
            config,
            vision_width=vision_width,
            load_text_params=load_text_params,
            config_text=None
        )  # 文本和跨模态编码器
        self.init_params.extend(init_params)

        self.vision_width = vision_width  # 视觉编码器的输出宽度
        self.text_width = self.text_encoder.config.hidden_size  # 文本编码器的隐藏层大小，例如BERT的768

        # 如果使用对比损失，初始化相关层和参数
        if use_contrastive_loss:
            self.embed_dim = config['embed_dim']  # 嵌入维度，例如配置中的256
            self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)  # 视觉投影层
            self.text_proj = nn.Linear(self.text_width, self.embed_dim)  # 文本投影层
            self.init_params.extend(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
            self.init_params.extend(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])

            self.temp = nn.Parameter(torch.ones([]) * config['temp'])  # 温度参数
            self.init_params.extend(['temp'])

        # 如果使用匹配损失，初始化ITM头
        if use_matching_loss:
            self.itm_head = build_mlp(input_dim=self.text_width, output_dim=2)  # 构建多层感知机
            self.init_params.extend(['itm_head.' + n for n, _ in self.itm_head.named_parameters()])

    def load_pretrained(self, ckpt_rpath, config, is_eval=False):
        """
        加载预训练模型权重

        Args:
            ckpt_rpath (str): 检查点路径
            config (dict): 配置字典
            is_eval (bool): 是否为评估模式
        """
        state_dict = load_pretrained(ckpt_rpath, config, is_eval=is_eval, load_text=True)
        msg = self.load_state_dict(state_dict, strict=False)
        print('从 %s 加载检查点' % ckpt_rpath)
        print("缺失的键: ", [p for p in msg.missing_keys if 'vision_encoder' not in p])
        print("意外的键: ", msg.unexpected_keys)

    def get_vision_embeds(self, image):
        """
        输入为批次图像： [batch_size, channels, height, width]
        image_embeds的格式为：　[batch_size, num_patches, embed_dim]
        image_embeds[:, 0, :]是第一个patch，其包含全局信息
        输出为投影后的[batch_size, embed_dim]　－> [batch_size, proj_dim]
        图像输出1024维度
        """
        image_embeds = self.vision_encoder(image)
        return F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        # return F.normalize(image_embeds[:, 0, :], dim=-1)

    def get_text_embeds(self, text_ids, text_atts):
        """
        encoder_width: 1024
        """
        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder
        text_embeds = encoder(text_ids, attention_mask=text_atts, return_dict=True, mode='text').last_hidden_state
        # return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        return F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)


    def get_cross_embeds(self, image_embeds, image_atts, text_ids=None, text_embeds=None, text_atts=None):
        """
        获取跨模态嵌入

        Args:
            image_embeds (Tensor): 图像嵌入
            image_atts (Tensor): 图像的注意力掩码
            text_ids (Tensor, optional): 文本ID序列
            text_embeds (Tensor, optional): 文本嵌入
            text_atts (Tensor, optional): 文本的注意力掩码

        Returns:
            Tensor: 跨模态的最后隐藏状态
        """
        assert text_atts is not None

        encoder = self.text_encoder.bert if hasattr(self.text_encoder, 'bert') else self.text_encoder

        if text_embeds is not None:
            return encoder(
                encoder_embeds=text_embeds,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
                mode='fusion',
            ).last_hidden_state
        elif text_ids is not None:
            return encoder(
                text_ids,
                attention_mask=text_atts,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            ).last_hidden_state
        else:
            raise ValueError("必须提供text_ids或text_embeds")


    def get_contrastive_loss(self, image_feat, text_feat, idx=None):
        """
        计算对比损失

        Args:
            image_feat (Tensor): 图像特征，已归一化
            text_feat (Tensor): 文本特征，已归一化
            idx (Tensor, optional): 索引，用于匹配正样本

        Returns:
            Tensor: 对比损失
        """
        assert image_feat.size(-1) == self.embed_dim
        assert text_feat.size(-1) == self.embed_dim

        # 在所有进程之间收集特征
        image_feat_all = allgather(image_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        text_feat_all = allgather(text_feat, torch.distributed.get_rank(), torch.distributed.get_world_size())
        # 计算相似度矩阵
        logits = image_feat_all @ text_feat_all.t() / self.temp

        bsz = image_feat_all.shape[0]

        if idx is None:
            # 如果没有提供索引，假设一一对应
            labels = torch.arange(bsz, device=image_feat.device)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.t(), labels)
        else:
            # 使用提供的索引计算正样本
            idx = idx.view(-1, 1)
            assert idx.size(0) == image_feat.size(0)
            idx_all = allgather(idx, torch.distributed.get_rank(), torch.distributed.get_world_size())
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            labels = pos_idx / pos_idx.sum(1, keepdim=True)

            loss_i2t = -torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(logits.t(), dim=1) * labels, dim=1).mean()

        return (loss_i2t + loss_t2i) / 2

    def get_matching_loss(self, image_embeds, image_atts, image_feat, text_embeds, text_atts, text_feat, idx=None):
        """
        计算匹配损失，包含硬负样本

        Args:
            image_embeds (Tensor): 图像嵌入
            image_atts (Tensor): 图像的注意力掩码
            image_feat (Tensor): 图像特征
            text_embeds (Tensor): 文本嵌入
            text_atts (Tensor): 文本的注意力掩码
            text_feat (Tensor): 文本特征
            idx (Tensor, optional): 索引，用于匹配正样本

        Returns:
            Tensor: 匹配损失
        """
        bs = image_embeds.size(0)
        with torch.no_grad():
            # 计算相似度
            sim_i2t = image_feat @ text_feat.t() / self.temp
            sim_t2i = text_feat @ image_feat.t() / self.temp

            # 计算权重
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-5
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-5

            if idx is None:
                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
            else:
                idx = idx.view(-1, 1)
                assert idx.size(0) == bs
                mask = torch.eq(idx, idx.t())
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i.masked_fill_(mask, 0)

        # 采样负样本
        image_embeds_neg = []
        image_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
            image_atts_neg.append(image_atts[neg_idx])

        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)
        image_atts_neg = torch.stack(image_atts_neg, dim=0)

        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        # 拼接正负样本
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text_atts, text_atts_neg], dim=0)
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts_neg, image_atts], dim=0)

        # 获取跨模态嵌入
        cross_pos = self.get_cross_embeds(
            image_embeds,
            image_atts,
            text_embeds=text_embeds,
            text_atts=text_atts
        )[:, 0, :]
        cross_neg = self.get_cross_embeds(
            image_embeds_all,
            image_atts_all,
            text_embeds=text_embeds_all,
            text_atts=text_atts_all
        )[:, 0, :]

        # 通过ITM头获取输出
        output = self.itm_head(torch.cat([cross_pos, cross_neg], dim=0))
        itm_labels = torch.cat([
            torch.ones(bs, dtype=torch.long),
            torch.zeros(2 * bs, dtype=torch.long)
        ], dim=0).to(image_embeds.device)

        return F.cross_entropy(output, itm_labels)
