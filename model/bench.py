import torch
from torch import nn

from model.xvlm import XVLMBase, load_pretrained


class Bench(XVLMBase):
    def __init__(self, config):
        """
        初始化Bench类，继承自XVLMBase
        """
        super().__init__(
            config,
            load_vision_params=False,
            load_text_params=False,
            use_contrastive_loss=True,
            use_matching_loss=True,
        )
        self.vision_proj = nn.Linear(1024, 768)  # 修改输出维度为 768
        self.text_proj = nn.Linear(768, 768)  # 修改输入和输出维度为 768

        # 可选：显式初始化新添加的层
        nn.init.xavier_uniform_(self.vision_proj.weight)
        nn.init.constant_(self.vision_proj.bias, 0)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.constant_(self.text_proj.bias, 0)

        # 初始化基类中的其他参数
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def load_pretrained(self, checkpoint_path, config, is_eval=False):
        # 加载检查点
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 过滤掉不匹配的层
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in self.state_dict():
                if self.state_dict()[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    print(
                        f"Skipping loading parameter '{k}' due to shape mismatch: {v.shape} vs {self.state_dict()[k].shape}")
            else:
                print(f"Skipping loading parameter '{k}' as it is not present in the current model.")

        # 加载过滤后的 state_dict
        msg = self.load_state_dict(filtered_state_dict, strict=False)
        print(msg)

        # 手动初始化不匹配的层
        # 假设您使用的是 nn.Linear 层
        if 'vision_proj.weight' not in filtered_state_dict:
            nn.init.xavier_uniform_(self.vision_proj.weight)
        if 'vision_proj.bias' not in filtered_state_dict:
            nn.init.constant_(self.vision_proj.bias, 0)
        if 'text_proj.weight' not in filtered_state_dict:
            nn.init.xavier_uniform_(self.text_proj.weight)
        if 'text_proj.bias' not in filtered_state_dict:
            nn.init.constant_(self.text_proj.bias, 0)


    def forward(self, image, desc_token):
        """
        前向传播函数
        """
        """
              前向传播函数
              返回:
                  image_embeds: [batch_size, 768] 的视觉特征
                  desc_embeds:  一个字典, key 为 "desc_embeds_{desc_key_clean}", 
                                value 为 当前 GPU 上拼接后的 [local_count, 768] 的张量
              """

        # 1. 获取图像嵌入
        image_embeds = self.get_vision_embeds(image)

        # 2. 用一个字典收集该 GPU 上所有描述的嵌入, 避免在 forward 中频繁 cat
        #    下面用 "desc_embeds_local" 来存储每个 desc_key_clean 对应的若干条 embedding

        # key_list = []  # 初始化 key_list
        # desc_tokens = []
        # desc_embeds= {}

        desc_embeds_local = {}
        for i in range(len(desc_token)):
            desc_text_dict = desc_token[i]
            text_embeds = {}
            for key, descriptions in desc_text_dict[0].items():
                text_embeds[key] = {}  # 初始化每个图像的存储字典
                for desc_key, tokens in descriptions.items():

                    # 获取当前描述的 input_ids 和 attention_mask
                    text_ids = tokens["input_ids"]
                    text_atts = tokens["attention_mask"] #[:512]         # 截断到512个标记

                    # 调用 get_text_embeds 并存储结果
                    text_embed = self.get_text_embeds(text_ids, text_atts)

                    # print(f"当前处理的描述键: {desc_key}")
                    desc_key_clean = desc_key.replace(" ", "_")  # 替换空格
                    # 3.4 收集到 desc_embeds_local
                    if desc_key_clean not in desc_embeds_local:
                        desc_embeds_local[desc_key_clean] = [text_embed]
                    else:
                        desc_embeds_local[desc_key_clean].append(text_embed)

            # 4. 在当前 GPU 上, 把相同 desc_key_clean 的列表拼成 [m, 768]
            #    注意: 这里 m 是该 GPU 上的此 desc_key_clean 的条数, 不是全局 batch
            #    DataParallel 会自动在第 0 维进行 gather, 让最终得到 [total_m, 768]
        desc_embeds_dict = {}
        for desc_key_clean, embed_list in desc_embeds_local.items():
            desc_embeds_dict[f"{desc_key_clean}"] = torch.cat(embed_list, dim=0)

            #         if desc_key_clean not in key_list:
            #             key_list.append(desc_key_clean )
            #             desc_embeds[f"{desc_key_clean}"] = text_embed.unsqueeze(0)
            #             # print(desc_embeds)
            #         else:
            #             desc_embeds[f"{desc_key_clean}"] = torch.cat(
            #                 (desc_embeds[f"{desc_key_clean}"], text_embed.unsqueeze(0).to(device)), dim=0
            #             )
            #
            #         text_embeds[key][desc_key_clean ] = text_embed  # 存储到对应位置
            # desc_tokens.append([text_embeds])

        return image_embeds, desc_embeds_dict


