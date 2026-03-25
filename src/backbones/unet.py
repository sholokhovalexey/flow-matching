from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import UNet2DModel as UNet


class UNet2DModelCustom(UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        time_embed_dim = self.config.time_embedding_dim or self.config.block_out_channels[0] * 4
        self.time_emb_cat = nn.Linear(2 * time_embed_dim, time_embed_dim)

        class_embed_type = self.config.class_embed_type
        num_class_embeds = self.config.num_class_embeds
        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            # add extra "null" class for CFG
            self.class_embedding = nn.Embedding(num_class_embeds + 1, time_embed_dim) # num_classes + 1

    def forward(self, sample, t, r=None, cond=None):

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        time_t_emb = self.time_proj(t).to(dtype=self.dtype)
        emb = self.time_embedding(time_t_emb)

        if self.class_embedding is not None:
            if cond is None:
                raise ValueError("cond should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                cond = self.time_proj(cond)

            class_emb = self.class_embedding(cond).to(dtype=self.dtype)
            emb = emb + class_emb

        elif self.class_embedding is None and cond is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if r is not None:
            time_r_emb = self.time_proj(r).to(dtype=self.dtype)
            r_emb = self.time_embedding(time_r_emb)
            emb = self.time_emb_cat(torch.cat([emb, r_emb], -1))

        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        return sample
    

def UNet2DModel(n_blocks=4, n_channels=32, layers_per_block=2, add_attention=False, **kwargs):
    block_out_channels = [n_channels * (k+1) for k in range(n_blocks)]
    if add_attention:
        down_block_types = ["DownBlock2D"] + ["AttnDownBlock2D"] * (n_blocks-1)
        up_block_types = ["AttnUpBlock2D"] * (n_blocks-1) + ["UpBlock2D"]
    else:
        down_block_types = ["DownBlock2D"] * n_blocks
        up_block_types = ["UpBlock2D"] * n_blocks
    return UNet2DModelCustom(
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        down_block_types=down_block_types, 
        up_block_types=up_block_types, 
        add_attention=add_attention, 
        **kwargs,
    )

