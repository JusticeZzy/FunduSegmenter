"""
MIT License

Copyright (c) 2021 Robin Strudel
Copyright (c) INRIA

Partly revised from https://github.com/rstrudel/segmenter
by Zhenyi Zhao @Vampire,Computing,University of Dundee
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FunduSegmenter(nn.Module):
    def __init__(
        self,
        pre_adapter,
        encoder,
        decoder,
        post_adapter,
    ):
        super().__init__()
        self.patch_size = encoder.patch_size
        self.pre_adapter = pre_adapter
        self.encoder = encoder
        self.decoder = decoder
        self.post_adapter = post_adapter

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        
        im = self.pre_adapter(im)          
        H, W = im.size(2), im.size(3)

        x, mid_features = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        mid_features[0] = mid_features[0][:, num_extra_tokens:]
        mid_features[1] = mid_features[1][:, num_extra_tokens:]
        mid_features[2] = mid_features[2][:, num_extra_tokens:]
        mid_features[3] = mid_features[3][:, num_extra_tokens:]
        
        x = self.decoder(x, (H, W))

        masks = self.post_adapter(x, mid_features, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)