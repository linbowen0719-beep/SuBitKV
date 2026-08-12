from diffusers import AutoencoderKLWan
import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation
import diffusers.models.autoencoders.vae as diff_vae
import diffusers.models.autoencoders.autoencoder_kl_wan as diff_wan_vae

# from inference.gen_traj import generate_camera_trajectory_local

CHUNK_SIZE = 4
KS = [
    [
        [0.5051, 0.0000, 0.5000],
        [0.0000, 0.8979, 0.5000],
        [0.0000, 0.0000, 1.0000],
    ]
]


def _env_flag(name, default):
    return os.getenv(name, default).lower() not in {"0", "false", "no"}


def _optional_int_env(name):
    value = os.getenv(name)
    return int(value) if value else None


class MyVAE(AutoencoderKLWan):
    def decode(self, z, return_dict=False, is_first_chunk=False):
        if _env_flag("HYWORLDPLAY_ENABLE_VAE_TILING", "0"):
            self.enable_tiling(
                tile_sample_min_height=_optional_int_env("HYWORLDPLAY_VAE_TILE_HEIGHT"),
                tile_sample_min_width=_optional_int_env("HYWORLDPLAY_VAE_TILE_WIDTH"),
                tile_sample_stride_height=_optional_int_env("HYWORLDPLAY_VAE_TILE_STRIDE_HEIGHT"),
                tile_sample_stride_width=_optional_int_env("HYWORLDPLAY_VAE_TILE_STRIDE_WIDTH"),
            )
        return (self._decode(z, is_first_chunk=is_first_chunk).sample,)

    def _decode(self, z: torch.Tensor, return_dict: bool = True, is_first_chunk=False):
        _, _, num_frame, height, width = z.shape
        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )

        if self.use_tiling and (
            width > tile_latent_min_width or height > tile_latent_min_height
        ):
            return self.tiled_decode(z, return_dict=return_dict)

        # print('use my vae with is_first_chunk', is_first_chunk)
        if is_first_chunk:
            self.clear_cache()

        x = self.post_quant_conv(z)
        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk=is_first_chunk,
                )
            else:
                out_ = self.decoder(
                    x[:, :, i : i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                )
                out = torch.cat([out, out_], 2)

        if self.config.patch_size is not None:
            out = diff_wan_vae.unpatchify(out, patch_size=self.config.patch_size)

        out = torch.clamp(out, min=-1.0, max=1.0)

        # self.clear_cache()
        if not return_dict:
            return (out,)

        return diff_vae.DecoderOutput(sample=out)
