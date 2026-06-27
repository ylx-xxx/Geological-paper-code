import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=7, stride=4, padding=3):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_)
            x_ = x_.reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_)
        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim,
        )
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape

        x = self.fc1(x)
        x = self.act(x)

        hidden_dim = x.shape[-1]
        x = x.transpose(1, 2).reshape(B, hidden_dim, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        sr_ratio=1,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(
            dim=dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = MixFFN(dim=dim, hidden_dim=hidden_dim, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.ffn(self.norm2(x), H, W)
        return x


class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        in_chans=14,
        variant="segformer_b1",
    ):
        super().__init__()

        if variant in ["segformer_b0", "mit_b0", "b0"]:
            embed_dims = [32, 64, 160, 256]
            depths = [2, 2, 2, 2]
            num_heads = [1, 2, 5, 8]
        elif variant in ["segformer_b1", "mit_b1", "b1"]:
            embed_dims = [64, 128, 320, 512]
            depths = [2, 2, 2, 2]
            num_heads = [1, 2, 5, 8]
        elif variant in ["segformer_b2", "mit_b2", "b2"]:
            embed_dims = [64, 128, 320, 512]
            depths = [3, 4, 6, 3]
            num_heads = [1, 2, 5, 8]
        else:
            raise ValueError(f"Unsupported SegFormer variant: {variant}")

        self.embed_dims = embed_dims
        self.depths = depths
        self.num_heads = num_heads
        self.sr_ratios = [8, 4, 2, 1]

        self.patch_embed1 = OverlapPatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            patch_size=7,
            stride=4,
            padding=3,
        )
        self.patch_embed2 = OverlapPatchEmbed(
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
            patch_size=3,
            stride=2,
            padding=1,
        )
        self.patch_embed3 = OverlapPatchEmbed(
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
            patch_size=3,
            stride=2,
            padding=1,
        )
        self.patch_embed4 = OverlapPatchEmbed(
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3],
            patch_size=3,
            stride=2,
            padding=1,
        )

        self.block1 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=4.0,
                sr_ratio=self.sr_ratios[0],
            )
            for _ in range(depths[0])
        ])
        self.block2 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=4.0,
                sr_ratio=self.sr_ratios[1],
            )
            for _ in range(depths[1])
        ])
        self.block3 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=4.0,
                sr_ratio=self.sr_ratios[2],
            )
            for _ in range(depths[2])
        ])
        self.block4 = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=4.0,
                sr_ratio=self.sr_ratios[3],
            )
            for _ in range(depths[3])
        ])

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def _tokens_to_feature(self, x, H, W):
        B, N, C = x.shape
        return x.transpose(1, 2).reshape(B, C, H, W).contiguous()

    def forward(self, x):
        outs = []

        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        c1 = self._tokens_to_feature(x, H, W)
        outs.append(c1)

        x, H, W = self.patch_embed2(c1)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        c2 = self._tokens_to_feature(x, H, W)
        outs.append(c2)

        x, H, W = self.patch_embed3(c2)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        c3 = self._tokens_to_feature(x, H, W)
        outs.append(c3)

        x, H, W = self.patch_embed4(c3)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        c4 = self._tokens_to_feature(x, H, W)
        outs.append(c4)

        return outs


class SegFormerDecoder(nn.Module):
    def __init__(self, in_channels, decoder_dim=256, num_classes=2, dropout=0.1):
        super().__init__()

        self.proj = nn.ModuleList([
            nn.Conv2d(c, decoder_dim, kernel_size=1, bias=False)
            for c in in_channels
        ])

        self.fuse = nn.Sequential(
            ConvBNReLU(decoder_dim * 4, decoder_dim, k=1, s=1, p=0),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_dim, num_classes, kernel_size=1),
        )

    def forward(self, feats, input_size):
        target_size = feats[0].shape[-2:]

        outs = []
        for f, proj in zip(feats, self.proj):
            x = proj(f)
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            outs.append(x)

        x = torch.cat(outs, dim=1)
        x = self.fuse(x)

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return x


class SwinUPerNetL4S(nn.Module):
    """
    Compatibility class name for existing train.py/test.py.

    Actual V7 model:
    Custom SegFormer-style network.
    """
    def __init__(
        self,
        num_classes=2,
        in_chans=14,
        backbone="segformer_b1",
        img_size=128,
        fpn_dim=256,
    ):
        super().__init__()

        if backbone in ["mit_b1"]:
            backbone = "segformer_b1"
        elif backbone in ["mit_b2"]:
            backbone = "segformer_b2"

        self.backbone = backbone
        self.in_chans = in_chans
        self.num_classes = num_classes

        print(f"[Info] V7 Custom SegFormer backbone: {backbone}")
        print(f"[Info] V7 input channels: {in_chans}")

        self.encoder = MixVisionTransformer(
            in_chans=in_chans,
            variant=backbone,
        )

        self.channels = self.encoder.embed_dims
        print(f"[Info] V7 encoder channels: {self.channels}")

        self.decoder = SegFormerDecoder(
            in_channels=self.channels,
            decoder_dim=fpn_dim,
            num_classes=num_classes,
            dropout=0.1,
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        out = self.decoder(feats, input_size=input_size)
        return out



def _adapt_first_patch_weight(old_w, new_w):
    """
    Adapt LoveDA RGB patch embedding to L4S 14-band patch embedding.
    old_w: [Cout, 3, k, k]
    new_w: [Cout, 14, k, k]
    """
    if old_w.ndim != 4 or new_w.ndim != 4:
        return None

    if old_w.shape[0] != new_w.shape[0] or old_w.shape[2:] != new_w.shape[2:]:
        return None

    out = new_w.clone()

    old_in = old_w.shape[1]
    new_in = new_w.shape[1]

    copy_ch = min(old_in, new_in)
    out[:, :copy_ch] = old_w[:, :copy_ch]

    if new_in > old_in:
        mean_w = old_w.mean(dim=1, keepdim=True)
        out[:, old_in:new_in] = mean_w.repeat(1, new_in - old_in, 1, 1)

    return out


def load_loveda_pretrained(model, ckpt_path):
    """
    Load SegFormer-B1 LoveDA checkpoint into L4S SegFormer-B1.
    3-channel patch embedding is adapted to 14-channel input.
    The final 7-class LoveDA segmentation head is skipped automatically.
    """
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[Warning] LoveDA SegFormer checkpoint not found: {ckpt_path}")
        return

    print(f"[Info] Loading SegFormer-LoveDA checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model_state = model.state_dict()

    converted = {}
    loaded = 0
    adapted = 0
    skipped = 0

    for k, v in state.items():
        nk = k

        if nk.startswith("module."):
            nk = nk[len("module."):]

        if nk not in model_state:
            skipped += 1
            continue

        if model_state[nk].shape == v.shape:
            converted[nk] = v
            loaded += 1
            continue

        if nk == "encoder.patch_embed1.proj.weight":
            new_v = _adapt_first_patch_weight(v, model_state[nk])
            if new_v is not None and new_v.shape == model_state[nk].shape:
                converted[nk] = new_v
                adapted += 1
                continue

        skipped += 1

    msg = model.load_state_dict(converted, strict=False)

    print("[Info] SegFormer-LoveDA transfer finished.")
    print(f"[Info] Direct loaded keys : {loaded}")
    print(f"[Info] Adapted keys       : {adapted}")
    print(f"[Info] Skipped keys       : {skipped}")
    print(f"[Info] Missing keys       : {len(msg.missing_keys)}")
    print(f"[Info] Unexpected keys    : {len(msg.unexpected_keys)}")

