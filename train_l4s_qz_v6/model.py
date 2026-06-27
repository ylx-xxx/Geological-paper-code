import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TopographyBranch(nn.Module):
    """
    Terrain branch for slope + DEM.
    Input: last 2 channels of Landslide4Sense, assumed to be slope and DEM.
    Output: multi-scale terrain features projected to FPN dimension.
    """
    def __init__(self, in_ch=2, out_ch=256):
        super().__init__()

        mid = out_ch // 2

        self.stem = nn.Sequential(
            ConvBNReLU(in_ch, mid, k=3, s=1, p=1),
            ConvBNReLU(mid, out_ch, k=3, s=1, p=1),
        )

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            for _ in range(4)
        ])

        # Start from weak terrain injection to avoid destroying V1 features.
        self.scales = nn.Parameter(torch.ones(4) * 0.10)

    def forward(self, topo, target_sizes):
        base = self.stem(topo)

        outs = []
        for i, size in enumerate(target_sizes):
            t = F.interpolate(
                base,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
            gate = self.gates[i](t)
            outs.append(self.scales[i] * gate * t)

        return outs


class SwinUPerNetL4S(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_chans=14,
        backbone="swin_tiny_patch4_window7_224.ms_in1k",
        img_size=128,
        fpn_dim=256,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.fpn_dim = fpn_dim

        self.encoder = timm.create_model(
            backbone,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
            in_chans=in_chans,
        )

        self.channels = self.encoder.feature_info.channels()
        print(f"[Info] L4S encoder channels: {self.channels}")
        print("[Info] V6 terrain branch enabled: slope/DEM channels -> FPN fusion")

        self.lateral = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, 1) for c in self.channels
        ])

        self.smooth = nn.ModuleList([
            ConvBNReLU(fpn_dim, fpn_dim) for _ in self.channels
        ])

        self.ppm = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(self.channels[-1], fpn_dim, 1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
            )
            for s in [1, 2, 3, 6]
        ])

        self.ppm_bottleneck = ConvBNReLU(fpn_dim * 4 + self.channels[-1], fpn_dim)

        self.topo_branch = TopographyBranch(in_ch=2, out_ch=fpn_dim)

        self.fuse = nn.Sequential(
            ConvBNReLU(fpn_dim * 4, fpn_dim),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

    def _to_nchw(self, feats):
        out = []
        for i, (f, c) in enumerate(zip(feats, self.channels)):
            if f.ndim != 4:
                raise RuntimeError(f"Feature {i} should be 4D, got {tuple(f.shape)}")

            if f.shape[1] == c:
                out.append(f.contiguous())
            elif f.shape[-1] == c:
                out.append(f.permute(0, 3, 1, 2).contiguous())
            else:
                raise RuntimeError(
                    f"Cannot infer feature layout for feature {i}, "
                    f"expected C={c}, got shape={tuple(f.shape)}"
                )
        return out

    def forward(self, x):
        input_size = x.shape[-2:]

        if x.shape[1] < 14:
            raise RuntimeError(f"V6 expects 14-channel input, got {x.shape[1]} channels")

        # Landslide4Sense: channels 12 and 13 are slope and DEM.
        topo = x[:, 12:14, :, :]

        feats = self.encoder(x)
        feats = self._to_nchw(feats)

        c1, c2, c3, c4 = feats

        ppm_outs = [c4]
        for ppm in self.ppm:
            y = ppm(c4)
            y = F.interpolate(
                y,
                size=c4.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            ppm_outs.append(y)

        p4 = self.ppm_bottleneck(torch.cat(ppm_outs, dim=1))

        laterals = [
            self.lateral[0](c1),
            self.lateral[1](c2),
            self.lateral[2](c3),
            p4,
        ]

        target_sizes = [lat.shape[-2:] for lat in laterals]
        topo_feats = self.topo_branch(topo, target_sizes)

        laterals = [
            lat + topo_feat
            for lat, topo_feat in zip(laterals, topo_feats)
        ]

        for i in range(3, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        outs = [self.smooth[i](laterals[i]) for i in range(4)]

        target_size = outs[0].shape[-2:]
        outs = [
            F.interpolate(
                o,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            for o in outs
        ]

        out = self.fuse(torch.cat(outs, dim=1))
        out = F.interpolate(
            out,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return out


def adapt_patch_embed_weight(old_w, new_w):
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
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print(f"[Warning] LoveDA checkpoint not found: {ckpt_path}")
        return

    print(f"[Info] Loading LoveDA checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

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

        if "patch_embed.proj.weight" in nk:
            new_v = adapt_patch_embed_weight(v, model_state[nk])
            if new_v is not None and new_v.shape == model_state[nk].shape:
                converted[nk] = new_v
                adapted += 1
                continue

        skipped += 1

    msg = model.load_state_dict(converted, strict=False)

    print("[Info] LoveDA pretrained transfer finished.")
    print(f"[Info] Direct loaded keys : {loaded}")
    print(f"[Info] Adapted keys       : {adapted}")
    print(f"[Info] Skipped keys       : {skipped}")
    print(f"[Info] Missing keys       : {len(msg.missing_keys)}")
    print(f"[Info] Unexpected keys    : {len(msg.unexpected_keys)}")
