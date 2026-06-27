
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SwinUPerNet(nn.Module):
    def __init__(
        self,
        num_classes=7,
        backbone="swin_tiny_patch4_window7_224.ms_in1k",
        pretrained=False,
        pretrained_path="/root/autodl-tmp/pre_model/model.safetensors",
        img_size=512,
        fpn_dim=256,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            backbone,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
        )

        self.channels = self.encoder.feature_info.channels()
        print(f"[Info] Encoder channels: {self.channels}")

        if pretrained:
            self.load_local_pretrained(pretrained_path)

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

        self.fuse = nn.Sequential(
            ConvBNReLU(fpn_dim * 4, fpn_dim),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, num_classes, 1),
        )

    def _clean_key(self, key):
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        if key.startswith("encoder."):
            key = key[len("encoder."):]
        return key

    def _convert_swin_key_for_featurelist(self, key):
        key = self._clean_key(key)

        # 去掉分类头权重，分割任务不需要
        if key.startswith("head."):
            return None
        if key.startswith("norm."):
            return None

        # timm FeatureListNet 中 layers.0 会变成 layers_0
        for i in range(4):
            key = key.replace(f"layers.{i}.", f"layers_{i}.")

        return key

    def load_local_pretrained(self, pretrained_path):
        if pretrained_path is None or not os.path.exists(pretrained_path):
            print(f"[Warning] Pretrained file not found: {pretrained_path}")
            print("[Warning] Use random initialization instead.")
            return

        print(f"[Info] Loading local pretrained weights: {pretrained_path}")

        if pretrained_path.endswith(".safetensors"):
            from safetensors.torch import load_file
            raw_state = load_file(pretrained_path)
        else:
            raw_state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(raw_state, dict):
                if "model" in raw_state:
                    raw_state = raw_state["model"]
                elif "state_dict" in raw_state:
                    raw_state = raw_state["state_dict"]

        encoder_state = self.encoder.state_dict()
        converted = {}

        for k, v in raw_state.items():
            nk = self._convert_swin_key_for_featurelist(k)
            if nk is None:
                continue

            if nk in encoder_state and encoder_state[nk].shape == v.shape:
                converted[nk] = v

        msg = self.encoder.load_state_dict(converted, strict=False)

        print("[Info] Local pretrained loaded.")
        print(f"[Info] Matched keys: {len(converted)}")
        print(f"[Info] Missing keys: {len(msg.missing_keys)}")
        print(f"[Info] Unexpected keys: {len(msg.unexpected_keys)}")

        if len(converted) < 100:
            print("[Warning] Matched keys are too few. Please check whether the weight file matches swin_tiny_patch4_window7_224.ms_in1k.")
        else:
            print("[Info] Pretrained weight mapping looks good.")

    def _to_nchw(self, feats):
        nchw_feats = []

        for i, (f, c) in enumerate(zip(feats, self.channels)):
            if f.ndim != 4:
                raise RuntimeError(f"Feature {i} should be 4D, but got shape {tuple(f.shape)}")

            if f.shape[1] == c:
                nchw_feats.append(f.contiguous())
                continue

            if f.shape[-1] == c:
                nchw_feats.append(f.permute(0, 3, 1, 2).contiguous())
                continue

            raise RuntimeError(
                f"Cannot infer layout for feature {i}. "
                f"Expected channel={c}, got shape={tuple(f.shape)}"
            )

        return nchw_feats

    def forward(self, x):
        input_size = x.shape[-2:]

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
