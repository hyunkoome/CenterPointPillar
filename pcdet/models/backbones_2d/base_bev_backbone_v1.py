import torch
import numpy as np
class BaseBEVBackboneV1(torch.nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = torch.nn.ModuleList()
        self.deblocks = torch.nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                torch.nn.ZeroPad2d(1),
                torch.nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                torch.nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                torch.nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    torch.nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    torch.nn.ReLU()
                ])
            self.blocks.append(torch.nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(torch.nn.Sequential(
                        torch.nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        torch.nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        torch.nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(torch.nn.Sequential(
                        torch.nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        torch.nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        torch.nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(torch.nn.Sequential(
                torch.nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                torch.nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                torch.nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict