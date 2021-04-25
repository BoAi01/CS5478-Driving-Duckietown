import sys

sys.path.insert(0, '/home/aibo/duck/gym-duckietown/learning')

import torch
import torch.nn as nn

from imitation.basic.resnet import resnet50
import math
import os

norm = lambda in_c: nn.GroupNorm(num_groups=32, num_channels=in_c)


def load_model_dic(model, ckpt_path, verbose=True, strict=True):
    """
    Load weights to encoder and take care of weight parallelism
    """
    assert os.path.exists(ckpt_path), f"trained encoder {ckpt_path} does not exist"

    try:
        model.load_state_dict(torch.load(ckpt_path), strict=strict)
    except:
        state_dict = torch.load(ckpt_path)
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=strict)
    if verbose:
        print(f'Model loaded: {ckpt_path}')

    return model


class ConvLSTMCellPeep(nn.Module):
    def __init__(self, in_size, in_c, hid_c, dropout_keep, kernel_size=3):
        super().__init__()
        self.height, self.width = in_size
        self.in_size = in_size
        self.in_c = in_c
        self.hid_c = hid_c
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_c + self.hid_c * 2, 2 * self.hid_c, kernel_size, 1, padding=kernel_size // 2),
            norm(2 * self.hid_c)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_c + self.hid_c * 2, self.hid_c, kernel_size, 1, padding=kernel_size // 2),
            norm(self.hid_c)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.in_c + self.hid_c, self.hid_c, kernel_size, 1, padding=kernel_size // 2),
            norm(self.hid_c)
        )
        self.dropout = nn.Dropout2d(p=1 - dropout_keep)  # dp2d instead of dropout for 2d feature map (Ma Xiao)
        print(f'ConvLSTMCell with peephole in_size = {in_size}, dropout {dropout_keep}')

    def forward(self, ins, seq_len, prev=None):

        if prev is None:
            h, c = self.init_states(ins.size(0))  # ins and prev not None at the same time (guaranteed)
        else:
            # print(f'len of prev {len(prev)}, type {type(prev)}')
            h, c = prev

        hs, cs = [], []  # store all intermediate h and c
        for i in range(seq_len):
            # prepare x: create one zero tensor if x is None (decoder mode)
            if ins is not None:
                x = ins[:, i]
            else:
                x = torch.zeros(h.size(0), self.in_c, self.height, self.width).cuda()

            x = self.dropout(x)  # conventional forward dropout
            h = self.dropout(h)  # variational inference based dropout (Gao, Y., et al. 2016)

            # f, i gates
            combined_conv = self.conv1(torch.cat([x, h, c], dim=1))

            i_t, f_t = torch.split(combined_conv, self.hid_c, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)

            # g gate
            g_t = self.conv3(torch.cat([x, h], dim=1))
            g_t = torch.tanh(g_t)

            # update cell state
            c_t = f_t * c + i_t * self.dropout(g_t)  # recurrent dropout (Semeniuta, S., et al., 2016)

            # o gate
            o_t = self.conv2(torch.cat([x, h, c_t], dim=1))
            o_t = torch.sigmoid(o_t)

            h_t = o_t * torch.tanh(c_t)

            h, c = h_t, c_t

            hs.append(h)
            cs.append(c)

        return hs, cs

    def init_states(self, batch_size):
        states = (torch.zeros(batch_size, self.hid_c, self.height, self.width),
                  torch.zeros(batch_size, self.hid_c, self.height, self.width))
        states = (states[0].cuda(), states[1].cuda())
        return states


class BottleneckLSTMCell(nn.Module):
    def __init__(self, in_size, in_c, hid_c, dropout_keep, out_size):
        super().__init__()
        self.hid_c = hid_c
        self.out_size = out_size

        self.cell = ConvLSTMCellPeep(in_size, in_c, hid_c, dropout_keep)
        self.downsample = nn.Sequential(
            nn.Conv2d(out_size, in_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU(True)
        )
        self.upsample = nn.Sequential(
            nn.Conv2d(in_c, out_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(True)
        )

    def forward(self, x, prev):
        bs, t, c, h, w = x.shape
        x = x.view(bs * t, c, h, w)
        x = self.downsample(x)
        x = x.view(bs, t, self.hid_c, h, w)
        x, cs = self.cell(x, t, prev=prev)  # prev could be None
        prev = (x[-1], cs[-1])  # only need the last dim
        x = torch.stack(x, dim=1)  # stack along t dim
        x = x.view(bs * t, self.hid_c, h, w)
        x = self.upsample(x)
        x = x.view(bs, t, self.out_size, h, w)
        return x, prev


class OneViewModel(nn.Module):
    CHANNELS = [256, 512, 1024]

    def __init__(self, resnet_path, spatial_size, channels, lstm_dropout_keep):
        super().__init__()
        self.resnet_path = resnet_path
        self.spatial_size = spatial_size

        sizes = [(int(math.ceil(spatial_size[0] / 2 / (2 ** i))),
                  int(math.ceil(spatial_size[1] / 2 / (2 ** i)))) for i in range(1, 4)]

        self.resnet_layers = self._get_resnets_layers()
        lstms = []
        for i in range(3):
            cell = BottleneckLSTMCell(sizes[i], channels[i], channels[i], lstm_dropout_keep, self.CHANNELS[i])
            lstms.append(cell)
        self.lstms = nn.ModuleList(lstms)

        pool_size = (1, 1)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)

    def _get_resnets_layers(self):
        resnet = load_model_dic(resnet50(norm_layer=lambda in_c: nn.BatchNorm2d(in_c)),
                                self.resnet_path, strict=False).get_layers(3)  # strict=False to accommodate norm layer
        layer1 = nn.Sequential(resnet[:5])
        layer2 = resnet[5]
        layer3 = resnet[6]
        return nn.ModuleList([layer1, layer2, layer3])

    def forward(self, x, prev):
        prevs = []
        for i in range(3):
            # print(7, i, x.shape)
            bs, t, c, h, w = x.shape
            x = x.view(bs * t, c, h, w)
            x = self.resnet_layers[i](x)
            _, c, h, w = x.shape
            x = x.view(bs, t, c, h, w)
            x, cs = self.lstms[i](x, prev[i] if prev else None)
            prevs.append(cs)

        x = self.pool(x.view(bs * t, c, h, w)).flatten(1).view(bs, t, c)
        return x, prevs


class ConvLSTMNet(nn.Module):
    """
    Plug convlstm into resnets
    """
    NUM_INTENTION = 3
    NUM_VIEWS = 3
    CHANNELS = [256, 512, 1024]

    def __init__(self, resnet_path, spatial_size, channels, fc_dropout_keep, lstm_dropout_keep):
        super(ConvLSTMNet, self).__init__()
        self.resnet_path = resnet_path

        self.view_model = OneViewModel(resnet_path, spatial_size, channels, lstm_dropout_keep)

        h, w = (1, 1)
        fc_in, fc_interm = self.CHANNELS[-1] * h * w, 2
        self.classifier = nn.Sequential(
            nn.Dropout(p=1 - fc_dropout_keep),
            nn.Linear(fc_in, 2 * 16, bias=True),
            nn.Dropout(p=1 - fc_dropout_keep),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(2 * 16, 2, bias=True)
        )

        print(f'convlstm model: fc_dropout_keep {fc_dropout_keep}, lstm_dropout_keep {lstm_dropout_keep}, '
              f'channels {channels}'
              f'\nWarning: Remember to manually reset/detach cell states!')

    def forward(self, view, prev):
        # assume input (bs, t, c, h, w)
        prevs = []
        x, cs = self.view_model(view, prev[0] if prev else None)
        prevs.append(cs)

        bs, t, c = x.shape
        x = x.view(bs * t, c)
        x = self.classifier(x)
        x = x.view(bs, t, -1)

        return x, prevs

    @staticmethod
    def detach_states(states):
        for depth_states in states:
            for i, (h, c) in enumerate(depth_states):
                h, c = h.detach(), c.detach()
                h.requires_grad, c.requires_grad = True, True
                depth_states[i] = (h, c)
        return states

    @staticmethod
    def derive_grad(y, x):
        for depth_y, depth_x in zip(y, x):
            for (yh, yc), (xh, xc) in zip(depth_y, depth_x):
                yc.backward(xc.grad, retain_graph=False)  # False still in testing

    def set_resnet_grad(self, grad, depth):
        for view in self.view_models:
            for param in view.resnet_layers[:depth].parameters():
                param.requires_grad = grad
            print(f'one view model param grad set to {grad}, till depth = {depth}')


if __name__ == '__main__':
    model = ConvLSTMNet('f:/resnet50.pth', (112, 112), [128, 192, 256], 0.7, 0.7, True, False,
                        sep_branch=True, skip_depth=[0, 1, 2]).cuda()
    out = model(torch.randn(1, 2, 3, 112, 112).cuda(),
                torch.randn(1, 2, 3, 112, 112).cuda(),
                torch.randn(1, 2, 3, 112, 112).cuda(),
                torch.randn(1, 2).cuda(),
                None)
    from count_model import params_count

    print(params_count(model))
    # print(out[0].shape, len(out[1]), len(out[1][0]))
