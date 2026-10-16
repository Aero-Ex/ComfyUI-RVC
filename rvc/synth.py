import math

import torch
from torch import nn
from torch.nn import functional as F

import comfy.ops


LRELU_SLOPE = 0.1
SAMPLE_RATES = {"32k": 32000, "40k": 40000, "48k": 48000}


def fold_weight_norm(state_dict, dim=0):
    """Replace weight_g/weight_v pairs by the weight they parametrize, so layers are plain convs at inference."""
    sd = dict(state_dict)
    for key in [k for k in sd if k.endswith(".weight_g")]:
        prefix = key[:-len("weight_g")]
        g, v = sd.pop(key), sd.pop(prefix + "weight_v")
        sd[prefix + "weight"] = torch._weight_norm(v.float(), g.float(), dim).to(v.dtype)
    return sd


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class LayerNorm(nn.Module):
    """LayerNorm over the channel axis of [B, C, T] inputs."""

    def __init__(self, channels, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.empty(channels, device=device, dtype=dtype))
        self.beta = nn.Parameter(torch.empty(channels, device=device, dtype=dtype))

    def forward(self, x):
        x = F.layer_norm(x.transpose(1, -1), (self.channels,), comfy.ops.cast_to_input(self.gamma, x), comfy.ops.cast_to_input(self.beta, x), self.eps)
        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    """VITS self-attention with windowed relative position keys and values."""

    def __init__(self, channels, n_heads, window_size, device=None, dtype=None, operations=None):
        super().__init__()
        self.n_heads = n_heads
        self.window_size = window_size
        self.k_channels = channels // n_heads
        self.conv_q = operations.Conv1d(channels, channels, 1, device=device, dtype=dtype)
        self.conv_k = operations.Conv1d(channels, channels, 1, device=device, dtype=dtype)
        self.conv_v = operations.Conv1d(channels, channels, 1, device=device, dtype=dtype)
        self.conv_o = operations.Conv1d(channels, channels, 1, device=device, dtype=dtype)
        self.emb_rel_k = nn.Parameter(torch.empty(1, window_size * 2 + 1, self.k_channels, device=device, dtype=dtype))
        self.emb_rel_v = nn.Parameter(torch.empty(1, window_size * 2 + 1, self.k_channels, device=device, dtype=dtype))

    def forward(self, x, attn_mask):
        b, _, t = x.shape
        q = self.conv_q(x).view(b, self.n_heads, self.k_channels, t).transpose(2, 3) * self.k_channels ** -0.5
        k = self.conv_k(x).view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        v = self.conv_v(x).view(b, self.n_heads, self.k_channels, t).transpose(2, 3)
        # The relative value term needs the attention probabilities themselves, so this cannot use a fused kernel.
        rel_k = self._relative_embeddings(comfy.ops.cast_to_input(self.emb_rel_k, q), t)
        scores = torch.matmul(q, k.transpose(-2, -1)) + self._relative_to_absolute(torch.matmul(q, rel_k.transpose(-2, -1)))
        scores = scores.masked_fill(attn_mask == 0, -1e4)
        p_attn = scores.float().softmax(dim=-1).to(q.dtype)
        rel_v = self._relative_embeddings(comfy.ops.cast_to_input(self.emb_rel_v, q), t)
        out = torch.matmul(p_attn, v) + torch.matmul(self._absolute_to_relative(p_attn), rel_v)
        return self.conv_o(out.transpose(2, 3).reshape(b, -1, t))

    def _relative_embeddings(self, embeddings, length):
        pad = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        if pad > 0:
            embeddings = F.pad(embeddings, (0, 0, pad, pad))
        return embeddings[:, start:start + 2 * length - 1]

    @staticmethod
    def _relative_to_absolute(x):
        batch, heads, length, _ = x.shape
        x = F.pad(x, (0, 1)).view(batch, heads, length * 2 * length)
        x = F.pad(x, (0, length - 1))
        return x.view(batch, heads, length + 1, 2 * length - 1)[:, :, :length, length - 1:]

    @staticmethod
    def _absolute_to_relative(x):
        batch, heads, length, _ = x.shape
        x = F.pad(x, (0, length - 1)).view(batch, heads, length ** 2 + length * (length - 1))
        x = F.pad(x, (length, 0))
        return x.view(batch, heads, length, 2 * length)[:, :, :, 1:]


class FFN(nn.Module):
    def __init__(self, channels, filter_channels, kernel_size, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv_1 = operations.Conv1d(channels, filter_channels, kernel_size, padding=kernel_size // 2, device=device, dtype=dtype)
        self.conv_2 = operations.Conv1d(filter_channels, channels, kernel_size, padding=kernel_size // 2, device=device, dtype=dtype)

    def forward(self, x, x_mask):
        x = torch.relu(self.conv_1(x * x_mask))
        return self.conv_2(x * x_mask) * x_mask


class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, window_size=10, device=None, dtype=None, operations=None):
        super().__init__()
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, n_heads, window_size, device=device, dtype=dtype, operations=operations))
            self.norm_layers_1.append(LayerNorm(hidden_channels, device=device, dtype=dtype))
            self.ffn_layers.append(FFN(hidden_channels, filter_channels, kernel_size, device=device, dtype=dtype, operations=operations))
            self.norm_layers_2.append(LayerNorm(hidden_channels, device=device, dtype=dtype))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn, norm_1, ffn, norm_2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            x = norm_1(x + attn(x, attn_mask))
            x = norm_2(x + ffn(x, x_mask))
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, f0=True, device=None, dtype=None, operations=None):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = operations.Linear(in_channels, hidden_channels, device=device, dtype=dtype)
        if f0:
            self.emb_pitch = operations.Embedding(256, hidden_channels, device=device, dtype=dtype)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, device=device, dtype=dtype, operations=operations)
        self.proj = operations.Conv1d(hidden_channels, out_channels * 2, 1, device=device, dtype=dtype)

    def forward(self, phone, pitch, lengths):
        x = self.emb_phone(phone)
        if pitch is not None:
            x = x + self.emb_pitch(pitch)
        x = F.leaky_relu(x * math.sqrt(self.hidden_channels), 0.1).transpose(1, -1)
        x_mask = (torch.arange(x.shape[2], device=x.device) < lengths.unsqueeze(1)).unsqueeze(1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        m, logs = torch.split(self.proj(x) * x_mask, self.out_channels, dim=1)
        return m, logs, x_mask


class WN(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, device=None, dtype=None, operations=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.cond_layer = operations.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1, device=device, dtype=dtype)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = dilation_rate ** i
            self.in_layers.append(operations.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=get_padding(kernel_size, dilation), device=device, dtype=dtype))
            res_skip_channels = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(operations.Conv1d(hidden_channels, res_skip_channels, 1, device=device, dtype=dtype))

    def forward(self, x, x_mask, g):
        output = torch.zeros_like(x)
        g = self.cond_layer(g).split(2 * self.hidden_channels, dim=1)
        h = self.hidden_channels
        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            x_in = in_layer(x) + g[i]
            acts = torch.tanh(x_in[:, :h]) * torch.sigmoid(x_in[:, h:])
            res_skip = res_skip_layer(acts)
            if i < self.n_layers - 1:
                x = (x + res_skip[:, :h]) * x_mask
                output = output + res_skip[:, h:]
            else:
                output = output + res_skip
        return output * x_mask


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size, dilation, device=None, dtype=None, operations=None):
        super().__init__()
        self.convs1 = nn.ModuleList([operations.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d), device=device, dtype=dtype) for d in dilation])
        self.convs2 = nn.ModuleList([operations.Conv1d(channels, channels, kernel_size, 1, padding=get_padding(kernel_size), device=device, dtype=dtype) for _ in dilation])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU_SLOPE)), LRELU_SLOPE))
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size, dilation, device=None, dtype=None, operations=None):
        super().__init__()
        self.convs = nn.ModuleList([operations.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d), device=device, dtype=dtype) for d in dilation])

    def forward(self, x):
        for c in self.convs:
            x = x + c(F.leaky_relu(x, LRELU_SLOPE))
        return x


class ResidualCouplingLayer(nn.Module):
    """Mean-only affine coupling, inverse direction only (inference)."""

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, device=None, dtype=None, operations=None):
        super().__init__()
        self.half_channels = channels // 2
        self.pre = operations.Conv1d(self.half_channels, hidden_channels, 1, device=device, dtype=dtype)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, device=device, dtype=dtype, operations=operations)
        self.post = operations.Conv1d(hidden_channels, self.half_channels, 1, device=device, dtype=dtype)

    def forward(self, x, x_mask, g):
        x0, x1 = x.split(self.half_channels, dim=1)
        m = self.post(self.enc(self.pre(x0) * x_mask, x_mask, g)) * x_mask
        return torch.cat([x0, (x1 - m) * x_mask], 1)


class Flip(nn.Module):
    pass


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, n_flows=4, device=None, dtype=None, operations=None):
        super().__init__()
        # Flips sit between the couplings in the checkpoint's module numbering.
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels, device=device, dtype=dtype, operations=operations))
            self.flows.append(Flip())

    def forward(self, x, x_mask, g):
        for flow in reversed(self.flows):
            x = torch.flip(x, [1]) if isinstance(flow, Flip) else flow(x, x_mask, g)
        return x


class SineGen(nn.Module):
    """Sine excitation for the NSF source: harmonics of f0 in voiced frames, noise in unvoiced ones."""

    def __init__(self, samp_rate, sine_amp=0.1, noise_std=0.003):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.sampling_rate = samp_rate

    def forward(self, f0, upp, generator):
        f0 = f0.unsqueeze(-1)
        rad = f0 / self.sampling_rate * torch.arange(1, upp + 1, device=f0.device)
        # Phase is accumulated per frame in fp32 and wrapped, so long inputs do not lose precision.
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad = (rad + F.pad(rad_acc, (0, 0, 1, -1))).reshape(f0.shape[0], -1, 1)
        sine_waves = torch.sin(2 * math.pi * rad) * self.sine_amp
        uv = F.interpolate((f0 > 0).to(f0.dtype).transpose(2, 1), scale_factor=float(upp), mode="nearest").transpose(2, 1)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * torch.randn(sine_waves.shape, device=f0.device, dtype=sine_waves.dtype, generator=generator)
        return sine_waves * uv + noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, device=None, dtype=None, operations=None):
        super().__init__()
        self.l_sin_gen = SineGen(sampling_rate)
        self.l_linear = operations.Linear(1, 1, device=device, dtype=dtype)

    def forward(self, f0, upp, generator):
        return torch.tanh(self.l_linear(self.l_sin_gen(f0, upp, generator).to(self.l_linear.weight.dtype)))


class Generator(nn.Module):
    """HiFi-GAN decoder, optionally driven by an NSF harmonic source (f0 voices)."""

    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels, sr=None, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.upp = math.prod(upsample_rates)
        self.conv_pre = operations.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3, device=device, dtype=dtype)
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        if sr is not None:
            self.m_source = SourceModuleHnNSF(sr, device=device, dtype=dtype, operations=operations)
            self.noise_convs = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(operations.ConvTranspose1d(upsample_initial_channel // (2 ** i), ch, k, u, padding=(k - u) // 2, device=device, dtype=dtype))
            if sr is not None:
                stride_f0 = math.prod(upsample_rates[i + 1:])
                if i + 1 < len(upsample_rates):
                    self.noise_convs.append(operations.Conv1d(1, ch, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2, device=device, dtype=dtype))
                else:
                    self.noise_convs.append(operations.Conv1d(1, ch, kernel_size=1, device=device, dtype=dtype))
            for rk, rd in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(resblock_cls(ch, rk, rd, device=device, dtype=dtype, operations=operations))
        self.conv_post = operations.Conv1d(ch, 1, 7, 1, padding=3, bias=False, device=device, dtype=dtype)
        self.cond = operations.Conv1d(gin_channels, upsample_initial_channel, 1, device=device, dtype=dtype)

    def forward(self, x, g, f0=None, generator=None):
        if f0 is not None:
            source = self.m_source(f0, self.upp, generator).transpose(1, 2)
        x = self.conv_pre(x) + self.cond(g)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            if f0 is not None:
                x = x + self.noise_convs[i](source)
            blocks = self.resblocks[i * self.num_kernels:(i + 1) * self.num_kernels]
            xs = blocks[0](x)
            for block in blocks[1:]:
                xs += block(x)
            x = xs / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x)))


class Synthesizer(nn.Module):
    """RVC voice: text encoder over content features (+ coarse pitch), inverse flow, HiFi-GAN/NSF decoder.

    config is the 18-entry list stored in RVC voice checkpoints.
    """

    def __init__(self, config, phone_channels, f0, device=None, dtype=None, operations=None):
        super().__init__()
        (_, _, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, _, resblock,
         resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
         upsample_kernel_sizes, spk_embed_dim, gin_channels, sr) = config
        sr = SAMPLE_RATES.get(sr, sr)
        self.enc_p = TextEncoder(phone_channels, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, f0=f0, device=device, dtype=dtype, operations=operations)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels, sr=sr if f0 else None, device=device, dtype=dtype, operations=operations)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels, device=device, dtype=dtype, operations=operations)
        self.emb_g = operations.Embedding(spk_embed_dim, gin_channels, device=device, dtype=dtype)
        self.dtype = dtype

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(fold_weight_norm(state_dict), strict=strict, assign=assign)

    def infer(self, phone, lengths, sid, generator, pitch=None, nsff0=None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, lengths)
        noise = torch.randn(m_p.shape, device=m_p.device, dtype=m_p.dtype, generator=generator)
        z = self.flow((m_p + torch.exp(logs_p) * noise * 0.66666) * x_mask, x_mask, g)
        return self.dec(z * x_mask, g, nsff0, generator)
