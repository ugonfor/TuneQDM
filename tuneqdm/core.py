"""
README — TuneQDM

Purpose
- Weight-only quantization (per-output-channel) with trainable scales.
- Timestep-aware scale update (TAS): keep int weights fixed; hold N expert copies of scales.
- Supports nn.Linear / nn.Conv2d. Simple API to quantize, enable training, set timestep, and export state.

Minimal Flow
1) Build float model.
2) Quantize: model = quantize_model(model, config)
3) Train-ready: enable_delta_training(model, train_delta_in=config.train_delta_in, train_bias=config.train_bias)
4) Training loop per step: set_timestep(model, t, total_denoising_steps=config.total_denoising_steps); forward/backward as usual.
5) Checkpoint: save_delta_state(model, path)

"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Constants & Config
# ============================================================================

EPSILON: float = 1e-8
DEFAULT_N_BITS: int = 8
DEFAULT_NUM_EXPERTS: int = 1
DEFAULT_TOTAL_STEPS: int = 1000


@dataclass
class TuneQDMConfig:
    """Configuration for TuneQDM + TAS.

    Attributes:
        n_bits: Quantization bit-width (int8 style supported).
        symmetric: Use symmetric quantization if True; else asymmetric (default).
        num_experts: Number of expert groups for timestep-aware scales. 1 disables TAS.
        total_denoising_steps: Total diffusion steps T used to compute expert index.
        train_delta_in: Whether to train delta_in in addition to delta_out.
        train_bias: Whether to train bias parameters.
    """
    n_bits: int = DEFAULT_N_BITS
    symmetric: bool = False
    num_experts: int = DEFAULT_NUM_EXPERTS
    total_denoising_steps: int = DEFAULT_TOTAL_STEPS
    train_delta_in: bool = True
    train_bias: bool = False


# ============================================================================
# Quantization Helper Functions
# ============================================================================

def _get_quantization_range(n_bits: int) -> Tuple[int, int]:
    """Calculate the signed quantization range for the given bit width.

    Args:
        n_bits: Number of bits for quantization.

    Returns:
        (qmin, qmax). For int8, (-128, 127).
    """
    qmin = -(1 << (n_bits - 1))
    qmax = (1 << (n_bits - 1)) - 1
    return qmin, qmax


def _compute_per_output_min_max(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-output-channel min and max.

    Args:
        weight: Weight tensor shaped (out_channels, ...).

    Returns:
        (min_values, max_values) each shaped (out_channels,).
    """
    flattened = weight.view(weight.size(0), -1)
    min_values = flattened.min(dim=1).values
    max_values = flattened.max(dim=1).values
    return min_values, max_values


def _compute_per_output_max_abs(weight: torch.Tensor) -> torch.Tensor:
    """Compute per-output-channel max absolute value.

    Args:
        weight: Weight tensor shaped (out_channels, ...).

    Returns:
        Max absolute values shaped (out_channels,). Clamped to positive.
    """
    flattened_abs = weight.view(weight.size(0), -1).abs()
    max_abs = flattened_abs.max(dim=1).values.clamp(min=EPSILON)
    return max_abs


def _broadcast_to_match(vector: torch.Tensor, reference: torch.Tensor, axis: int = 0) -> torch.Tensor:
    """Broadcast 1D vector to match a reference tensor along axis.

    Args:
        vector: 1D tensor to broadcast.
        reference: Reference tensor to match shape.
        axis: Axis along which to place the vector entries.

    Returns:
        Broadcasted tensor compatible with the reference.
    """
    shape = [1] * reference.dim()
    shape[axis] = -1
    return vector.view(*shape)


def _calculate_scale_and_zero_point(
    weight: torch.Tensor,
    n_bits: int,
    symmetric: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate per-output-channel scale and zero-point.

    Args:
        weight: Weight tensor shaped (out_channels, ...).
        n_bits: Number of quantization bits.
        symmetric: If True, symmetric (zp=0). Else asymmetric.

    Returns:
        (scale, zero_point) each shaped (out_channels,).
    """
    qmin, qmax = _get_quantization_range(n_bits)
    device = weight.device

    if symmetric:
        max_abs = _compute_per_output_max_abs(weight)
        scale = (max_abs / max(qmax, 1)).clamp(min=EPSILON).to(device)
        zero_point = torch.zeros_like(scale, device=device)
    else:
        wmin, wmax = _compute_per_output_min_max(weight)
        scale = ((wmax - wmin) / max(qmax - qmin, 1)).clamp(min=EPSILON).to(device)
        zero_point = (qmin - torch.round(wmin / scale)).to(device)

    return scale, zero_point


def _quantize_per_output_channel(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    n_bits: int,
    symmetric: bool
) -> torch.Tensor:
    """Quantize weights to int8 per output channel.

    Args:
        weight: Weight tensor to quantize (float).
        scale: Per-output scale (out,).
        zero_point: Per-output zero-point (out,).
        n_bits: Quantization bits.
        symmetric: If True, ignore zero-point.

    Returns:
        Quantized weight tensor as int8 with same shape as input.
    """
    qmin, qmax = _get_quantization_range(n_bits)
    scale_bc = _broadcast_to_match(scale, weight, axis=0)
    zero_point_bc = _broadcast_to_match(zero_point, weight, axis=0)

    if symmetric:
        quantized = torch.round(weight / scale_bc)
    else:
        quantized = torch.round(weight / scale_bc + zero_point_bc)

    return torch.clamp(quantized, qmin, qmax).to(torch.int8)


def _compute_expert_index(t: Union[int, torch.Tensor], total_steps: int, num_experts: int) -> int:
    """Compute expert index i = floor(t * N / T).

    Args:
        t: Diffusion timestep (int or 0/1-D tensor).
        total_steps: Total denoising steps T.
        num_experts: Number of experts N.

    Returns:
        Expert index in [0, N-1].
    """
    if isinstance(t, torch.Tensor):
        t_val = int(t[0].item()) if t.dim() > 0 else int(t.item())
    else:
        t_val = int(t)
    total_steps = max(int(total_steps), 1)
    num_experts = max(int(num_experts), 1)
    idx = (t_val * num_experts) // total_steps
    idx = min(max(idx, 0), num_experts - 1)
    return idx


# ============================================================================
# Quantized Layer Classes
# ============================================================================

class TQLinear(nn.Module):
    """Quantized Linear layer with timestep-aware expert scales.

    Stores int8 weights and zero-points as buffers.
    Holds N expert copies of (delta_out, delta_in) and picks one by timestep t.

    Forward dequantization:
        W = (W_int - zp) * delta_out[i] * delta_in[i]
    where i = floor(t * N / T).

    Note:
        If num_experts == 1, behaves like the original single-scale TuneQDM layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_int: torch.Tensor,
        zero_point_bc: torch.Tensor,
        delta_out_init: torch.Tensor,
        n_bits: int,
        symmetric: bool,
        num_experts: int,
        total_denoising_steps: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_bits = int(n_bits)
        self.symmetric = bool(symmetric)
        self.num_experts = int(num_experts)
        self.total_denoising_steps = int(total_denoising_steps)

        # Fixed quantized weights and zero-point
        self.register_buffer("weight_int", weight_int.contiguous().to(torch.int8))
        self.register_buffer("zero_point", zero_point_bc.contiguous())

        # Expert parameters
        # delta_out_experts: (N, out, 1)
        # delta_in_experts:  (N, 1, in)
        delta_out_base = delta_out_init.view(self.out_features, 1)
        delta_in_base = torch.ones(1, self.in_features, device=weight_int.device if device is None else device)

        self.delta_out_experts = nn.Parameter(
            delta_out_base.unsqueeze(0).repeat(self.num_experts, 1, 1)
        )
        self.delta_in_experts = nn.Parameter(
            delta_in_base.unsqueeze(0).repeat(self.num_experts, 1, 1)
        )

        # Bias kept frozen by default; created at from_float if present
        self.bias: Optional[nn.Parameter] = None

        # Runtime index buffer (non-persistent)
        self.register_buffer("current_expert_index", torch.tensor(0, dtype=torch.long), persistent=False)

    @staticmethod
    def from_float(module: nn.Linear, config: TuneQDMConfig) -> "TQLinear":
        """Create a TQLinear with experts from a float Linear.

        Args:
            module: Float nn.Linear.
            config: TuneQDMConfig.

        Returns:
            TQLinear instance with expert parameters initialized.
        """
        with torch.no_grad():
            weight = module.weight.data
            scale, zero_point = _calculate_scale_and_zero_point(weight, config.n_bits, config.symmetric)
            weight_int = _quantize_per_output_channel(weight, scale, zero_point, config.n_bits, config.symmetric)
            zero_point_bc = _broadcast_to_match(zero_point, weight_int, axis=0)

        layer = TQLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            weight_int=weight_int,
            zero_point_bc=zero_point_bc,
            delta_out_init=scale,
            n_bits=config.n_bits,
            symmetric=config.symmetric,
            num_experts=config.num_experts,
            total_denoising_steps=config.total_denoising_steps,
            device=weight_int.device,
        )

        if module.bias is not None:
            layer.bias = nn.Parameter(module.bias.detach().clone())
            layer.bias.requires_grad_(False)

        return layer

    def set_timestep(self, t: Union[int, torch.Tensor], total_denoising_steps: Optional[int] = None) -> None:
        """Select expert index based on the given timestep.

        Args:
            t: Diffusion timestep.
            total_denoising_steps: Optional override of total steps T.
        """
        if total_denoising_steps is not None:
            self.total_denoising_steps = int(total_denoising_steps)
        idx = _compute_expert_index(t, self.total_denoising_steps, self.num_experts)
        self.current_expert_index.data = torch.tensor(idx, dtype=torch.long, device=self.current_expert_index.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation with dequantized expert-selected weights.

        Args:
            x: Input tensor of shape (N, in_features).

        Returns:
            Output tensor of shape (N, out_features).
        """
        idx = int(self.current_expert_index.item())
        delta_out = self.delta_out_experts[idx]  # (out, 1)
        delta_in = self.delta_in_experts[idx]    # (1, in)

        weight_float = (self.weight_int.float() - self.zero_point) * delta_out * delta_in
        weight_float = weight_float.to(x.dtype)
        return F.linear(x, weight_float, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, n_bits={self.n_bits}, symmetric={self.symmetric}, "
            f"experts={self.num_experts}, T={self.total_denoising_steps}"
        )


class TQConv2d(nn.Module):
    """Quantized Conv2d layer with timestep-aware expert scales.

    Stores int weights and zero-points as buffers.
    Holds N expert copies of (delta_out, delta_in) and picks one by timestep t.

    Forward dequantization:
        W = (W_int - zp) * delta_out[i] * delta_in[i]
    where i = floor(t * N / T).
    """

    def __init__(
        self,
        original_conv: nn.Conv2d,
        weight_int: torch.Tensor,
        zero_point_bc: torch.Tensor,
        delta_out_init: torch.Tensor,
        n_bits: int,
        symmetric: bool,
        num_experts: int,
        total_denoising_steps: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        # Convolution meta copied from original module
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups

        self.n_bits = int(n_bits)
        self.symmetric = bool(symmetric)
        self.num_experts = int(num_experts)
        self.total_denoising_steps = int(total_denoising_steps)

        # Fixed quantized weights & zero-point
        self.register_buffer("weight_int", weight_int.contiguous().to(torch.int8))
        self.register_buffer("zero_point", zero_point_bc.contiguous())

        # Expert parameters
        out_channels = weight_int.shape[0]
        in_channels = weight_int.shape[1]

        delta_out_base = delta_out_init.view(out_channels, 1, 1, 1)
        delta_in_base = torch.ones(1, in_channels, 1, 1, device=weight_int.device if device is None else device)

        # delta_out: (N, Cout, 1, 1, 1), delta_in: (N, 1, Cin, 1, 1)
        self.delta_out_experts = nn.Parameter(
            delta_out_base.unsqueeze(0).repeat(self.num_experts, 1, 1, 1, 1)
        )
        self.delta_in_experts = nn.Parameter(
            delta_in_base.unsqueeze(0).repeat(self.num_experts, 1, 1, 1, 1)
        )

        # Bias kept frozen by default
        self.bias: Optional[nn.Parameter] = None

        # Runtime index
        self.register_buffer("current_expert_index", torch.tensor(0, dtype=torch.long), persistent=False)

    @staticmethod
    def from_float(module: nn.Conv2d, config: TuneQDMConfig) -> "TQConv2d":
        """Create a TQConv2d with experts from a float Conv2d.

        Args:
            module: Float nn.Conv2d.
            config: TuneQDMConfig.

        Returns:
            TQConv2d instance with expert parameters initialized.
        """
        with torch.no_grad():
            weight = module.weight.data
            scale, zero_point = _calculate_scale_and_zero_point(weight, config.n_bits, config.symmetric)
            weight_int = _quantize_per_output_channel(weight, scale, zero_point, config.n_bits, config.symmetric)
            zero_point_bc = _broadcast_to_match(zero_point, weight_int, axis=0)

        layer = TQConv2d(
            original_conv=module,
            weight_int=weight_int,
            zero_point_bc=zero_point_bc,
            delta_out_init=scale,
            n_bits=config.n_bits,
            symmetric=config.symmetric,
            num_experts=config.num_experts,
            total_denoising_steps=config.total_denoising_steps,
            device=weight_int.device,
        )

        if module.bias is not None:
            layer.bias = nn.Parameter(module.bias.detach().clone())
            layer.bias.requires_grad_(False)

        return layer

    def set_timestep(self, t: Union[int, torch.Tensor], total_denoising_steps: Optional[int] = None) -> None:
        """Select expert index based on the given timestep.

        Args:
            t: Diffusion timestep.
            total_denoising_steps: Optional override of total steps T.
        """
        if total_denoising_steps is not None:
            self.total_denoising_steps = int(total_denoising_steps)
        idx = _compute_expert_index(t, self.total_denoising_steps, self.num_experts)
        self.current_expert_index.data = torch.tensor(idx, dtype=torch.long, device=self.current_expert_index.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolution with dequantized expert-selected weights.

        Args:
            x: Input tensor (N, Cin, H, W).

        Returns:
            Output tensor (N, Cout, H_out, W_out).
        """
        idx = int(self.current_expert_index.item())
        delta_out = self.delta_out_experts[idx]  # (Cout,1,1,1)
        delta_in = self.delta_in_experts[idx]    # (1,Cin,1,1)

        weight_float = (self.weight_int.float() - self.zero_point) * delta_out * delta_in
        weight_float = weight_float.to(x.dtype)
        return F.conv2d(
            x, weight_float, self.bias,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        )

    def extra_repr(self) -> str:
        in_channels = self.weight_int.shape[1]
        out_channels = self.weight_int.shape[0]
        kernel_size = tuple(self.weight_int.shape[2:])
        return (
            f"{in_channels}, {out_channels}, kernel_size={kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, "
            f"bias={self.bias is not None}, n_bits={self.n_bits}, symmetric={self.symmetric}, "
            f"experts={self.num_experts}, T={self.total_denoising_steps}"
        )


# ============================================================================
# Model Quantization & Training Utilities
# ============================================================================

def quantize_model(model: nn.Module, config: TuneQDMConfig) -> nn.Module:
    """Replace all Linear/Conv2d with quantized expert-aware versions.

    Args:
        model: Float model to quantize.
        config: TuneQDMConfig.

    Returns:
        Model with nn.Linear -> TQLinear, nn.Conv2d -> TQConv2d.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            quantized_layer = TQLinear.from_float(child, config)
            setattr(model, name, quantized_layer)
        elif isinstance(child, nn.Conv2d):
            quantized_layer = TQConv2d.from_float(child, config)
            setattr(model, name, quantized_layer)
        else:
            quantize_model(child, config)
    return model


def enable_delta_training(model: nn.Module, train_delta_in: bool = True, train_bias: bool = False) -> None:
    """Freeze all parameters except scale experts (and optionally bias).

    Args:
        model: Quantized model.
        train_delta_in: If True, train delta_in experts.
        train_bias: If True, train bias parameters.
    """
    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, (TQLinear, TQConv2d)):
            module.delta_out_experts.requires_grad_(True)
            module.delta_in_experts.requires_grad_(train_delta_in)
            if module.bias is not None and train_bias:
                module.bias.requires_grad_(True)


def set_timestep(model: nn.Module, t: Union[int, torch.Tensor], total_denoising_steps: Optional[int] = None) -> None:
    """Set timestep for all quantized layers to select expert index.

    Args:
        model: Model containing TQLinear/TQConv2d.
        t: Diffusion timestep.
        total_denoising_steps: Optional override of T for all layers.
    """
    for module in model.modules():
        if isinstance(module, (TQLinear, TQConv2d)):
            module.set_timestep(t, total_denoising_steps)


def apply_tuneqdm(
    model: nn.Module,
    config: TuneQDMConfig
) -> nn.Module:
    """Quantize a model and make it ready for TuneQDM training with TAS.

    Args:
        model: Float model to quantize.
        config: TuneQDMConfig.

    Returns:
        Quantized model ready for training (scales/bias grad flags set).
    """
    model = quantize_model(model, config)
    enable_delta_training(model, train_delta_in=config.train_delta_in, train_bias=config.train_bias)
    return model


# ============================================================================
# State Management Utilities
# ============================================================================

def export_delta_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Export quantization-related parameters including expert scales.

    Args:
        model: Quantized model.

    Returns:
        Dictionary containing delta_out_experts, delta_in_experts, bias,
        weight_int, and zero_point buffers.
    """
    delta_state: Dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if (
            key.endswith("delta_out_experts")
            or key.endswith("delta_in_experts")
            or key.endswith("bias")
            or key.endswith("weight_int")
            or key.endswith("zero_point")
        ):
            delta_state[key] = value
    return delta_state


def save_delta_state(model: nn.Module, path: str, config: Optional[TuneQDMConfig] = None) -> None:
    """Save quantization-related parameters and config to a file.

    Args:
        model: Quantized model.
        path: Destination path for torch.save.
        config: Optional TuneQDMConfig to save alongside weights.
    """
    delta_state = export_delta_state(model)
    save_dict = {"delta_state": delta_state}
    if config is not None:
        save_dict["config"] = {
            "n_bits": config.n_bits,
            "symmetric": config.symmetric,
            "num_experts": config.num_experts,
            "total_denoising_steps": config.total_denoising_steps,
            "train_delta_in": config.train_delta_in,
            "train_bias": config.train_bias,
        }
    torch.save(save_dict, path)


def load_delta_state(
    model: nn.Module,
    path: str,
    map_location: Optional[Union[str, torch.device]] = None
) -> Optional[TuneQDMConfig]:
    """Load quantization-related parameters from a file.

    Args:
        model: Quantized model to load parameters into.
        path: Path to the saved state dictionary.
        map_location: Device mapping for torch.load.

    Returns:
        TuneQDMConfig if saved with config, else None.
    """
    checkpoint = torch.load(path, map_location=map_location)

    # Handle both old format (direct state_dict) and new format (with config)
    if "delta_state" in checkpoint:
        state_dict = checkpoint["delta_state"]
        config_dict = checkpoint.get("config", None)
    else:
        # Legacy format: checkpoint is the state_dict itself
        state_dict = checkpoint
        config_dict = None

    model.load_state_dict(state_dict, strict=False)

    if config_dict is not None:
        return TuneQDMConfig(**config_dict)
    return None


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    class SmallNet(nn.Module):
        """Tiny example network containing Linear and Conv2d."""
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(16, 16)
            self.conv = nn.Conv2d(8, 8, kernel_size=3, padding=1)

        def forward(self, x_vec: torch.Tensor, x_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass for demonstration."""
            return self.fc(x_vec), self.conv(x_img)

    # 1) Build float model
    net = SmallNet()

    # 2) Prepare config
    cfg = TuneQDMConfig(
        n_bits=8,
        symmetric=False,
        num_experts=4,
        total_denoising_steps=1000,
        train_delta_in=True,
        train_bias=False,
    )

    # 3) Quantize and enable training for scale experts
    net = apply_tuneqdm(net, cfg)

    # 4) Set timestep each iteration before forward
    batch_timestep = torch.tensor([427])  # example diffusion timestep
    set_timestep(net, batch_timestep, total_denoising_steps=cfg.total_denoising_steps)

    # 5) Dummy forward
    x_vec = torch.randn(2, 16)
    x_img = torch.randn(2, 8, 32, 32)
    y_vec, y_img = net(x_vec, x_img)

    # Now run your loss/backward/step as usual.
