"""MAISI VAE-GAN training core (Stage 1), replicated from NVIDIA MAISI.

Every component and weight is pinned to MAISI's released recipe — see
docs/2026-06-05-maisi-vae-recipe.md and configs/vae_pet_maisi.json. Faithful to
`generation/maisi/maisi_train_vae_tutorial.ipynb` + `scripts/utils.py` (MONAI tutorials).

  loss_g = L1(recon) + kl_weight*KL + perceptual_weight*Perceptual + adv_weight*LSGAN(gen)
  loss_d = 0.5 * (LSGAN(fake) + LSGAN(real))

The only PET changes live outside this file: input is single-channel SUV normalized by
`petinct.data.suv_norm`; reconstruction fidelity is judged by `petinct.metrics.suv_fidelity`.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math
from types import MethodType

import torch


def _activation_checkpointing_active(module) -> bool:
    """Return whether an installed checkpoint wrapper should execute.

    Training remains the default activation condition.  Deterministic
    full-field validation may opt in temporarily: it needs the same bounded
    execution graph as training, while the codec itself must remain in eval
    mode.
    """
    return torch.is_grad_enabled() and (
        module.training
        or bool(
            getattr(
                module,
                "_petinct_force_checkpoint_in_eval",
                False,
            )
        )
    )


@contextmanager
def force_activation_checkpointing_in_eval(autoencoder):
    """Temporarily activate installed wrappers without calling ``train()``.

    This is intended for read-only, full-field validation.  It preserves eval
    semantics while allowing non-reentrant checkpoints and the memory-bounded
    MAISI upsample implementation to run.  Modules without an installed
    wrapper are left untouched.
    """
    wrapped_modules = [
        module
        for module in autoencoder.modules()
        if getattr(module, "_petinct_checkpoint_wrapped", False)
    ]
    previous = [
        (
            module,
            getattr(
                module,
                "_petinct_force_checkpoint_in_eval",
                None,
            ),
            hasattr(module, "_petinct_force_checkpoint_in_eval"),
        )
        for module in wrapped_modules
    ]
    try:
        for module in wrapped_modules:
            module._petinct_force_checkpoint_in_eval = True
        yield
    finally:
        for module, value, existed in previous:
            if existed:
                module._petinct_force_checkpoint_in_eval = value
            else:
                delattr(module, "_petinct_force_checkpoint_in_eval")


def kl_loss(z_mu, z_sigma):
    """KL divergence to N(0,1), averaged over batch. Verbatim MAISI (scripts/utils.py:KL_loss)."""
    eps = 1e-10
    kl = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + eps) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl) / kl.shape[0]


def build_autoencoder(autoencoder_def: dict, num_splits: int = 1):
    """Instantiate MAISI's AutoencoderKlMaisi from the config dict, in TRAINING mode.

    Two config fields are Tensor-Split-Parallelism (TSP) low-VRAM *inference* tricks and are
    overridden for AMP training (the MAISI VAE tutorial trains this way):
      - num_splits  -> 1     (config's 4/16 splits each conv for inference; 1 = no split).
      - norm_float16 -> False (fp16 group-norm emits half activations that hit the next conv's
        float32 bias whenever there is no autocast — e.g. a plain eval()/no_grad forward —
        giving "Input type (Half) and bias type (float)". AMP handles training precision instead.)
    """
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

    kwargs = {k: v for k, v in autoencoder_def.items() if not k.startswith("_")}
    kwargs["num_splits"] = num_splits
    kwargs["norm_float16"] = False
    return AutoencoderKlMaisi(**kwargs)


def disable_maisi_internal_cuda_cache_flush(autoencoder) -> int:
    """Disable MAISI's in-forward ``torch.cuda.empty_cache`` calls.

    MAISI's ``save_mem`` path flushes the CUDA allocator from inside encoder
    and decoder forwards.  That is an inference-oriented memory strategy and
    is unsafe when the same forward is recomputed by activation checkpointing:
    asynchronous CUDA faults can surface at the allocator flush and abort one
    DDP rank.  Training already bounds activation memory with non-reentrant
    block checkpointing, so allocator flushing must remain disabled.

    Returns the number of MAISI modules whose explicit ``save_mem`` contract
    was disabled.  A zero count is an error at the training entry because it
    would mean the installed MAISI implementation no longer matches this
    safety contract.
    """
    disabled = 0
    for module in autoencoder.modules():
        if hasattr(module, "save_mem"):
            module.save_mem = False
            disabled += 1
    return disabled


def enable_encoder_decoder_checkpointing(autoencoder) -> None:
    """Checkpoint the complete MAISI encoder and decoder without changing keys.

    ``AutoencoderKlMaisi`` accepts ``use_checkpointing`` but replaces the base
    encoder/decoder with custom MAISI modules that do not consult that flag.
    This instance-level wrapper therefore makes the requested behavior real
    while leaving every module, parameter and ``state_dict`` key unchanged.
    Non-reentrant checkpointing is required because training images do not
    require gradients even though model parameters do.
    """
    from torch.utils.checkpoint import checkpoint

    for module_name in ("encoder", "decoder"):
        module = getattr(autoencoder, module_name)
        if getattr(module, "_petinct_checkpoint_wrapped", False):
            raise ValueError(f"{module_name} is already checkpoint wrapped")
        original_forward = module.forward

        def checkpointed_forward(
            self,
            x,
            *,
            _original_forward=original_forward,
        ):
            if _activation_checkpointing_active(self):
                return checkpoint(
                    _original_forward,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            return _original_forward(x)

        module.forward = MethodType(checkpointed_forward, module)
        module._petinct_checkpoint_wrapped = True


def enable_maisi_block_checkpointing(autoencoder) -> None:
    """Checkpoint every custom MAISI encoder/decoder block independently.

    Whole-encoder/decoder checkpointing reduces forward memory but reconstructs
    every internal activation at once during backward. Per-block wrapping bounds
    recomputation memory to one MAISI block while preserving modules and state
    keys.
    """
    from torch.utils.checkpoint import checkpoint

    for container_name in ("encoder", "decoder"):
        container = getattr(autoencoder, container_name)
        blocks = getattr(container, "blocks", None)
        if blocks is None:
            raise ValueError(
                f"{container_name} has no MAISI blocks ModuleList"
            )
        for block_index, block in enumerate(blocks):
            if getattr(block, "_petinct_checkpoint_wrapped", False):
                raise ValueError(
                    f"{container_name}.blocks[{block_index}] is already "
                    "checkpoint wrapped"
                )
            original_forward = block.forward

            def checkpointed_forward(
                self,
                x,
                *,
                _original_forward=original_forward,
            ):
                if _activation_checkpointing_active(self):
                    return checkpoint(
                        _original_forward,
                        x,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                return _original_forward(x)

            block.forward = MethodType(checkpointed_forward, block)
            block._petinct_checkpoint_wrapped = True


def enable_maisi_segment_checkpointing(autoencoder) -> int:
    """Checkpoint MAISI residual blocks in two arithmetic-preserving segments.

    A complete ``MaisiResBlock`` checkpoint still records all activations
    inside the block during backward recomputation. At full PET resolution,
    the two normalization/activation/convolution cascades can together occupy
    almost an A800. Splitting the unchanged forward into
    ``norm1->SiLU->conv1`` and ``norm2->SiLU->conv2`` bounds the live
    recomputation graph to one cascade. Non-residual MAISI blocks retain the
    existing whole-block checkpoint behavior.
    """
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
        MaisiResBlock,
    )
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint

    wrapped = 0
    for container_name in ("encoder", "decoder"):
        container = getattr(autoencoder, container_name)
        blocks = getattr(container, "blocks", None)
        if blocks is None:
            raise ValueError(
                f"{container_name} has no MAISI blocks ModuleList"
            )
        for block_index, block in enumerate(blocks):
            if getattr(block, "_petinct_checkpoint_wrapped", False):
                raise ValueError(
                    f"{container_name}.blocks[{block_index}] is already "
                    "checkpoint wrapped"
                )
            original_forward = block.forward

            if isinstance(block, MaisiResBlock):

                def segmented_forward(
                    self,
                    x,
                    *,
                    _original_forward=original_forward,
                ):
                    if not _activation_checkpointing_active(self):
                        return _original_forward(x)

                    def first_segment(value):
                        context = (
                            torch.cuda.device(value.device)
                            if value.device.type == "cuda"
                            else nullcontext()
                        )
                        with context:
                            return self.conv1(F.silu(self.norm1(value)))

                    def second_segment(value):
                        context = (
                            torch.cuda.device(value.device)
                            if value.device.type == "cuda"
                            else nullcontext()
                        )
                        with context:
                            return self.conv2(F.silu(self.norm2(value)))

                    h = checkpoint(
                        first_segment,
                        x,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    h = checkpoint(
                        second_segment,
                        h,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    shortcut = (
                        self.nin_shortcut(x)
                        if self.in_channels != self.out_channels
                        else x
                    )
                    return shortcut + h

                block.forward = MethodType(segmented_forward, block)
                block._petinct_segment_checkpointed = True
            else:

                def checkpointed_forward(
                    self,
                    x,
                    *,
                    _original_forward=original_forward,
                ):
                    if _activation_checkpointing_active(self):
                        def device_guarded_forward(value):
                            context = (
                                torch.cuda.device(value.device)
                                if value.device.type == "cuda"
                                else nullcontext()
                            )
                            with context:
                                return _original_forward(value)

                        return checkpoint(
                            device_guarded_forward,
                            x,
                            use_reentrant=False,
                            preserve_rng_state=True,
                        )
                    return _original_forward(x)

                block.forward = MethodType(checkpointed_forward, block)

            block._petinct_checkpoint_wrapped = True
            wrapped += 1

    if wrapped == 0:
        raise ValueError("no MAISI encoder/decoder blocks found")
    return wrapped


def enable_maisi_operation_checkpointing(
    autoencoder,
    upsample_computation_dtype: str = "preserve",
    upsample_channel_splits: int = 1,
    audit_upsample_operations: bool = False,
) -> int:
    """Checkpoint every MAISI residual operation independently.

    At a full 400x400 in-plane field, even a checkpointed
    ``norm->SiLU->conv`` cascade retains the recomputed normalization and
    activation while Conv3d allocates its full input gradient. Separating all
    six unchanged operations in a residual block bounds the live
    recomputation graph to one operation. Forward arithmetic, parameters,
    residual branches and state-dict keys remain unchanged.
    """
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
        MaisiDownsample,
        MaisiResBlock,
        MaisiUpsample,
    )
    from torch.nn import functional as F
    from torch.utils.checkpoint import checkpoint

    if upsample_computation_dtype not in {"preserve", "float16"}:
        raise ValueError(
            "upsample_computation_dtype must be 'preserve' or 'float16', "
            f"got {upsample_computation_dtype!r}"
        )
    if upsample_channel_splits < 1:
        raise ValueError(
            "upsample_channel_splits must be >= 1, "
            f"got {upsample_channel_splits}"
        )

    def device_guarded(function):
        def guarded(value):
            context = (
                torch.cuda.device(value.device)
                if value.device.type == "cuda"
                else nullcontext()
            )
            with context:
                return function(value)

        return guarded

    wrapped = 0
    for container_name in ("encoder", "decoder"):
        container = getattr(autoencoder, container_name)
        blocks = getattr(container, "blocks", None)
        if blocks is None:
            raise ValueError(
                f"{container_name} has no MAISI blocks ModuleList"
            )
        for block_index, block in enumerate(blocks):
            if getattr(block, "_petinct_checkpoint_wrapped", False):
                raise ValueError(
                    f"{container_name}.blocks[{block_index}] is already "
                    "checkpoint wrapped"
                )
            original_forward = block.forward
            if isinstance(block, MaisiResBlock):

                def operation_checkpointed_forward(
                    self,
                    x,
                    *,
                    _original_forward=original_forward,
                ):
                    if not _activation_checkpointing_active(self):
                        return _original_forward(x)

                    h = checkpoint(
                        device_guarded(self.norm1),
                        x,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    h = checkpoint(
                        device_guarded(F.silu),
                        h,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    h = checkpoint(
                        device_guarded(self.conv1),
                        h,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    h = checkpoint(
                        device_guarded(self.norm2),
                        h,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    h = checkpoint(
                        device_guarded(F.silu),
                        h,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    h = checkpoint(
                        device_guarded(self.conv2),
                        h,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    shortcut = (
                        self.nin_shortcut(x)
                        if self.in_channels != self.out_channels
                        else x
                    )
                    return shortcut + h

                block.forward = MethodType(
                    operation_checkpointed_forward,
                    block,
                )
                block._petinct_operation_checkpointed = True
            elif isinstance(block, MaisiUpsample):

                def operation_checkpointed_upsample(
                    self,
                    x,
                    *,
                    _original_forward=original_forward,
                ):
                    if not _activation_checkpointing_active(self):
                        return _original_forward(x)
                    if self.use_convtranspose:
                        return checkpoint(
                            device_guarded(self.conv),
                            x,
                            use_reentrant=False,
                            preserve_rng_state=True,
                        )

                    def interpolate(value):
                        if upsample_computation_dtype == "float16":
                            # CUDA autocast places trilinear interpolation on
                            # its FP32 list. Merely casting ``value`` is
                            # therefore insufficient: autocast immediately
                            # promotes it again, producing a 25-GiB
                            # full-resolution buffer for this model. Disable
                            # autocast only around the unchanged interpolation
                            # operator so its explicit FP16 input is honored.
                            with torch.autocast(
                                device_type=value.device.type,
                                enabled=False,
                            ):
                                return F.interpolate(
                                    value.to(dtype=torch.float16),
                                    scale_factor=2.0,
                                    mode="trilinear",
                                )
                        return F.interpolate(
                            value,
                            scale_factor=2.0,
                            mode="trilinear",
                        )

                    # Trilinear interpolation is channel independent. CUDA's
                    # FP16 backward nevertheless allocates an FP32
                    # accumulation buffer shaped like its complete output.
                    # Independent channel checkpoints preserve the exact
                    # operator and channel order while bounding that temporary
                    # buffer to one channel group.
                    channel_chunks = torch.chunk(
                        x,
                        min(upsample_channel_splits, x.shape[1]),
                        dim=1,
                    )
                    upsampled_chunks = [
                        checkpoint(
                            device_guarded(interpolate),
                            chunk,
                            use_reentrant=False,
                            preserve_rng_state=True,
                        )
                        for chunk in channel_chunks
                    ]

                    if audit_upsample_operations:
                        def audit_gradient(label, gradient):
                            device = gradient.device
                            print(
                                f"[upsample-backward] {label} "
                                f"grad={tuple(gradient.shape)}:"
                                f"{gradient.dtype} device={device} "
                                f"alloc_gib="
                                f"{torch.cuda.memory_allocated(device) / 2**30:.3f} "
                                f"reserved_gib="
                                f"{torch.cuda.memory_reserved(device) / 2**30:.3f}",
                                flush=True,
                            )
                            return gradient

                        for chunk_index, upsampled in enumerate(
                            upsampled_chunks
                        ):
                            upsampled.register_hook(
                                lambda gradient, index=chunk_index: (
                                    audit_gradient(
                                        f"interpolate_chunk={index}",
                                        gradient,
                                    )
                                )
                            )

                    if upsample_computation_dtype == "float16":
                        with torch.autocast(
                            device_type=x.device.type,
                            enabled=False,
                        ):
                            h = torch.cat(upsampled_chunks, dim=1)
                    else:
                        h = torch.cat(upsampled_chunks, dim=1)
                    if audit_upsample_operations:
                        h.register_hook(
                            lambda gradient: audit_gradient(
                                "concatenated_interpolate_output",
                                gradient,
                            )
                        )
                    return checkpoint(
                        device_guarded(self.conv),
                        h,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )

                block.forward = MethodType(
                    operation_checkpointed_upsample,
                    block,
                )
                block._petinct_operation_checkpointed = True
            elif isinstance(block, MaisiDownsample):

                def operation_checkpointed_downsample(
                    self,
                    x,
                    *,
                    _original_forward=original_forward,
                ):
                    if not _activation_checkpointing_active(self):
                        return _original_forward(x)
                    h = checkpoint(
                        device_guarded(
                            lambda value: F.pad(
                                value,
                                self.pad,
                                mode="constant",
                                value=0.0,
                            )
                        ),
                        x,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                    return checkpoint(
                        device_guarded(self.conv),
                        h,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )

                block.forward = MethodType(
                    operation_checkpointed_downsample,
                    block,
                )
                block._petinct_operation_checkpointed = True
            else:

                def checkpointed_forward(
                    self,
                    x,
                    *,
                    _original_forward=original_forward,
                ):
                    if _activation_checkpointing_active(self):
                        return checkpoint(
                            device_guarded(_original_forward),
                            x,
                            use_reentrant=False,
                            preserve_rng_state=True,
                        )
                    return _original_forward(x)

                block.forward = MethodType(checkpointed_forward, block)
            block._petinct_checkpoint_wrapped = True
            wrapped += 1

    if wrapped == 0:
        raise ValueError("no MAISI encoder/decoder blocks found")
    return wrapped


def enable_torch_group_norm(
    autoencoder,
    computation_dtype: str = "float32",
) -> int:
    """Use PyTorch's fused GroupNorm formula in MAISI GroupNorm modules.

    MAISI's released implementation normalizes every group into a Python list
    and concatenates all groups afterward. For a full in-plane PET volume,
    that concatenation temporarily requires several copies of a multi-gigabyte
    activation. ``torch.nn.functional.group_norm`` computes the same
    normalization and uses the existing affine parameters, without changing
    the module tree or any ``state_dict`` key.

    ``computation_dtype='float32'`` matches MAISI with
    ``norm_float16=False``. ``'float16'`` activates MAISI's released
    ``norm_float16`` memory mode under AMP and halves the full-resolution
    normalization tensor. This option is memory engineering, not a model or
    objective change.
    """
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
        MaisiGroupNorm3D,
    )
    from torch.nn import functional as F

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    if computation_dtype not in dtype_by_name:
        raise ValueError(
            "computation_dtype must be 'float32' or 'float16', got "
            f"{computation_dtype!r}"
        )
    target_dtype = dtype_by_name[computation_dtype]

    wrapped = 0
    for module in autoencoder.modules():
        if not isinstance(module, MaisiGroupNorm3D):
            continue
        if getattr(module, "_petinct_torch_group_norm", False):
            raise ValueError(
                "MAISI GroupNorm is already using the torch implementation"
            )

        def torch_group_norm_forward(self, input):
            if input.ndim != 5:
                raise ValueError("Expected a 5D tensor")
            # GroupNorm is on PyTorch autocast's float32 list. Disable the
            # surrounding AMP context here so MAISI's explicit float16 memory
            # mode is not silently promoted back to the 21.97-GiB float32
            # tensor that this option is designed to avoid.
            with torch.autocast(device_type=input.device.type, enabled=False):
                return F.group_norm(
                    input.to(dtype=target_dtype),
                    self.num_groups,
                    (
                        self.weight.to(dtype=target_dtype)
                        if self.weight is not None
                        else None
                    ),
                    (
                        self.bias.to(dtype=target_dtype)
                        if self.bias is not None
                        else None
                    ),
                    self.eps,
                )

        module.forward = MethodType(torch_group_norm_forward, module)
        module._petinct_torch_group_norm = True
        wrapped += 1

    if wrapped == 0:
        raise ValueError("no MaisiGroupNorm3D modules found")
    return wrapped


def enable_encoder_decoder_model_parallel(
    autoencoder,
    encoder_device,
    decoder_device,
    output_device=None,
    decoder_high_device=None,
    decoder_split_index=None,
    decoder_middle_device=None,
    decoder_second_split_index=None,
) -> None:
    """Place the MAISI encoder and decoder on separate devices.

    The module tree, parameters, state-dict keys and VAE arithmetic are
    unchanged. Only the latent tensor is copied to ``decoder_device`` and the
    one-channel reconstruction is copied back to ``output_device``. This
    removes simultaneous encoder and decoder full-resolution gradients from a
    single GPU while keeping all losses and the discriminator on the encoder
    device.

    The caller must apply this after all whole-model ``.to(...)`` calls.
    """
    encoder_device = torch.device(encoder_device)
    decoder_device = torch.device(decoder_device)
    decoder_high_device = (
        torch.device(decoder_high_device)
        if decoder_high_device is not None
        else None
    )
    decoder_middle_device = (
        torch.device(decoder_middle_device)
        if decoder_middle_device is not None
        else None
    )
    output_device = torch.device(output_device or encoder_device)
    if getattr(autoencoder, "_petinct_model_parallel", False):
        raise ValueError("autoencoder model parallelism is already enabled")

    autoencoder.encoder.to(encoder_device)
    autoencoder.quant_conv_mu.to(encoder_device)
    autoencoder.quant_conv_log_sigma.to(encoder_device)
    autoencoder.post_quant_conv.to(decoder_device)
    autoencoder.decoder.to(decoder_device)
    if (decoder_high_device is None) != (decoder_split_index is None):
        raise ValueError(
            "decoder_high_device and decoder_split_index must be provided "
            "together"
        )
    if (decoder_middle_device is None) != (
        decoder_second_split_index is None
    ):
        raise ValueError(
            "decoder_middle_device and decoder_second_split_index must be "
            "provided together"
        )
    if decoder_middle_device is not None and decoder_high_device is None:
        raise ValueError(
            "a middle decoder stage also requires decoder_high_device"
        )
    if decoder_high_device is not None:
        blocks = getattr(autoencoder.decoder, "blocks", None)
        if blocks is None:
            raise ValueError("staged decoder model parallelism requires blocks")
        decoder_split_index = int(decoder_split_index)
        if not 0 < decoder_split_index < len(blocks):
            raise ValueError(
                f"decoder_split_index must be in [1,{len(blocks) - 1}], "
                f"got {decoder_split_index}"
            )
        if decoder_middle_device is not None:
            decoder_second_split_index = int(decoder_second_split_index)
            if not (
                decoder_split_index
                < decoder_second_split_index
                < len(blocks)
            ):
                raise ValueError(
                    "decoder_second_split_index must be greater than "
                    f"{decoder_split_index} and below {len(blocks)}, got "
                    f"{decoder_second_split_index}"
                )
            for block in blocks[
                decoder_split_index:decoder_second_split_index
            ]:
                block.to(decoder_middle_device)
            for block in blocks[decoder_second_split_index:]:
                block.to(decoder_high_device)
        else:
            for block in blocks[decoder_split_index:]:
                block.to(decoder_high_device)

        def staged_decoder_forward(self, value):
            for block_index, block in enumerate(self.blocks):
                if block_index < decoder_split_index:
                    block_device = decoder_device
                elif (
                    decoder_middle_device is not None
                    and block_index < decoder_second_split_index
                ):
                    block_device = decoder_middle_device
                else:
                    block_device = decoder_high_device
                context = (
                    torch.cuda.device(block_device)
                    if block_device.type == "cuda"
                    else nullcontext()
                )
                with context:
                    value = value.to(block_device)
                    value = block(value)
            return value

        autoencoder.decoder.forward = MethodType(
            staged_decoder_forward,
            autoencoder.decoder,
        )

    def active_device(device):
        return (
            torch.cuda.device(device)
            if device.type == "cuda"
            else nullcontext()
        )

    def parallel_encode(self, x):
        with active_device(encoder_device):
            x = x.to(encoder_device)
            if self.use_checkpoint:
                h = torch.utils.checkpoint.checkpoint(
                    self.encoder,
                    x,
                    use_reentrant=False,
                )
            else:
                h = self.encoder(x)
            z_mu = self.quant_conv_mu(h)
            z_log_var = self.quant_conv_log_sigma(h)
            z_log_var = torch.clamp(z_log_var, -30.0, 20.0)
            z_sigma = torch.exp(z_log_var / 2)
            return z_mu.to(output_device), z_sigma.to(output_device)

    def parallel_decode(self, z):
        with active_device(decoder_device):
            z = z.to(decoder_device)
            z = self.post_quant_conv(z)
            if self.use_checkpoint:
                z = torch.utils.checkpoint.checkpoint(
                    self.decoder,
                    z,
                    use_reentrant=False,
                )
            else:
                z = self.decoder(z)
            return z.to(output_device)

    autoencoder.encode = MethodType(parallel_encode, autoencoder)
    autoencoder.decode = MethodType(parallel_decode, autoencoder)
    autoencoder._petinct_model_parallel = True
    autoencoder._petinct_encoder_device = str(encoder_device)
    autoencoder._petinct_decoder_device = str(decoder_device)
    autoencoder._petinct_decoder_high_device = (
        str(decoder_high_device)
        if decoder_high_device is not None
        else None
    )
    autoencoder._petinct_decoder_middle_device = (
        str(decoder_middle_device)
        if decoder_middle_device is not None
        else None
    )
    autoencoder._petinct_decoder_split_index = decoder_split_index
    autoencoder._petinct_decoder_second_split_index = (
        decoder_second_split_index
    )
    autoencoder._petinct_output_device = str(output_device)


def build_discriminator(spatial_dims: int = 3):
    """PatchDiscriminator exactly as MAISI's VAE tutorial (3 layers, 32 ch, INSTANCE norm)."""
    from monai.networks.nets import PatchDiscriminator

    return PatchDiscriminator(
        spatial_dims=spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    )


def build_loss_fns(recon_loss: str = "l1", spatial_dims: int = 3, with_perceptual: bool = True, device=None):
    """Return (intensity_loss, adv_loss, perceptual_loss) as in MAISI.

    perceptual_loss is moved to ``device`` when given — its lpips backbone registers
    ``shift``/``scale`` buffers that must live on the same device as the inputs (MAISI's
    tutorial does ``PerceptualLoss(...).to(device)``). It is None if the backbone cannot be
    fetched (e.g. offline / no torchvision), so the core pipeline can still be smoke-tested.
    """
    from torch.nn import L1Loss, MSELoss
    from monai.losses.adversarial_loss import PatchAdversarialLoss

    intensity = MSELoss() if recon_loss == "l2" else L1Loss(reduction="mean")
    adv = PatchAdversarialLoss(criterion="least_squares")
    perceptual = None
    if with_perceptual:
        try:
            from monai.losses.perceptual import PerceptualLoss

            perceptual = PerceptualLoss(
                spatial_dims=spatial_dims, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
            ).eval()
            if device is not None:
                perceptual = perceptual.to(device)  # lpips shift/scale buffers must match input device
        except Exception as e:  # offline / torchvision missing
            print(f"[warn] perceptual loss unavailable ({type(e).__name__}: {e}); proceeding without it")
            perceptual = None
    return intensity, adv, perceptual


def vae_train_step(
    autoencoder,
    discriminator,
    images,
    opt_g,
    opt_d,
    loss_fns,
    weights,
    amp=False,
    scaler_g=None,
    scaler_d=None,
    anatomy_labels=None,
    quantitative_loss_fn=None,
    saved_tensors_offload: str = "none",
    offload_pin_memory: bool = False,
    generator_backward_staging: str = "none",
    audit_gradients: bool = False,
    audit_loss_gradients: bool = False,
    audit_loss_gradients_only: bool = False,
):
    """One VAE step. Returns a scalar loss dict.

    VERBATIM MAISI loop when ``weights['adv'] > 0`` with a discriminator: BOTH optimizers are zeroed
    at the top, the generator backward leaks ``adv_weight*grad(gen_adv)`` into ``D.grad``, and MAISI
    KEEPS it (the D phase does NOT re-zero) — this weakens D and balances the GAN. Confirmed against
    maisi_train_vae_tutorial.ipynb + an independent codex audit (2026-06-05). With ``weights['adv']
    == 0`` or no discriminator: a pure VAE step (L1 + KL [+ perceptual]) — an ablation, not MAISI.

    ``quantitative_loss_fn`` is opt-in. When absent, the baseline arithmetic and
    optimizer path are unchanged. When present, it receives normalized
    ``(reconstruction, target, anatomy_labels)`` and adds the proposed
    anatomy-balanced auxiliary objective to the generator only.

    ``saved_tensors_offload='cpu'`` is an explicit training-memory strategy for
    full-field 3-D inputs. It preserves the forward arithmetic and model
    architecture, but moves tensors saved for backward to CPU until their
    gradients are evaluated. The default ``'none'`` is the baseline path.
    """
    intensity_loss, adv_loss, perceptual_loss = loss_fns
    device_type = images.device.type
    use_gan = weights["adv"] > 0 and discriminator is not None
    if saved_tensors_offload not in {"none", "cpu"}:
        raise ValueError(
            "saved_tensors_offload must be one of {'none', 'cpu'}, "
            f"got {saved_tensors_offload!r}"
        )
    if saved_tensors_offload == "cpu" and device_type != "cuda":
        raise ValueError(
            "saved_tensors_offload='cpu' is a CUDA training strategy; "
            f"received tensors on {images.device}"
        )
    if generator_backward_staging not in {"none", "outputs"}:
        raise ValueError(
            "generator_backward_staging must be one of {'none', 'outputs'}, "
            f"got {generator_backward_staging!r}"
        )
    if audit_loss_gradients_only and not audit_loss_gradients:
        raise ValueError(
            "audit_loss_gradients_only requires audit_loss_gradients=True"
        )

    def saved_tensor_context():
        if saved_tensors_offload == "cpu":
            return torch.autograd.graph.save_on_cpu(
                pin_memory=offload_pin_memory,
                device_type="cuda",
            )
        return nullcontext()

    opt_g.zero_grad(set_to_none=True)
    if use_gan:
        opt_d.zero_grad(set_to_none=True)  # MAISI zeros BOTH at the top; the generator backward then
        # leaks adv_weight*grad(gen_adv) into D.grad and MAISI KEEPS it (do NOT re-zero before the D
        # phase) — weakens D, balances the GAN. Verbatim maisi_train_vae_tutorial.ipynb.
    with saved_tensor_context():
        with torch.autocast(device_type=device_type, enabled=amp):
            if audit_loss_gradients_only:
                # The calibration target is d(loss head)/d(reconstruction), not
                # d(loss)/d(theta_AE). Keeping the codec forward outside
                # autograd avoids retaining its full-field 3-D activation graph
                # and guarantees this mode cannot update codec parameters.
                with torch.no_grad():
                    recon, z_mu, z_sigma = autoencoder(images)
                recon = recon.detach().requires_grad_(True)
            else:
                recon, z_mu, z_sigma = autoencoder(images)
            recons = intensity_loss(recon, images)
            kl = kl_loss(z_mu, z_sigma)
            p = perceptual_loss(recon.float(), images.float()) if perceptual_loss is not None else torch.zeros((), device=images.device)
            loss_g = recons + weights["kl"] * kl + weights["perceptual"] * p
            quantitative_parts = {}
            if quantitative_loss_fn is not None:
                if anatomy_labels is None:
                    raise ValueError("quantitative_loss_fn requires anatomy_labels")
                quantitative_loss, quantitative_parts = quantitative_loss_fn(
                    recon.float(),
                    images.float(),
                    anatomy_labels,
                )
                loss_g = loss_g + quantitative_loss
            if use_gan:
                logits_fake = discriminator(recon.contiguous().float())[-1]
                gen_adv = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + weights["adv"] * gen_adv
            else:
                gen_adv = torch.zeros((), device=images.device)

    gradient_metrics = {}
    if audit_loss_gradients:
        effective_output_terms = {
            "global_l1": recons,
            "perceptual": weights["perceptual"] * p,
            "adversarial": weights["adv"] * gen_adv,
        }
        if quantitative_loss_fn is not None:
            effective_output_terms.update(
                {
                    "region_l1": (
                        quantitative_loss_fn.region_l1_weight
                        * quantitative_parts["region_l1"]
                    ),
                    "region_wasserstein": (
                        quantitative_loss_fn.region_wasserstein_weight
                        * quantitative_parts["region_wasserstein"]
                    ),
                }
            )
        gradient_metrics.update(
            _output_loss_gradient_metrics(
                effective_output_terms,
                recon,
                gradient_scale=(
                    float(scaler_g.get_scale())
                    if amp and scaler_g is not None
                    else 1.0
                ),
            )
        )
    if audit_loss_gradients_only:
        metrics = {
            "recons": float(recons.detach()),
            "kl": float(kl.detach()),
            "perceptual": float(p.detach()),
            "gen_adv": float(gen_adv.detach()),
            "disc": 0.0,
            "loss_g": float(loss_g.detach()),
            "audit_only": 1.0,
        }
        metrics.update(
            {
                name: float(value.detach())
                for name, value in quantitative_parts.items()
            }
        )
        metrics.update(gradient_metrics)
        return metrics
    scaled_loss_g = (
        scaler_g.scale(loss_g)
        if amp and scaler_g is not None
        else loss_g
    )
    if generator_backward_staging == "outputs":
        # First differentiate the loss heads only as far as the VAE outputs.
        # This releases the perceptual, adversarial and regional-distribution
        # graphs before the high-resolution encoder/decoder is recomputed.
        # Include D parameters in the same autograd call to preserve MAISI's
        # intentional generator-phase discriminator-gradient leak exactly.
        discriminator_parameters = (
            tuple(
                parameter
                for parameter in discriminator.parameters()
                if parameter.requires_grad
            )
            if use_gan
            else ()
        )
        vae_outputs = (recon, z_mu, z_sigma)
        differentiable_outputs = tuple(
            output for output in vae_outputs if output.requires_grad
        )
        if not recon.requires_grad:
            raise RuntimeError(
                "generator output-gradient staging requires reconstruction "
                "to carry a gradient"
            )
        targets = (*differentiable_outputs, *discriminator_parameters)
        staged_gradients = torch.autograd.grad(
            scaled_loss_g,
            targets,
            allow_unused=True,
        )
        n_outputs = len(differentiable_outputs)
        output_gradients = staged_gradients[:n_outputs]
        if any(gradient is None for gradient in output_gradients):
            raise RuntimeError(
                "generator output-gradient staging produced an unused VAE "
                "output"
            )
        for parameter, gradient in zip(
            discriminator_parameters,
            staged_gradients[n_outputs:],
            strict=True,
        ):
            if gradient is not None:
                parameter.grad = gradient
        torch.autograd.backward(
            differentiable_outputs,
            output_gradients,
        )
    else:
        scaled_loss_g.backward()

    if amp and scaler_g is not None:
        scaler_g.unscale_(opt_g)  # MAISI
        if audit_gradients:
            gradient_metrics.update(_gradient_metrics(autoencoder))
        scaler_g.step(opt_g)
        scaler_g.update()
    else:
        if audit_gradients:
            gradient_metrics.update(_gradient_metrics(autoencoder))
        opt_g.step()

    if use_gan:
        with saved_tensor_context():
            with torch.autocast(device_type=device_type, enabled=amp):  # opt_d NOT re-zeroed (MAISI)
                logits_fake = discriminator(recon.contiguous().detach())[-1]
                d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = 0.5 * (d_fake + d_real)
        if amp and scaler_d is not None:
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
        else:
            loss_d.backward()
            opt_d.step()
    else:
        loss_d = torch.zeros((), device=images.device)

    metrics = {
        "recons": float(recons.detach()), "kl": float(kl.detach()), "perceptual": float(p.detach()),
        "gen_adv": float(gen_adv.detach()), "disc": float(loss_d.detach()), "loss_g": float(loss_g.detach()),
    }
    metrics.update(
        {
            name: float(value.detach())
            for name, value in quantitative_parts.items()
        }
    )
    metrics.update(gradient_metrics)
    return metrics


def _gradient_metrics(module) -> dict[str, float]:
    """Return synchronized generator-gradient diagnostics for gate runs."""
    total_squared = 0.0
    maximum = 0.0
    tensor_count = 0
    all_finite = True
    for parameter in module.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        tensor_count += 1
        detached = gradient.detach()
        finite = bool(torch.isfinite(detached).all().item())
        all_finite = all_finite and finite
        if finite:
            norm = float(torch.linalg.vector_norm(detached.float()).item())
            total_squared += norm * norm
            maximum = max(maximum, float(detached.abs().max().item()))
    gradient_l2 = total_squared**0.5 if all_finite and tensor_count else float("nan")
    gradient_max = maximum if all_finite and tensor_count else float("nan")
    return {
        "grad_l2": gradient_l2,
        "grad_max": gradient_max,
        "grad_finite": float(all_finite and tensor_count > 0),
        "grad_tensors": float(tensor_count),
    }


def _output_loss_gradient_metrics(
    components: dict[str, torch.Tensor],
    reconstruction: torch.Tensor,
    gradient_scale: float = 1.0,
) -> dict[str, float]:
    """Audit effective loss-head gradients at the VAE reconstruction output.

    AMP training differentiates a GradScaler-multiplied generator loss and
    unscales parameter gradients afterwards. For full-field FP16 volumes,
    directly differentiating an unscaled mean loss can underflow at the
    reconstruction tensor even though the real scaled training path does not.
    Apply the same common scale here, then report gradients divided by it.
    """
    if not math.isfinite(gradient_scale) or gradient_scale <= 0:
        raise ValueError(
            f"gradient_scale must be finite and positive, got {gradient_scale}"
        )
    metrics: dict[str, float] = {
        "output_grad_audit_scale": float(gradient_scale)
    }
    for name, component in components.items():
        if not component.requires_grad:
            metrics[f"output_grad_{name}_l2"] = 0.0
            metrics[f"output_grad_{name}_max"] = 0.0
            metrics[f"output_grad_{name}_finite"] = 1.0
            continue
        gradient = torch.autograd.grad(
            component * gradient_scale,
            reconstruction,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            metrics[f"output_grad_{name}_l2"] = 0.0
            metrics[f"output_grad_{name}_max"] = 0.0
            metrics[f"output_grad_{name}_finite"] = 1.0
            continue
        detached = gradient.detach().float() / gradient_scale
        finite = bool(torch.isfinite(detached).all().item())
        metrics[f"output_grad_{name}_finite"] = float(finite)
        metrics[f"output_grad_{name}_l2"] = (
            float(torch.linalg.vector_norm(detached).item())
            if finite
            else float("nan")
        )
        metrics[f"output_grad_{name}_max"] = (
            float(detached.abs().max().item())
            if finite
            else float("nan")
        )
    return metrics


def smoke_test(patch: int = 32, n_steps: int = 2, device: str | None = None):
    """CPU smoke test (paper-sop Phase 3): build the MAISI VAE + discriminator, run a
    few steps on synthetic SUV-like data, assert finite losses. Verifies the pipeline
    end-to-end before committing GPU hours on BJMU."""
    import json
    import math
    import os

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs", "vae_pet_maisi.json"))
    cfg = json.load(open(cfg_path))
    ae_def = dict(cfg["autoencoder_def"])
    tr = cfg["autoencoder_train"]
    # norm_float16 (TSP-inference fp16 norm) is forced off inside build_autoencoder for AMP training

    autoencoder = build_autoencoder(ae_def, num_splits=1).to(dev)
    discriminator = build_discriminator(cfg["spatial_dims"]).to(dev)
    loss_fns = build_loss_fns(tr["recon_loss"], cfg["spatial_dims"], with_perceptual=True, device=dev)
    weights = {"kl": tr["kl_weight"], "perceptual": tr["perceptual_weight"], "adv": tr["adv_weight"]}
    opt_g = torch.optim.Adam(autoencoder.parameters(), lr=tr["lr"], eps=1e-8)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=tr["lr"], eps=1e-8)

    n_params = sum(p.numel() for p in autoencoder.parameters())
    print(f"AutoencoderKlMaisi: {n_params/1e6:.1f}M params | perceptual: {'on' if loss_fns[2] is not None else 'OFF(no torchvision)'} | device {dev} | patch {patch}^3")

    for step in range(n_steps):
        images = torch.rand(1, 1, patch, patch, patch, device=dev)  # synthetic SUV in [0,1]
        out = vae_train_step(autoencoder, discriminator, images, opt_g, opt_d, loss_fns, weights, amp=False)
        for k, v in out.items():
            assert math.isfinite(v), f"non-finite {k} at step {step}: {out}"
        print(f"step {step}: " + "  ".join(f"{k}={v:.4f}" for k, v in out.items()))
    print("SMOKE_OK")


if __name__ == "__main__":
    smoke_test()
