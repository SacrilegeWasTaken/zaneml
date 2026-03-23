const std = @import("std");

pub const Activation           = @import("activation.zig").Activation;
pub const Tensor               = @import("tensor.zig").Tensor;
pub const Backend              = @import("backend.zig").Backend;
pub const Layer                = @import("layer.zig").Layer;
pub const LayerNorm            = @import("layernorm.zig").LayerNorm;
pub const MultiHeadAttention   = @import("attention.zig").MultiHeadAttention;
pub const Sequential           = @import("sequential.zig").Sequential;
pub const Residual             = @import("residual.zig").Residual;
pub const Network              = @import("network.zig").Network;
pub const TransformerBlock     = @import("transformer_block.zig").TransformerBlock;
pub const BlockConfig          = @import("transformer_block.zig").BlockConfig;
pub const TransformerStack     = @import("transformer_stack.zig").TransformerStack;
pub const PositionalEmbedding  = @import("positional_embedding.zig").PositionalEmbedding;

/// Optimizer and its sub-types
pub const Optimizer  = @import("optimizer.zig").Optimizer;

/// Learning rate schedule (also accessible via Network.LRSchedule)
pub const LRSchedule = @import("network.zig").LRSchedule;

/// Inverted dropout module
pub const Dropout    = @import("dropout.zig").Dropout;

/// RMS Layer Normalization
pub const RMSNorm    = @import("rmsnorm.zig").RMSNorm;

/// Token embedding lookup table
pub const Embedding  = @import("embedding.zig").Embedding;

/// Fixed-size Adam moment buffer helper
pub const MomentBuf  = @import("optimizer_state.zig").MomentBuf;

/// Tape-based automatic differentiation
pub const Tape           = @import("autograd.zig").Tape;
pub const AutogradTensor = @import("autograd.zig").Tensor;

/// Two-layer MLP trained via tape-based autograd (plug-in for Network)
pub const TapeMLP = @import("tape_mlp.zig").TapeMLP;

/// TransformerStack wrapped as a tape-compatible module (same GPU perf, autograd composable)
pub const TapeTransformerStack = @import("tape_transformer.zig").TapeTransformerStack;

/// Backend optimizer dispatch — use in custom models that implement updateWeights.
pub const OptimizerImpl = @import("backend.zig").OptimizerImpl;

/// Metal compute engine — GPU buffer management and kernel dispatch.
pub const MetalEngine = @import("metal/engine.zig").MetalEngine;

/// Return a pointer to the singleton Metal engine (initialises it on first call).
/// Only meaningful when compiling with a Metal-capable target.
pub fn getMetalEngine() !*MetalEngine {
    return @import("backend/metal.zig").getEngine();
}

