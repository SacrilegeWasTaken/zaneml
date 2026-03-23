const std = @import("std");

const backend_mod = @import("backend.zig");
const Backend     = backend_mod.Backend;

const LayerNormT = @import("layernorm.zig").LayerNorm;
const RMSNormT   = @import("rmsnorm.zig").RMSNorm;
const LayerT     = @import("layer.zig").Layer;
const MhaT       = @import("attention.zig").MultiHeadAttention;
const Activation = @import("activation.zig").Activation;
const Optimizer  = @import("optimizer.zig").Optimizer;

/// Compile-time configuration for a TransformerBlock.
pub const BlockConfig = struct {
    /// Which normalization layer to use before each sub-layer.
    norm:           NormType,
    /// Activation function for the first FFN linear layer.
    ffn_activation: Activation,
    /// If true, attention uses a causal (lower-triangular) mask.
    causal:         bool,

    pub const NormType = enum { layer_norm, rms_norm };
};

/// Pre-LN Transformer Block:
///
///   out = FFN(Norm2(h)) + h
///   h   = Attn(Norm1(x)) + x
///
/// Compile-time parameters:
///   backend  -- compute backend
///   d_model  -- model dimension
///   n_heads  -- number of attention heads (d_model % n_heads == 0)
///   d_ff     -- FFN hidden size (typically 4 * d_model)
///   max_seq  -- maximum sequence length
///   cfg      -- BlockConfig: norm type, FFN activation, causal flag
///
/// The struct lives on the heap (init returns *Self).
/// input/output: flat slices [seq * d_model], row-major.
pub fn TransformerBlock(
    comptime backend: Backend,
    comptime d_model:  usize,
    comptime n_heads:  usize,
    comptime d_ff:     usize,
    comptime max_seq:  usize,
    comptime cfg:      BlockConfig,
) type {
    const Norm = switch (cfg.norm) {
        .layer_norm => LayerNormT(backend, d_model),
        .rms_norm   => RMSNormT(backend, d_model),
    };
    const Fc  = LayerT(backend);
    const Mha = MhaT(backend, d_model, n_heads, max_seq, cfg.causal);

    return struct {
        //  modules 
        norm1: Norm,
        norm2: Norm,
        attn:  *Mha,
        ffn1:  Fc,    // d_model -> d_ff,    cfg.ffn_activation
        ffn2:  Fc,    // d_ff    -> d_model,  linear

        //  intermediate forward activations 
        norm1_out: [max_seq * d_model]f32,
        h:         [max_seq * d_model]f32,
        norm2_out: [max_seq * d_model]f32,
        ffn1_out:  [max_seq * d_ff]f32,

        //  per-position norm caches (needed in backward) 
        // Both LayerNorm and RMSNorm expose x_norm_buf and last_rstd.
        norm1_xn:   [max_seq * d_model]f32,
        norm1_rstd: [max_seq]f32,
        norm2_xn:   [max_seq * d_model]f32,
        norm2_rstd: [max_seq]f32,

        //  per-position pre-activation Layer caches (needed in backward) ─
        ffn1_pre: [max_seq * d_ff]f32,
        ffn2_pre: [max_seq * d_model]f32,

        //  backward gradient buffers 
        grad_ffn1_out:  [max_seq * d_ff]f32,
        grad_norm2_out: [max_seq * d_model]f32,
        grad_h:         [max_seq * d_model]f32,
        grad_norm1_out: [max_seq * d_model]f32,

        //  scratch: temporary buffer for per-position norm backward 
        scratch: [d_model]f32,

        last_seq:  usize,
        allocator: std.mem.Allocator,

        const Self = @This();

        //  init / deinit 

        /// Allocate and initialize a TransformerBlock on the heap.
        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            self.allocator = allocator;
            self.last_seq  = 0;

            self.norm1 = Norm.init(1e-5);
            self.norm2 = Norm.init(1e-5);
            self.attn  = try Mha.init(allocator);
            self.ffn1  = try Fc.init(allocator, .{
                .n_in = d_model, .n_out = d_ff,    .activation = cfg.ffn_activation,
            });
            self.ffn2  = try Fc.init(allocator, .{
                .n_in = d_ff,    .n_out = d_model, .activation = .linear,
            });
            return self;
        }

        /// Free all resources.
        pub fn deinit(self: *Self) void {
            self.attn.deinit(self.allocator);
            self.ffn1.deinit();
            self.ffn2.deinit();
            self.allocator.destroy(self);
        }

        //  forward 

        /// Generic forward: seq is inferred from input.len / d_model.
        pub fn forward(self: *Self, input: []const f32, output: []f32) void {
            self.forwardSeq(input, output, input.len / d_model);
        }

        /// Explicit-seq forward.
        /// input  -- [seq * d_model], row-major
        /// output -- [seq * d_model]
        pub fn forwardSeq(self: *Self, input: []const f32, output: []f32, seq: usize) void {
            std.debug.assert(seq <= max_seq);
            std.debug.assert(input.len  == seq * d_model);
            std.debug.assert(output.len == seq * d_model);
            self.last_seq = seq;

            // sublayer 1: Norm1 -> Attn -> +x
            for (0..seq) |t| {
                const x  = input[t * d_model ..][0..d_model];
                const no = self.norm1_out[t * d_model ..][0..d_model];
                self.norm1.forward(x, no);
                @memcpy(self.norm1_xn[t * d_model ..][0..d_model], &self.norm1.x_norm_buf);
                self.norm1_rstd[t] = self.norm1.last_rstd;
            }

            self.attn.forward(
                self.norm1_out[0..seq * d_model],
                self.h[0..seq * d_model],
                seq,
            );

            for (self.h[0..seq * d_model], input) |*hi, xi| hi.* += xi;

            // sublayer 2: Norm2 -> FFN -> +h
            for (0..seq) |t| {
                const x  = self.h[t * d_model ..][0..d_model];
                const no = self.norm2_out[t * d_model ..][0..d_model];
                self.norm2.forward(x, no);
                @memcpy(self.norm2_xn[t * d_model ..][0..d_model], &self.norm2.x_norm_buf);
                self.norm2_rstd[t] = self.norm2.last_rstd;
            }

            for (0..seq) |t| {
                const n2o = self.norm2_out[t * d_model ..][0..d_model];
                const f1o = self.ffn1_out[t * d_ff ..][0..d_ff];
                self.ffn1.forward(n2o, f1o);
                @memcpy(self.ffn1_pre[t * d_ff ..][0..d_ff], self.ffn1.pre_activation_buffer);

                const f2o = output[t * d_model ..][0..d_model];
                self.ffn2.forward(f1o, f2o);
                @memcpy(self.ffn2_pre[t * d_model ..][0..d_model], self.ffn2.pre_activation_buffer);
            }

            for (output[0..seq * d_model], self.h[0..seq * d_model]) |*oi, hi| oi.* += hi;
        }

        //  backward 

        /// Backward pass.
        /// input    -- original forward input [seq * d_model]
        /// grad_out -- [seq * d_model]
        /// grad_in  -- [seq * d_model] (written)
        pub fn backward(
            self:     *Self,
            input:    []const f32,
            grad_out: []const f32,
            grad_in:  []f32,
        ) !void {
            const seq = self.last_seq;
            const smd = seq * d_model;

            @memcpy(self.grad_h[0..smd], grad_out[0..smd]);

            for (0..seq) |t| {
                @memcpy(self.ffn2.pre_activation_buffer, self.ffn2_pre[t * d_model ..][0..d_model]);
                try self.ffn2.backward(
                    self.ffn1_out[t * d_ff ..][0..d_ff],
                    grad_out[t * d_model ..][0..d_model],
                    self.grad_ffn1_out[t * d_ff ..][0..d_ff],
                );
            }

            for (0..seq) |t| {
                @memcpy(self.ffn1.pre_activation_buffer, self.ffn1_pre[t * d_ff ..][0..d_ff]);
                try self.ffn1.backward(
                    self.norm2_out[t * d_model ..][0..d_model],
                    self.grad_ffn1_out[t * d_ff ..][0..d_ff],
                    self.grad_norm2_out[t * d_model ..][0..d_model],
                );
            }

            for (0..seq) |t| {
                @memcpy(&self.norm2.x_norm_buf, self.norm2_xn[t * d_model ..][0..d_model]);
                self.norm2.last_rstd = self.norm2_rstd[t];
                self.norm2.backward(
                    self.h[t * d_model ..][0..d_model],
                    self.grad_norm2_out[t * d_model ..][0..d_model],
                    &self.scratch,
                );
                for (self.grad_h[t * d_model ..][0..d_model], &self.scratch) |*gh, s| gh.* += s;
            }

            self.attn.backward(
                self.norm1_out[0..smd],
                self.grad_h[0..smd],
                self.grad_norm1_out[0..smd],
            );

            @memcpy(grad_in[0..smd], self.grad_h[0..smd]);

            for (0..seq) |t| {
                @memcpy(&self.norm1.x_norm_buf, self.norm1_xn[t * d_model ..][0..d_model]);
                self.norm1.last_rstd = self.norm1_rstd[t];
                self.norm1.backward(
                    input[t * d_model ..][0..d_model],
                    self.grad_norm1_out[t * d_model ..][0..d_model],
                    &self.scratch,
                );
                for (grad_in[t * d_model ..][0..d_model], &self.scratch) |*gi, s| gi.* += s;
            }
        }

        //  weight update 

        /// Update all submodule weights using the given optimizer.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            self.norm1.updateWeights(opt, lr, t);
            self.norm2.updateWeights(opt, lr, t);
            self.attn.updateWeights(opt, lr, t);
            self.ffn1.updateWeights(opt, lr, t);
            self.ffn2.updateWeights(opt, lr, t);
        }

        /// Sum of squared gradients across all submodules.
        pub fn gradNormSq(self: *const Self) f32 {
            return self.norm1.gradNormSq()
                 + self.norm2.gradNormSq()
                 + self.attn.gradNormSq()
                 + self.ffn1.gradNormSq()
                 + self.ffn2.gradNormSq();
        }

        /// Scale gradients of all submodules by s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            self.norm1.scaleGrads(s);
            self.norm2.scaleGrads(s);
            self.attn.scaleGrads(s);
            self.ffn1.scaleGrads(s);
            self.ffn2.scaleGrads(s);
        }
    };
}
