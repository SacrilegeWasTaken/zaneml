const std = @import("std");
const backend_mod = @import("backend.zig");
const Backend = backend_mod.Backend;
const Optimizer = @import("optimizer.zig").Optimizer;

/// Learnable positional embeddings.
///
/// forward:  out[t] = input[t] + embed[t]
/// backward: grad_embed[t] += grad_out[t]  (identity w.r.t. input)
///           grad_in[t]     = grad_out[t]
///
/// Size: max_seq x d_model parameters stored as struct fields (not heap).
pub fn PositionalEmbedding(comptime backend: Backend, comptime d_model: usize, comptime max_seq: usize) type {
    const Impl    = backend_mod.PositionalEmbeddingImpl(backend);
    const OptImpl = backend_mod.OptimizerImpl(backend);

    return struct {
        embed:      [max_seq * d_model]f32,
        grad_embed: [max_seq * d_model]f32,

        /// Adam moment buffers
        m_embed: [max_seq * d_model]f32,
        v_embed: [max_seq * d_model]f32,

        const Self = @This();

        /// Initialize with small random values (scale=0.02).
        pub fn init() Self {
            var self: Self = undefined;
            var rng = std.Random.DefaultPrng.init(12345);
            const scale = 0.02;
            for (&self.embed) |*e| e.* = rng.random().floatNorm(f32) * scale;
            @memset(&self.grad_embed, 0);
            @memset(&self.m_embed, 0);
            @memset(&self.v_embed, 0);
            return self;
        }

        /// Forward pass: input/output are [seq * d_model]; seq is inferred from input.len.
        pub fn forward(self: *Self, input: []const f32, output: []f32) void {
            Impl.forward(input, output, self.embed[0..input.len]);
        }

        /// Backward pass: passes gradient through (identity) and accumulates grad_embed.
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) void {
            _ = input;
            Impl.backward(grad_out, grad_in, self.grad_embed[0..grad_out.len]);
        }

        /// Update embeddings using the given optimizer. t is the 1-based step counter.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            OptImpl.update(opt, t, lr, &self.embed, &self.grad_embed, &self.m_embed, &self.v_embed);
        }

        /// Returns the sum of squared gradients.
        pub fn gradNormSq(self: *const Self) f32 {
            var sum: f32 = 0;
            for (self.grad_embed) |g| sum += g * g;
            return sum;
        }

        /// Multiply all gradients by s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            for (&self.grad_embed) |*g| g.* *= s;
        }
    };
}
