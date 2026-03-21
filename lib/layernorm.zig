const std = @import("std");
const backend_mod = @import("backend.zig");
const Backend = backend_mod.Backend;
const optimizer_mod = @import("optimizer.zig");
const Optimizer = optimizer_mod.Optimizer;

/// Layer Normalization.
/// Normalizes a feature vector x of size `dim`:
///   y = (x - mean) / sqrt(var + eps) * gamma + beta
/// gamma and beta are learnable parameters, initialized to 1 and 0.
pub fn LayerNorm(comptime backend: Backend, comptime dim: usize) type {
    return struct {
        gamma: [dim]f32,       // scale, learnable
        beta:  [dim]f32,       // shift, learnable
        eps:   f32,

        // gradients for backward
        grad_gamma: [dim]f32,
        grad_beta:  [dim]f32,

        // Adam moment buffers for gamma and beta
        m_gamma: [dim]f32,
        v_gamma: [dim]f32,
        m_beta:  [dim]f32,
        v_beta:  [dim]f32,

        // cache for backward: normalized input before gamma/beta
        x_norm_buf: [dim]f32,
        // mean and reciprocal std from last forward (needed in backward)
        last_mean: f32,
        last_rstd: f32,        // 1 / sqrt(var + eps)

        const Self    = @This();
        const Impl    = backend_mod.LayerNormImpl(backend);
        const OptImpl = backend_mod.OptimizerImpl(backend);

        /// Initialize LayerNorm with given epsilon. gamma=1, beta=0.
        pub fn init(eps: f32) Self {
            var self: Self = undefined;
            self.eps = eps;
            self.last_mean = 0;
            self.last_rstd = 0;
            @memset(&self.grad_gamma, 0);
            @memset(&self.grad_beta,  0);
            @memset(&self.x_norm_buf, 0);
            @memset(&self.m_gamma, 0);
            @memset(&self.v_gamma, 0);
            @memset(&self.m_beta,  0);
            @memset(&self.v_beta,  0);
            // gamma = 1, beta = 0
            for (&self.gamma) |*g| g.* = 1.0;
            @memset(&self.beta, 0);
            return self;
        }

        /// Forward pass: input and output are slices of length dim.
        pub fn forward(self: *Self, input: []const f32, out: []f32) void {
            std.debug.assert(input.len == dim and out.len == dim);
            Impl.forward(input, out, &self.gamma, &self.beta, self.eps, &self.x_norm_buf, &self.last_mean, &self.last_rstd);
        }

        /// Backward pass: accumulates grad_gamma, grad_beta and writes grad_in.
        /// grad_out and grad_in are slices of length dim.
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) void {
            _ = input;
            std.debug.assert(grad_out.len == dim and grad_in.len == dim);
            Impl.backward(grad_out, grad_in, &self.grad_gamma, &self.grad_beta, &self.x_norm_buf, &self.gamma, self.last_rstd);
        }

        /// Update gamma and beta using the given optimizer. t is the 1-based step counter.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            OptImpl.update(opt, t, lr, &self.gamma, &self.grad_gamma, &self.m_gamma, &self.v_gamma);
            OptImpl.update(opt, t, lr, &self.beta,  &self.grad_beta,  &self.m_beta,  &self.v_beta);
        }

        /// Returns the sum of squared gradients.
        pub fn gradNormSq(self: *const Self) f32 {
            var sum: f32 = 0;
            for (self.grad_gamma) |g| sum += g * g;
            for (self.grad_beta)  |g| sum += g * g;
            return sum;
        }

        /// Multiply all gradients by s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            for (&self.grad_gamma) |*g| g.* *= s;
            for (&self.grad_beta)  |*g| g.* *= s;
        }
    };
}
