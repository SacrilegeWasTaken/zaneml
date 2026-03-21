const std = @import("std");
const backend_mod = @import("backend.zig");
const Backend = backend_mod.Backend;
const Optimizer = @import("optimizer.zig").Optimizer;

/// RMS Layer Normalization: y = x / rms(x) * gamma
/// where rms(x) = sqrt(mean(x^2) + eps)
///
/// gamma is a learnable per-dimension scale (initialized to 1).
/// No shift parameter (unlike LayerNorm).
pub fn RMSNorm(comptime backend: Backend, comptime dim: usize) type {
    const Impl    = backend_mod.RMSNormImpl(backend);
    const OptImpl = backend_mod.OptimizerImpl(backend);

    return struct {
        gamma:      [dim]f32,
        grad_gamma: [dim]f32,
        /// Adam moment buffers for gamma
        m_gamma: [dim]f32,
        v_gamma: [dim]f32,
        eps:        f32,
        /// rms value from the last forward pass (needed in backward)
        last_rstd:   f32,
        /// normalized x buffer: x_norm[i] = x[i] / rms(x)
        x_norm_buf: [dim]f32,

        const Self = @This();

        /// Initialize RMSNorm with given epsilon. gamma=1.
        pub fn init(eps: f32) Self {
            var self: Self = undefined;
            self.eps = eps;
            self.last_rstd = 1.0;
            for (&self.gamma) |*g| g.* = 1.0;
            @memset(&self.grad_gamma, 0);
            @memset(&self.m_gamma, 0);
            @memset(&self.v_gamma, 0);
            @memset(&self.x_norm_buf, 0);
            return self;
        }

        /// Forward pass: out[i] = gamma[i] * x[i] / rms(x)
        pub fn forward(self: *Self, input: []const f32, out: []f32) void {
            std.debug.assert(input.len == dim and out.len == dim);
            Impl.forward(input, out, &self.gamma, self.eps, &self.x_norm_buf, &self.last_rstd);
        }

        /// Backward pass.
        /// grad_in[i] = (1/rms) * gamma[i] * (grad_out[i] - x_norm[i] * mean(grad_out * x_norm * gamma))
        /// grad_gamma[i] += grad_out[i] * x_norm[i]
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) void {
            _ = input;
            std.debug.assert(grad_out.len == dim and grad_in.len == dim);
            Impl.backward(grad_out, grad_in, &self.grad_gamma, &self.x_norm_buf, &self.gamma, self.last_rstd);
        }

        /// Update gamma using the given optimizer. t is the 1-based step counter.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            OptImpl.update(opt, t, lr, &self.gamma, &self.grad_gamma, &self.m_gamma, &self.v_gamma);
        }

        /// Returns the sum of squared gradients.
        pub fn gradNormSq(self: *const Self) f32 {
            var sum: f32 = 0;
            for (self.grad_gamma) |g| sum += g * g;
            return sum;
        }

        /// Multiply all gradients by s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            for (&self.grad_gamma) |*g| g.* *= s;
        }
    };
}
