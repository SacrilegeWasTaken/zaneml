const std = @import("std");
const backend_mod = @import("backend.zig");
const Backend = backend_mod.Backend;
const optimizer_mod = @import("optimizer.zig");
const Optimizer = optimizer_mod.Optimizer;

/// RMS Layer Normalization: y = x / rms(x) * gamma
/// where rms(x) = sqrt(mean(x^2) + eps)
///
/// gamma is a learnable per-dimension scale (initialized to 1).
/// No shift parameter (unlike LayerNorm).
pub fn RMSNorm(comptime backend: Backend, comptime dim: usize) type {
    _ = backend; // backend dispatch not needed for CPU-only implementation

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

            var mean_sq: f32 = 0;
            for (input) |x| mean_sq += x * x;
            mean_sq /= @as(f32, @floatFromInt(dim));

            const rms = @sqrt(mean_sq + self.eps);
            self.last_rstd = rms;

            for (input, out, &self.x_norm_buf, self.gamma) |x, *o, *xn, g| {
                xn.* = x / rms;
                o.* = g * xn.*;
            }
        }

        /// Backward pass.
        /// grad_in[i] = (1/rms) * gamma[i] * (grad_out[i] - x_norm[i] * mean(grad_out * x_norm))
        /// grad_gamma[i] += grad_out[i] * x_norm[i]
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) void {
            _ = input;
            std.debug.assert(grad_out.len == dim and grad_in.len == dim);

            // Accumulate grad_gamma
            for (grad_out, &self.x_norm_buf, &self.grad_gamma) |go, xn, *gg| {
                gg.* += go * xn;
            }

            // mean(grad_out * x_norm * gamma) used in the projection term
            var dot: f32 = 0;
            for (grad_out, &self.x_norm_buf, self.gamma) |go, xn, g| {
                dot += go * xn * g;
            }
            dot /= @as(f32, @floatFromInt(dim));

            const inv_rms = 1.0 / self.last_rstd;
            for (grad_in, grad_out, &self.x_norm_buf, self.gamma) |*gi, go, xn, g| {
                gi.* = inv_rms * (g * go - xn * dot);
            }
        }

        /// Update gamma using the given optimizer. t is the 1-based step counter.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            opt.update(t, lr, &self.gamma, &self.grad_gamma, &self.m_gamma, &self.v_gamma);
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
