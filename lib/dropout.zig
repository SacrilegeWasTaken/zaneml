const std = @import("std");
const optimizer_mod = @import("optimizer.zig");
const Optimizer = optimizer_mod.Optimizer;

/// Inverted dropout: during training scales kept activations by 1/(1-rate).
/// Stores binary mask for backward pass.
/// Set training=false at inference time.
pub fn Dropout(comptime max_size: usize, comptime rate: f32) type {
    return struct {
        mask:     [max_size]bool,
        training: bool,
        rng:      std.Random.DefaultPrng,

        const Self = @This();

        /// Initialize with a random seed.
        pub fn init(seed: u64) Self {
            return .{
                .mask     = [_]bool{true} ** max_size,
                .training = true,
                .rng      = std.Random.DefaultPrng.init(seed),
            };
        }

        /// Forward pass. If not training or rate==0, copies input unchanged.
        pub fn forward(self: *Self, input: []const f32, output: []f32) void {
            if (!self.training or rate == 0.0) {
                @memcpy(output[0..input.len], input);
                return;
            }
            const scale = 1.0 / (1.0 - rate);
            for (input, output[0..input.len], self.mask[0..input.len]) |x, *o, *keep| {
                keep.* = self.rng.random().float(f32) >= rate;
                o.* = if (keep.*) x * scale else 0;
            }
        }

        /// Backward pass: propagates gradient through the saved mask.
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) void {
            _ = input;
            const scale = 1.0 / (1.0 - rate);
            for (grad_in, grad_out, self.mask[0..grad_out.len]) |*gi, go, keep| {
                gi.* = if (keep) go * scale else 0;
            }
        }

        /// No-op: Dropout has no learnable parameters.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            _ = .{ self, opt, lr, t };
        }

        /// Returns 0: Dropout has no gradients.
        pub fn gradNormSq(self: *const Self) f32 {
            _ = self;
            return 0;
        }

        /// No-op: Dropout has no gradients to scale.
        pub fn scaleGrads(self: *Self, s: f32) void {
            _ = .{ self, s };
        }
    };
}
