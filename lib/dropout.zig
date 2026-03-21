const std = @import("std");
const backend_mod = @import("backend.zig");
const Backend = backend_mod.Backend;
const Optimizer = @import("optimizer.zig").Optimizer;

/// Inverted dropout: during training scales kept activations by 1/(1-rate).
/// Stores binary mask for backward pass.
/// Set training=false at inference time.
pub fn Dropout(comptime backend: Backend, comptime max_size: usize, comptime rate: f32) type {
    const Impl = backend_mod.DropoutImpl(backend);

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
            Impl.forward(input, output, &self.mask, rate, self.training, &self.rng);
        }

        /// Backward pass: propagates gradient through the saved mask.
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) void {
            _ = input;
            Impl.backward(grad_out, grad_in, &self.mask, rate);
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
