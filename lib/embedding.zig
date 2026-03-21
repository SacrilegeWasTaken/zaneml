const std = @import("std");
const backend_mod = @import("backend.zig");
const Backend = backend_mod.Backend;
const Optimizer = @import("optimizer.zig").Optimizer;

/// Token embedding lookup table.
/// Input is discrete token indices ([]const u32), output is [indices.len * d_model]f32.
///
/// NOTE: This module does NOT fit the standard forward([]const f32, []f32) interface
/// because input is discrete indices. Do NOT use with Network directly.
pub fn Embedding(comptime backend: Backend, comptime vocab_size: usize, comptime d_model: usize) type {
    const Impl    = backend_mod.EmbeddingImpl(backend);
    const OptImpl = backend_mod.OptimizerImpl(backend);

    return struct {
        table:      [vocab_size * d_model]f32,
        grad_table: [vocab_size * d_model]f32,
        /// Adam moment buffers for the embedding table
        m_table: [vocab_size * d_model]f32,
        v_table: [vocab_size * d_model]f32,

        const Self = @This();

        /// Initialize with Xavier uniform: scale = sqrt(1 / d_model).
        pub fn init() Self {
            var self: Self = undefined;
            var rng = std.Random.DefaultPrng.init(@bitCast(std.time.milliTimestamp()));
            const scale = @sqrt(1.0 / @as(f32, @floatFromInt(d_model)));
            for (&self.table) |*t| t.* = rng.random().floatNorm(f32) * scale;
            @memset(&self.grad_table, 0);
            @memset(&self.m_table, 0);
            @memset(&self.v_table, 0);
            return self;
        }

        /// Forward lookup: output[i * d_model .. (i+1) * d_model] = table[indices[i]]
        pub fn forward(self: *Self, indices: []const u32, output: []f32) void {
            Impl.forward(d_model, indices, output, &self.table);
        }

        /// Accumulate gradients into grad_table.
        /// grad_out must be [indices.len * d_model].
        pub fn backward(self: *Self, indices: []const u32, grad_out: []const f32) void {
            Impl.backward(d_model, indices, grad_out, &self.grad_table);
        }

        /// Update the embedding table using the given optimizer. t is the 1-based step counter.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            OptImpl.update(opt, t, lr, &self.table, &self.grad_table, &self.m_table, &self.v_table);
        }

        /// Returns the sum of squared gradients.
        pub fn gradNormSq(self: *const Self) f32 {
            var sum: f32 = 0;
            for (self.grad_table) |g| sum += g * g;
            return sum;
        }

        /// Multiply all gradients by s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            for (&self.grad_table) |*g| g.* *= s;
        }

        /// Zero all gradients.
        pub fn zeroGrads(self: *Self) void {
            @memset(&self.grad_table, 0);
        }
    };
}
