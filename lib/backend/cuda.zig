const std = @import("std");
const Activation = @import("../activation.zig").Activation;

pub const CudaBackend = struct {
    pub fn forward(
        weights: []const f32,
        input: []const f32,
        biases: []const f32,
        pre_activation_out: []f32,
        out: []f32,
        activation: Activation,
    ) void {
        _ = .{ weights, input, biases, pre_activation_out, out, activation };
        @panic("CUDA backend not implemented yet");
    }

    pub fn backward(
        allocator: std.mem.Allocator,
        weights: []const f32,
        input: []const f32,
        biases: []const f32,
        pre_activation: []const f32,
        grad_out: []const f32,
        grad_in: []f32,
        grad_w: []f32,
        grad_b: []f32,
        activation: Activation,
    ) !void {
        _ = .{ allocator, weights, input, biases, pre_activation, grad_out, grad_in, grad_w, grad_b, activation };
        @panic("CUDA backend not implemented yet");
    }
};
