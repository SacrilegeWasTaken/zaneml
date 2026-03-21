const std = @import("std");
const Activation = @import("../activation.zig").Activation;

pub const AneBackend = struct {
    pub fn forward(
        weights: []const f32,
        input: []const f32,
        biases: []const f32,
        pre_activation_out: []f32,
        out: []f32,
        activation: Activation,
    ) void {
        _ = .{ weights, input, biases, pre_activation_out, out, activation };
        @panic("ANE backend not implemented yet");
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
        @panic("ANE backend not implemented yet");
    }
};

pub const AneLayerNorm = struct {
    pub fn forward(
        input: []const f32, out: []f32, gamma: []const f32, beta: []const f32,
        eps: f32, x_norm_out: []f32, mean_out: *f32, rstd_out: *f32,
    ) void {
        _ = .{ input, out, gamma, beta, eps, x_norm_out, mean_out, rstd_out };
        @panic("ANE LayerNorm not implemented yet");
    }
    pub fn backward(
        grad_out: []const f32, grad_in: []f32, grad_gamma: []f32, grad_beta: []f32,
        x_norm: []const f32, gamma: []const f32, rstd: f32,
    ) void {
        _ = .{ grad_out, grad_in, grad_gamma, grad_beta, x_norm, gamma, rstd };
        @panic("ANE LayerNorm not implemented yet");
    }
};

pub const AneAttention = struct {
    pub fn forward(
        comptime d_model: usize, comptime n_heads: usize, comptime d_head: usize, comptime max_seq: usize,
        attn_scale: f32, seq: usize, input: []const f32, output: []f32,
        wq_f: []const f32, wk_f: []const f32, wv_f: []const f32, wo_f: []const f32,
        cache_q: []f32, cache_k: []f32, cache_v: []f32, cache_attn: []f32, cache_concat: []f32, scratch: []f32,
        causal: bool,
    ) void {
        _ = .{ d_model, n_heads, d_head, max_seq, attn_scale, seq, input, output, wq_f, wk_f, wv_f, wo_f, cache_q, cache_k, cache_v, cache_attn, cache_concat, scratch, causal };
        @panic("ANE Attention not implemented yet");
    }
    pub fn backward(
        comptime d_model: usize, comptime n_heads: usize, comptime d_head: usize,
        attn_scale: f32, seq: usize, input: []const f32, grad_out: []const f32, grad_in: []f32,
        wq_f: []const f32, wk_f: []const f32, wv_f: []const f32, wo_f: []const f32,
        cache_q: []const f32, cache_k: []const f32, cache_v: []const f32, cache_attn: []const f32, cache_concat: []const f32,
        grad_wq: []f32, grad_wk: []f32, grad_wv: []f32, grad_wo: []f32,
        grad_q: []f32, grad_k: []f32, grad_v: []f32, grad_concat: []f32, scratch: []f32,
    ) void {
        _ = .{ d_model, n_heads, d_head, attn_scale, seq, input, grad_out, grad_in, wq_f, wk_f, wv_f, wo_f, cache_q, cache_k, cache_v, cache_attn, cache_concat, grad_wq, grad_wk, grad_wv, grad_wo, grad_q, grad_k, grad_v, grad_concat, scratch };
        @panic("ANE Attention not implemented yet");
    }
};
