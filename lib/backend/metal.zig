const std = @import("std");
const Activation = @import("../activation.zig").Activation;
const engine_mod = @import("../metal/engine.zig");
const MetalEngine = engine_mod.MetalEngine;
const GpuBuffer = engine_mod.GpuBuffer;
const Grid = MetalEngine.Grid;

// ── Lazy singleton 

var g_engine: ?MetalEngine = null;

fn getEngine() !*MetalEngine {
    if (g_engine == null) {
        g_engine = try MetalEngine.init(std.heap.page_allocator);
    }
    return &g_engine.?;
}

/// Params structs — must match MSL layout (extern struct for C ABI).
const MatmulParams = extern struct { M: u32, K: u32, N: u32 };
const ElementwiseParams = extern struct { length: u32 };
const SGDParams = extern struct { length: u32, lr: f32 };
const AdamParams = extern struct {
    length: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: u32,
};

// MetalBackend — Linear layer forward/backward

pub const MetalBackend = struct {
    pub fn forward(
        weights: []const f32,
        input: []const f32,
        biases: []const f32,
        pre_activation_out: []f32,
        out: []f32,
        activation: Activation,
    ) void {
        const eng = getEngine() catch @panic("Metal init failed");
        eng.waitIfPending();
        const n_out = out.len;
        const n_in = input.len;

        // Cached buffers (weights/biases are stable pointers)
        const buf_w = eng.getOrUpload(weights);
        const buf_x = eng.getOrUpload(input);
        const buf_b = eng.getOrUpload(biases);

        // Transient output buffers
        const buf_z = eng.createBuffer(f32, n_out) catch @panic("OOM");
        defer eng.releaseBuffer(buf_z);

        eng.beginRecording();

        // z = W @ x
        const matmul_pipe = eng.getPipeline("matmul_f32") catch @panic("pipeline");
        eng.encodeTyped(matmul_pipe, &.{ buf_w, buf_x, buf_z }, MatmulParams{
            .M = @intCast(n_out), .K = @intCast(n_in), .N = 1,
        }, .{ .x = 1, .y = @intCast(n_out) }, null);

        // za = z + bias (reuse buf_z as input, write to a new buffer)
        const buf_za = eng.createBuffer(f32, n_out) catch @panic("OOM");
        defer eng.releaseBuffer(buf_za);
        const add_pipe = eng.getPipeline("add_f32") catch @panic("pipeline");
        eng.encodeTyped(add_pipe, &.{ buf_z, buf_b, buf_za }, ElementwiseParams{
            .length = @intCast(n_out),
        }, .{ .x = @intCast(n_out) }, null);

        // Activation
        const act_kernel = activationKernelName(activation);
        var buf_act_out: ?GpuBuffer = null;
        if (act_kernel) |name| {
            buf_act_out = eng.createBuffer(f32, n_out) catch @panic("OOM");
            const act_pipe = eng.getPipeline(name) catch @panic("pipeline");
            eng.encodeTyped(act_pipe, &.{ buf_za, buf_act_out.? }, ElementwiseParams{
                .length = @intCast(n_out),
            }, .{ .x = @intCast(n_out) }, null);
        }

        // Single GPU submission for entire forward
        eng.commitAndWait();

        eng.downloadTo(buf_za, pre_activation_out);
        if (buf_act_out) |bao| {
            eng.downloadTo(bao, out);
            eng.releaseBuffer(bao);
        } else {
            @memcpy(out, pre_activation_out);
        }
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
        _ = biases;
        const eng = try getEngine();
        eng.waitIfPending();
        const n_out = grad_out.len;
        const n_in = input.len;

        // delta = grad_out * activation'(pre_activation)  — CPU (elementwise, cheap)
        const cpu = @import("cpu.zig");
        const delta = try allocator.alloc(f32, n_out);
        defer allocator.free(delta);
        for (delta, pre_activation, grad_out) |*di, pre, go| {
            di.* = go * cpu.applyActivationBackward(activation, pre);
        }

        const buf_delta = eng.getOrUpload(delta);
        const buf_input = eng.getOrUpload(input);
        const buf_w = eng.getOrUpload(weights);

        // Transient output buffers
        const buf_gw = try eng.createBuffer(f32, n_out * n_in);
        defer eng.releaseBuffer(buf_gw);
        @memset(buf_gw.asSlice(f32), 0);

        const buf_gi = try eng.createBuffer(f32, n_in);
        defer eng.releaseBuffer(buf_gi);
        @memset(buf_gi.asSlice(f32), 0);

        eng.beginRecording();

        const matmul_pipe = try eng.getPipeline("matmul_f32");

        // grad_w = delta[n_out×1] @ input[1×n_in] → [n_out×n_in]
        eng.encodeTyped(matmul_pipe, &.{ buf_delta, buf_input, buf_gw }, MatmulParams{
            .M = @intCast(n_out), .K = 1, .N = @intCast(n_in),
        }, .{ .x = @intCast(n_in), .y = @intCast(n_out) }, null);

        // grad_in = delta[1×n_out] @ W[n_out×n_in] → [1×n_in]
        eng.encodeTyped(matmul_pipe, &.{ buf_delta, buf_w, buf_gi }, MatmulParams{
            .M = 1, .K = @intCast(n_out), .N = @intCast(n_in),
        }, .{ .x = @intCast(n_in), .y = 1 }, null);

        // Single GPU submission
        eng.commitAndWait();

        eng.downloadAccumTo(buf_gw, grad_w);
        for (grad_b, delta) |*gb, di| gb.* += di;
        @memset(grad_in, 0);
        eng.downloadTo(buf_gi, grad_in);
    }
};

fn activationKernelName(act: Activation) ?[:0]const u8 {
    return switch (act) {
        .relu => "relu_forward",
        .sigmoid => "sigmoid_forward",
        .tanh => "tanh_forward",
        .silu => "silu_forward",
        .gelu => "gelu_forward",
        .linear => null,
        .leaky_relu => @panic("leaky_relu not yet on Metal"),
        .elu => @panic("elu not yet on Metal"),
    };
}

// MetalMatmul — used by Tape autograd

pub const MetalMatmul = struct {
    pub fn forward(a: []const f32, b: []const f32, out: []f32, m: usize, k: usize, n: usize) void {
        const eng = getEngine() catch @panic("Metal init failed");
        eng.waitIfPending();

        const buf_a = eng.getOrUpload(a);
        const buf_b = eng.getOrUpload(b);
        const buf_out = eng.createBuffer(f32, m * n) catch @panic("OOM");
        defer eng.releaseBuffer(buf_out);

        eng.beginRecording();
        const pipe = eng.getPipeline("matmul_f32") catch @panic("pipeline");
        eng.encodeTyped(pipe, &.{ buf_a, buf_b, buf_out }, MatmulParams{
            .M = @intCast(m), .K = @intCast(k), .N = @intCast(n),
        }, .{ .x = @intCast(n), .y = @intCast(m) }, null);
        eng.commitAndWait();

        eng.downloadTo(buf_out, out);
    }

    pub fn backwardA(grad_out: []const f32, b: []const f32, grad_a: []f32, m: usize, k: usize, n: usize) void {
        const eng = getEngine() catch @panic("Metal init failed");
        eng.waitIfPending();

        const buf_gc = eng.getOrUpload(grad_out);
        const buf_b = eng.getOrUpload(b);
        const buf_ga = eng.getOrUploadMut(grad_a);

        eng.beginRecording();
        const pipe = eng.getPipeline("matmul_backward_a") catch @panic("pipeline");
        eng.encodeTyped(pipe, &.{ buf_gc, buf_b, buf_ga }, MatmulParams{
            .M = @intCast(m), .K = @intCast(k), .N = @intCast(n),
        }, .{ .x = @intCast(k), .y = @intCast(m) }, null);
        eng.commitAndWait();

        eng.downloadTo(buf_ga, grad_a);
    }

    pub fn backwardB(a: []const f32, grad_out: []const f32, grad_b: []f32, m: usize, k: usize, n: usize) void {
        const eng = getEngine() catch @panic("Metal init failed");
        eng.waitIfPending();

        const buf_a = eng.getOrUpload(a);
        const buf_gc = eng.getOrUpload(grad_out);
        const buf_gb = eng.getOrUploadMut(grad_b);

        eng.beginRecording();
        const pipe = eng.getPipeline("matmul_backward_b") catch @panic("pipeline");
        eng.encodeTyped(pipe, &.{ buf_a, buf_gc, buf_gb }, MatmulParams{
            .M = @intCast(m), .K = @intCast(k), .N = @intCast(n),
        }, .{ .x = @intCast(n), .y = @intCast(k) }, null);
        eng.commitAndWait();

        eng.downloadTo(buf_gb, grad_b);
    }
};

// MetalOptimizer

pub const MetalOptimizer = struct {
    pub fn update(
        opt: @import("../optimizer.zig").Optimizer,
        t: usize,
        lr: f32,
        params: []f32,
        grads: []f32,
        m: []f32,
        v: []f32,
    ) void {
        const eng = getEngine() catch @panic("Metal init failed");
        eng.waitIfPending();
        const len: u32 = @intCast(params.len);

        switch (opt.kind) {
            .sgd => {
                const buf_p = eng.getOrUploadMut(params);
                const buf_g = eng.getOrUpload(grads);

                eng.beginRecording();
                const pipe = eng.getPipeline("sgd_update") catch @panic("pipeline");
                eng.encodeTyped(pipe, &.{ buf_p, buf_g }, SGDParams{
                    .length = len, .lr = lr,
                }, .{ .x = len }, null);
                eng.commitAndWait();

                eng.downloadTo(buf_p, params);
            },
            .adam => |cfg| {
                dispatchAdam(eng, params, grads, m, v, len, lr, cfg.beta1, cfg.beta2, cfg.eps, 0, t);
            },
            .adamw => |cfg| {
                dispatchAdam(eng, params, grads, m, v, len, lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay, t);
            },
        }
        @memset(grads, 0);
    }

    fn dispatchAdam(
        eng: *MetalEngine,
        params: []f32, grads: []f32, m: []f32, v: []f32,
        len: u32, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, t: usize,
    ) void {
        const buf_p = eng.getOrUploadMut(params);
        const buf_g = eng.getOrUpload(grads);
        const buf_m = eng.getOrUploadMut(m);
        const buf_v = eng.getOrUploadMut(v);

        eng.beginRecording();
        const pipe = eng.getPipeline("adam_update") catch @panic("pipeline");
        eng.encodeTyped(pipe, &.{ buf_p, buf_g, buf_m, buf_v }, AdamParams{
            .length = len, .lr = lr,
            .beta1 = beta1, .beta2 = beta2, .eps = eps,
            .weight_decay = wd, .t = @intCast(t),
        }, .{ .x = len }, null);
        eng.commitAndWait();

        eng.downloadTo(buf_p, params);
        eng.downloadTo(buf_m, m);
        eng.downloadTo(buf_v, v);
    }
};

// MetalLoss — CPU fallback (reduction-heavy, needs sum)

pub const MetalLoss = struct {
    pub fn mse(output: []const f32, target: []const f32, grad_out: []f32, inv: f32) f32 {
        const cpu = @import("cpu.zig");
        return cpu.CpuLoss.mse(output, target, grad_out, inv);
    }
    pub fn cross_entropy(output: []const f32, target: []const f32, grad_out: []f32, inv: f32) f32 {
        const cpu = @import("cpu.zig");
        return cpu.CpuLoss.cross_entropy(output, target, grad_out, inv);
    }
};

// Norms — CPU fallback (reduction-heavy, small dim)

pub const MetalLayerNorm = struct {
    pub fn forward(input: []const f32, out: []f32, gamma: []const f32, beta: []const f32, eps: f32, x_norm_out: []f32, mean_out: *f32, rstd_out: *f32) void {
        @import("cpu.zig").CpuLayerNorm.forward(input, out, gamma, beta, eps, x_norm_out, mean_out, rstd_out);
    }
    pub fn backward(grad_out: []const f32, grad_in: []f32, grad_gamma: []f32, grad_beta: []f32, x_norm: []const f32, gamma: []const f32, rstd: f32) void {
        @import("cpu.zig").CpuLayerNorm.backward(grad_out, grad_in, grad_gamma, grad_beta, x_norm, gamma, rstd);
    }
};

pub const MetalRMSNorm = struct {
    pub fn forward(input: []const f32, out: []f32, gamma: []const f32, eps: f32, x_norm_out: []f32, rstd_out: *f32) void {
        @import("cpu.zig").CpuRMSNorm.forward(input, out, gamma, eps, x_norm_out, rstd_out);
    }
    pub fn backward(grad_out: []const f32, grad_in: []f32, grad_gamma: []f32, x_norm: []const f32, gamma: []const f32, rstd: f32) void {
        @import("cpu.zig").CpuRMSNorm.backward(grad_out, grad_in, grad_gamma, x_norm, gamma, rstd);
    }
};

// Remaining — CPU fallback

pub const MetalAttention = struct {
    pub fn forward(comptime d_model: usize, comptime n_heads: usize, comptime d_head: usize, comptime max_seq: usize, attn_scale: f32, seq: usize, input: []const f32, output: []f32, wq_f: []const f32, wk_f: []const f32, wv_f: []const f32, wo_f: []const f32, cache_q: []f32, cache_k: []f32, cache_v: []f32, cache_attn: []f32, cache_concat: []f32, scratch: []f32, causal: bool) void {
        @import("cpu.zig").CpuAttention.forward(d_model, n_heads, d_head, max_seq, attn_scale, seq, input, output, wq_f, wk_f, wv_f, wo_f, cache_q, cache_k, cache_v, cache_attn, cache_concat, scratch, causal);
    }
    pub fn backward(comptime d_model: usize, comptime n_heads: usize, comptime d_head: usize, attn_scale: f32, seq: usize, input: []const f32, grad_out: []const f32, grad_in: []f32, wq_f: []const f32, wk_f: []const f32, wv_f: []const f32, wo_f: []const f32, cache_q: []const f32, cache_k: []const f32, cache_v: []const f32, cache_attn: []const f32, cache_concat: []const f32, grad_wq: []f32, grad_wk: []f32, grad_wv: []f32, grad_wo: []f32, grad_q: []f32, grad_k: []f32, grad_v: []f32, grad_concat: []f32, scratch: []f32) void {
        @import("cpu.zig").CpuAttention.backward(d_model, n_heads, d_head, attn_scale, seq, input, grad_out, grad_in, wq_f, wk_f, wv_f, wo_f, cache_q, cache_k, cache_v, cache_attn, cache_concat, grad_wq, grad_wk, grad_wv, grad_wo, grad_q, grad_k, grad_v, grad_concat, scratch);
    }
};

pub const MetalPositionalEmbedding = struct {
    pub fn forward(input: []const f32, output: []f32, embed: []const f32) void {
        @import("cpu.zig").CpuPositionalEmbedding.forward(input, output, embed);
    }
    pub fn backward(grad_out: []const f32, grad_in: []f32, grad_embed: []f32) void {
        @import("cpu.zig").CpuPositionalEmbedding.backward(grad_out, grad_in, grad_embed);
    }
};

pub const MetalEmbedding = struct {
    pub fn forward(d_model: usize, indices: []const u32, output: []f32, table: []const f32) void {
        @import("cpu.zig").CpuEmbedding.forward(d_model, indices, output, table);
    }
    pub fn backward(d_model: usize, indices: []const u32, grad_out: []const f32, grad_table: []f32) void {
        @import("cpu.zig").CpuEmbedding.backward(d_model, indices, grad_out, grad_table);
    }
};

pub const MetalDropout = struct {
    pub fn forward(input: []const f32, output: []f32, mask: []bool, rate: f32, training: bool, rng: *std.Random.DefaultPrng) void {
        @import("cpu.zig").CpuDropout.forward(input, output, mask, rate, training, rng);
    }
    pub fn backward(grad_out: []const f32, grad_in: []f32, mask: []const bool, rate: f32) void {
        @import("cpu.zig").CpuDropout.backward(grad_out, grad_in, mask, rate);
    }
};
