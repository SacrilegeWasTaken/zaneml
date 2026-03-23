const std = @import("std");
const Activation = @import("../activation.zig").Activation;
const engine_mod = @import("../metal/engine.zig");
const MetalEngine = engine_mod.MetalEngine;
pub const GpuBuffer = engine_mod.GpuBuffer;
const Pipeline = engine_mod.Pipeline;
const Grid = MetalEngine.Grid;

// ── Lazy singleton

var g_engine: ?MetalEngine = null;

pub fn getEngine() !*MetalEngine {
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

// ── Tiled matmul helpers ─────────────────────────────────────────────────

const TILE: u32 = 16;
const TG_MEM: usize = 2 * TILE * TILE * @sizeOf(f32); // 2048 bytes

/// Round up to next multiple of TILE.
fn tileAlign(x: u32) u32 {
    return ((x + TILE - 1) / TILE) * TILE;
}

/// Encode a tiled matmul dispatch (with threadgroup memory).
fn encodeTiledMatmul(
    eng: *MetalEngine,
    pipe: Pipeline,
    bufs: []const GpuBuffer,
    params: MatmulParams,
    out_rows: u32,
    out_cols: u32,
) void {
    eng.encodeTypedTgMem(pipe, bufs, params, .{ .x = tileAlign(out_cols), .y = tileAlign(out_rows) }, .{ .x = TILE, .y = TILE }, TG_MEM);
}

const ActBwdParams = extern struct { length: u32, act_type: u32 };
const ReduceBiasParams = extern struct { seq: u32, n_out: u32 };

fn activationTypeId(act: Activation) u32 {
    return switch (act) {
        .linear => 0,
        .relu => 1,
        .sigmoid => 2,
        .tanh => 3,
        .silu => 4,
        .gelu => 5,
        .leaky_relu => @panic("leaky_relu not yet on Metal"),
        .elu => @panic("elu not yet on Metal"),
    };
}

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

        // Scratch buffers — pooled, never freed until engine deinit.
        // Slots: 0 = matmul output (z), 1 = bias-add output (za), 2 = activation output.
        const buf_z  = eng.getScratch(f32, n_out, 0);
        const buf_za = eng.getScratch(f32, n_out, 1);

        eng.beginRecording();

        // z = W @ x
        const matmul_pipe = eng.getPipeline("matmul_tiled_f32") catch @panic("pipeline");
        const mp = MatmulParams{ .M = @intCast(n_out), .K = @intCast(n_in), .N = 1 };
        encodeTiledMatmul(eng, matmul_pipe, &.{ buf_w, buf_x, buf_z }, mp, mp.M, mp.N);

        // za = z + bias
        const add_pipe = eng.getPipeline("add_f32") catch @panic("pipeline");
        eng.encodeTyped(add_pipe, &.{ buf_z, buf_b, buf_za }, ElementwiseParams{
            .length = @intCast(n_out),
        }, .{ .x = @intCast(n_out) }, null);

        // Activation (slot 2 for activation output)
        const act_kernel = activationKernelName(activation);
        var buf_act_out: ?GpuBuffer = null;
        if (act_kernel) |name| {
            buf_act_out = eng.getScratch(f32, n_out, 2);
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

        // Scratch buffers — pooled (slot 3 = grad_w, slot 4 = grad_in for backward).
        // Using slots 3/4 avoids collision with forward's slots 0/1/2.
        const buf_gw = eng.getScratch(f32, n_out * n_in, 3);
        @memset(buf_gw.asSlice(f32)[0 .. n_out * n_in], 0);

        const buf_gi = eng.getScratch(f32, n_in, 4);
        @memset(buf_gi.asSlice(f32)[0..n_in], 0);

        eng.beginRecording();

        const matmul_pipe = try eng.getPipeline("matmul_tiled_f32");

        // grad_w = delta[n_out×1] @ input[1×n_in] → [n_out×n_in]
        const mp_gw = MatmulParams{ .M = @intCast(n_out), .K = 1, .N = @intCast(n_in) };
        encodeTiledMatmul(eng, matmul_pipe, &.{ buf_delta, buf_input, buf_gw }, mp_gw, mp_gw.M, mp_gw.N);

        // grad_in = delta[1×n_out] @ W[n_out×n_in] → [1×n_in]
        const mp_gi = MatmulParams{ .M = 1, .K = @intCast(n_out), .N = @intCast(n_in) };
        encodeTiledMatmul(eng, matmul_pipe, &.{ buf_delta, buf_w, buf_gi }, mp_gi, mp_gi.M, mp_gi.N);

        // Single GPU submission
        eng.commitAndWait();

        eng.downloadAccumTo(buf_gw, grad_w);
        for (grad_b, delta) |*gb, di| gb.* += di;
        @memset(grad_in, 0);
        eng.downloadTo(buf_gi, grad_in);
    }
};

/// Params for the batch bias-add kernel (must match MSL layout).
const AddBiasBatchParams = extern struct { seq: u32, n_out: u32 };

// MetalBackend — batch variants for full-sequence FFN (one GPU call per layer).

pub const MetalBatchFFN = struct {
    /// Forward pass for a linear layer over an entire token sequence at once.
    ///
    /// input_batch  : [seq × n_in]
    /// pre_act_batch: [seq × n_out]  written (pre-activation values for backward)
    /// out_batch    : [seq × n_out]  written (post-activation output)
    ///
    /// Reduces seq separate command-buffer round-trips to a single one.
    pub fn forward(
        weights: []const f32,
        input_batch: []const f32,
        biases: []const f32,
        pre_act_batch: []f32,
        out_batch: []f32,
        activation: Activation,
        n_in: usize,
        n_out: usize,
        seq: usize,
    ) void {
        const eng = getEngine() catch @panic("Metal init failed");
        eng.waitIfPending();
        const total = seq * n_out;

        const buf_w  = eng.getOrUpload(weights);
        const buf_x  = eng.getOrUpload(input_batch);
        const buf_b  = eng.getOrUpload(biases);
        const buf_z  = eng.getScratch(f32, total, 0);
        const buf_za = eng.getScratch(f32, total, 1);

        eng.beginRecording();

        // z[seq×n_out] = input_batch[seq×n_in] × W^T[n_out×n_in]
        const bT_pipe = eng.getPipeline("matmul_bT_tiled_f32") catch @panic("pipeline");
        const mp_fwd = MatmulParams{ .M = @intCast(seq), .K = @intCast(n_in), .N = @intCast(n_out) };
        encodeTiledMatmul(eng, bT_pipe, &.{ buf_x, buf_w, buf_z }, mp_fwd, mp_fwd.M, mp_fwd.N);

        // za[seq×n_out] = z + broadcast(bias)
        const bias_pipe = eng.getPipeline("add_bias_batch_f32") catch @panic("pipeline");
        eng.encodeTyped(bias_pipe, &.{ buf_z, buf_b, buf_za }, AddBiasBatchParams{
            .seq = @intCast(seq), .n_out = @intCast(n_out),
        }, .{ .x = @intCast(n_out), .y = @intCast(seq) }, null);

        // Activation (slot 2)
        const act_kernel = activationKernelName(activation);
        var buf_act: ?GpuBuffer = null;
        if (act_kernel) |name| {
            buf_act = eng.getScratch(f32, total, 2);
            const act_pipe = eng.getPipeline(name) catch @panic("pipeline");
            eng.encodeTyped(act_pipe, &.{ buf_za, buf_act.? }, ElementwiseParams{
                .length = @intCast(total),
            }, .{ .x = @intCast(total) }, null);
        }

        eng.commitAndWait();

        eng.downloadTo(buf_za, pre_act_batch);
        if (buf_act) |ba| {
            eng.downloadTo(ba, out_batch);
        } else {
            @memcpy(out_batch, pre_act_batch);
        }
    }

    /// Backward pass for a linear layer over an entire token sequence.
    ///
    /// Accumulates grad_w and grad_b; writes grad_in_batch.
    pub fn backward(
        allocator: std.mem.Allocator,
        weights: []const f32,
        input_batch: []const f32,
        pre_act_batch: []const f32,
        grad_out_batch: []const f32,
        grad_in_batch: []f32,
        grad_w: []f32,
        grad_b: []f32,
        activation: Activation,
        n_in: usize,
        n_out: usize,
        seq: usize,
    ) !void {
        _ = allocator;
        const eng = try getEngine();
        eng.waitIfPending();
        const total = seq * n_out;

        const buf_pre   = eng.getOrUpload(pre_act_batch);
        const buf_go    = eng.getOrUpload(grad_out_batch);
        const buf_input = eng.getOrUpload(input_batch);
        const buf_w     = eng.getOrUpload(weights);
        const buf_gb    = eng.getOrUploadMut(grad_b);
        const buf_delta = eng.getScratch(f32, total, 50);    // GPU-side delta
        const buf_gw    = eng.getScratch(f32, n_out * n_in, 3);
        const buf_gi    = eng.getScratch(f32, seq * n_in, 4);

        @memset(buf_gw.asSlice(f32)[0 .. n_out * n_in], 0);
        @memset(buf_gi.asSlice(f32)[0 .. seq * n_in], 0);

        eng.beginRecording();

        // delta[seq×n_out] = grad_out * act'(pre_act)  — GPU
        const act_bwd_pipe = try eng.getPipeline("fused_act_backward");
        eng.encodeTyped(act_bwd_pipe, &.{ buf_pre, buf_go, buf_delta }, ActBwdParams{
            .length = @intCast(total), .act_type = activationTypeId(activation),
        }, .{ .x = @intCast(total) }, null);

        const bw_b_pipe = try eng.getPipeline("matmul_backward_b_tiled");
        const matmul_pipe = try eng.getPipeline("matmul_tiled_f32");

        // grad_w[n_out×n_in] += delta_batch^T @ input_batch
        const mp_gw = MatmulParams{ .M = @intCast(seq), .K = @intCast(n_out), .N = @intCast(n_in) };
        encodeTiledMatmul(eng, bw_b_pipe, &.{ buf_delta, buf_input, buf_gw }, mp_gw, mp_gw.K, mp_gw.N);

        // grad_in_batch[seq×n_in] = delta_batch[seq×n_out] @ W[n_out×n_in]
        const mp_gi = MatmulParams{ .M = @intCast(seq), .K = @intCast(n_out), .N = @intCast(n_in) };
        encodeTiledMatmul(eng, matmul_pipe, &.{ buf_delta, buf_w, buf_gi }, mp_gi, mp_gi.M, mp_gi.N);

        // grad_b[n_out] += sum over seq of delta — GPU
        const rb_pipe = try eng.getPipeline("reduce_bias_grad");
        eng.encodeTyped(rb_pipe, &.{ buf_delta, buf_gb }, ReduceBiasParams{
            .seq = @intCast(seq), .n_out = @intCast(n_out),
        }, .{ .x = @intCast(n_out) }, null);

        eng.commitAndWait();

        eng.downloadAccumTo(buf_gw, grad_w);
        eng.downloadTo(buf_gb, grad_b);
        @memset(grad_in_batch, 0);
        eng.downloadTo(buf_gi, grad_in_batch);
    }
};

// ── Params structs for new kernels ───────────────────────────────────────

const AttnScoreParams  = extern struct { seq: u32, n_heads: u32, d_head: u32, scale: f32, causal: u32 };
const AttnCtxParams    = extern struct { seq: u32, n_heads: u32, d_head: u32 };
const AttnGradQKParams = extern struct { seq: u32, n_heads: u32, d_head: u32, scale: f32 };
const SoftmaxParams    = extern struct { n_rows: u32, row_len: u32 };
const SoftmaxBwdParams = extern struct { n_rows: u32, row_len: u32 };
const PeLenParams      = extern struct { length: u32 };

// ── MetalBatchNorm ───────────────────────────────────────────────────────

/// Batch LayerNorm/RMSNorm for all sequence positions at once.
/// Requires d_model to be a power of 2 and <= 1024.
pub const MetalBatchNorm = struct {

    pub fn layernorm_fwd(
        input: []const f32,    // [seq × d]
        gamma: []const f32,    // [d]
        beta:  []const f32,    // [d]
        eps:   f32,
        out:        []f32,     // [seq × d]
        x_norm_out: []f32,     // [seq × d]
        rstd_out:   []f32,     // [seq]
        seq: usize,
        d:   usize,
    ) void {
        const eng = getEngine() catch @panic("Metal init failed");
        const buf_x   = eng.getOrUpload(input);
        const buf_g   = eng.getOrUpload(gamma);
        const buf_b   = eng.getOrUpload(beta);
        const buf_out = eng.getScratch(f32, seq * d, 5);
        const buf_xn  = eng.getScratch(f32, seq * d, 6);
        const buf_rs  = eng.getScratch(f32, seq,     7);

        eng.beginRecording();
        const pipe = eng.getPipeline("layernorm_fwd_seq") catch @panic("pipeline layernorm_fwd_seq");
        // threadgroup size = d, grid = (d, seq), threadgroup memory = d floats
        const tg = MetalEngine.Grid{ .x = @intCast(d), .y = 1 };
        eng.encodeTgMem(pipe, &.{ buf_x, buf_g, buf_b, buf_out, buf_xn, buf_rs },
            @ptrCast(&LNFwdParams{ .d = @intCast(d), .eps = eps }),
            @sizeOf(LNFwdParams),
            .{ .x = @intCast(d), .y = @intCast(seq) }, tg, d * @sizeOf(f32));
        eng.commitAndWait();

        eng.downloadTo(buf_out, out);
        eng.downloadTo(buf_xn,  x_norm_out);
        eng.downloadTo(buf_rs,  rstd_out);
    }

    pub fn layernorm_bwd(
        grad_out:   []const f32,  // [seq × d]
        x_norm:     []const f32,  // [seq × d]
        gamma:      []const f32,  // [d]
        rstd:       []const f32,  // [seq]
        grad_gamma: []f32,        // [d]  (accumulated)
        grad_beta:  []f32,        // [d]  (accumulated)
        grad_in:    []f32,        // [seq × d]
        seq: usize,
        d:   usize,
    ) void {
        const eng = getEngine() catch @panic("Metal init failed");
        const buf_go  = eng.getOrUpload(grad_out);
        const buf_xn  = eng.getOrUpload(x_norm);
        const buf_g   = eng.getOrUpload(gamma);
        const buf_rs  = eng.getOrUpload(rstd);
        // grad_gamma / grad_beta are accumulated — upload current CPU values so GPU starts from them
        const buf_gg  = eng.getOrUploadMut(grad_gamma);
        const buf_gb  = eng.getOrUploadMut(grad_beta);
        const buf_gi  = eng.getScratch(f32, seq * d, 8);

        eng.beginRecording();
        const pipe = eng.getPipeline("layernorm_bwd_seq") catch @panic("pipeline layernorm_bwd_seq");
        const tg = MetalEngine.Grid{ .x = @intCast(d), .y = 1 };
        eng.encodeTgMem(pipe, &.{ buf_go, buf_xn, buf_g, buf_rs, buf_gg, buf_gb, buf_gi },
            @ptrCast(&LNBwdParams{ .d = @intCast(d) }),
            @sizeOf(LNBwdParams),
            .{ .x = @intCast(d), .y = @intCast(seq) }, tg, d * @sizeOf(f32));
        eng.commitAndWait();

        eng.downloadTo(buf_gg, grad_gamma);
        eng.downloadTo(buf_gb, grad_beta);
        eng.downloadTo(buf_gi, grad_in);
    }

    pub fn rmsnorm_fwd(
        input: []const f32,
        gamma: []const f32,
        eps:   f32,
        out:        []f32,
        x_norm_out: []f32,
        rms_out:    []f32,
        seq: usize,
        d:   usize,
    ) void {
        const eng = getEngine() catch @panic("Metal init failed");
        const buf_x   = eng.getOrUpload(input);
        const buf_g   = eng.getOrUpload(gamma);
        const buf_out = eng.getScratch(f32, seq * d, 5);
        const buf_xn  = eng.getScratch(f32, seq * d, 6);
        const buf_rms = eng.getScratch(f32, seq,     7);

        eng.beginRecording();
        const pipe = eng.getPipeline("rmsnorm_fwd_seq") catch @panic("pipeline rmsnorm_fwd_seq");
        const tg = MetalEngine.Grid{ .x = @intCast(d), .y = 1 };
        eng.encodeTgMem(pipe, &.{ buf_x, buf_g, buf_out, buf_xn, buf_rms },
            @ptrCast(&RMSFwdParams{ .d = @intCast(d), .eps = eps }),
            @sizeOf(RMSFwdParams),
            .{ .x = @intCast(d), .y = @intCast(seq) }, tg, d * @sizeOf(f32));
        eng.commitAndWait();

        eng.downloadTo(buf_out, out);
        eng.downloadTo(buf_xn,  x_norm_out);
        eng.downloadTo(buf_rms, rms_out);
    }

    pub fn rmsnorm_bwd(
        grad_out:   []const f32,
        x_norm:     []const f32,
        gamma:      []const f32,
        rms:        []const f32,
        grad_gamma: []f32,
        grad_in:    []f32,
        seq: usize,
        d:   usize,
    ) void {
        const eng = getEngine() catch @panic("Metal init failed");
        const buf_go  = eng.getOrUpload(grad_out);
        const buf_xn  = eng.getOrUpload(x_norm);
        const buf_g   = eng.getOrUpload(gamma);
        const buf_rms = eng.getOrUpload(rms);
        const buf_gg  = eng.getOrUploadMut(grad_gamma);
        const buf_gi  = eng.getScratch(f32, seq * d, 8);

        eng.beginRecording();
        const pipe = eng.getPipeline("rmsnorm_bwd_seq") catch @panic("pipeline rmsnorm_bwd_seq");
        const tg = MetalEngine.Grid{ .x = @intCast(d), .y = 1 };
        eng.encodeTgMem(pipe, &.{ buf_go, buf_xn, buf_g, buf_rms, buf_gg, buf_gi },
            @ptrCast(&RMSBwdParams{ .d = @intCast(d) }),
            @sizeOf(RMSBwdParams),
            .{ .x = @intCast(d), .y = @intCast(seq) }, tg, d * @sizeOf(f32));
        eng.commitAndWait();

        eng.downloadTo(buf_gg, grad_gamma);
        eng.downloadTo(buf_gi, grad_in);
    }
};

// ── MetalBatchPE ─────────────────────────────────────────────────────────

pub const MetalBatchPE = struct {
    pub fn forward(input: []const f32, embed: []const f32, output: []f32) void {
        const eng = getEngine() catch @panic("Metal init failed");
        const len: u32 = @intCast(input.len);
        const buf_in  = eng.getOrUpload(input);
        const buf_em  = eng.getOrUpload(embed);
        const buf_out = eng.getScratch(f32, input.len, 9);

        eng.beginRecording();
        const pipe = eng.getPipeline("pe_fwd") catch @panic("pipeline pe_fwd");
        eng.encodeTyped(pipe, &.{ buf_in, buf_em, buf_out },
            PeLenParams{ .length = len }, .{ .x = len }, null);
        eng.commitAndWait();
        eng.downloadTo(buf_out, output);
    }

    pub fn backward_embed(grad_out: []const f32, grad_embed: []f32) void {
        const eng = getEngine() catch @panic("Metal init failed");
        const len: u32 = @intCast(grad_out.len);
        const buf_go  = eng.getOrUpload(grad_out);
        const buf_ge  = eng.getOrUploadMut(grad_embed);

        eng.beginRecording();
        const pipe = eng.getPipeline("pe_bwd_embed") catch @panic("pipeline pe_bwd_embed");
        eng.encodeTyped(pipe, &.{ buf_go, buf_ge },
            PeLenParams{ .length = len }, .{ .x = len }, null);
        eng.commitAndWait();
        eng.downloadTo(buf_ge, grad_embed);
    }
};

// ── MetalBatchAttn ───────────────────────────────────────────────────────

pub const MetalBatchAttn = struct {

    pub fn forward(
        comptime d_model: usize,
        comptime n_heads: usize,
        comptime d_head:  usize,
        attn_scale: f32,
        seq: usize,
        input:        []const f32,   // [seq × d_model]
        output:       []f32,         // [seq × d_model]
        wq_f: []const f32,
        wk_f: []const f32,
        wv_f: []const f32,
        wo_f: []const f32,
        cache_q:      []f32,         // [seq × d_model]
        cache_k:      []f32,         // [seq × d_model]
        cache_v:      []f32,         // [seq × d_model]
        cache_attn:   []f32,         // [n_heads × seq × seq]
        cache_concat: []f32,         // [seq × d_model]
        causal: bool,
    ) void {
        const eng = getEngine() catch @panic("Metal init failed");
        eng.waitIfPending();
        const smd  = seq * d_model;
        const shh  = n_heads * seq * seq;

        const buf_x   = eng.getOrUpload(input);
        const buf_wq  = eng.getOrUpload(wq_f);
        const buf_wk  = eng.getOrUpload(wk_f);
        const buf_wv  = eng.getOrUpload(wv_f);
        const buf_wo  = eng.getOrUpload(wo_f);

        // scratch slots 10-15 reserved for attention intermediates
        const buf_q   = eng.getScratch(f32, smd, 10);
        const buf_k   = eng.getScratch(f32, smd, 11);
        const buf_v   = eng.getScratch(f32, smd, 12);
        const buf_sc  = eng.getScratch(f32, shh, 13);  // attention scores
        const buf_ctx = eng.getScratch(f32, smd, 14);
        const buf_out = eng.getScratch(f32, smd, 15);

        eng.beginRecording();

        // Q = input × Wq^T,  K = input × Wk^T,  V = input × Wv^T
        const bT  = eng.getPipeline("matmul_bT_tiled_f32") catch @panic("pipeline matmul_bT_tiled_f32");
        const mp  = MatmulParams{ .M = @intCast(seq), .K = @intCast(d_model), .N = @intCast(d_model) };
        encodeTiledMatmul(eng, bT, &.{ buf_x, buf_wq, buf_q }, mp, mp.M, mp.N);
        encodeTiledMatmul(eng, bT, &.{ buf_x, buf_wk, buf_k }, mp, mp.M, mp.N);
        encodeTiledMatmul(eng, bT, &.{ buf_x, buf_wv, buf_v }, mp, mp.M, mp.N);

        // attention scores [n_heads × seq × seq]
        const sp = AttnScoreParams{
            .seq = @intCast(seq), .n_heads = @intCast(n_heads),
            .d_head = @intCast(d_head), .scale = attn_scale,
            .causal = if (causal) 1 else 0,
        };
        const sc_pipe = eng.getPipeline("attn_scores_fwd") catch @panic("pipeline attn_scores_fwd");
        eng.encodeTyped(sc_pipe, &.{ buf_q, buf_k, buf_sc }, sp,
            .{ .x = @intCast(seq), .y = @intCast(seq), .z = @intCast(n_heads) }, null);

        // softmax in-place on buf_sc (needs threadgroup memory = seq floats)
        const sm_pipe = eng.getPipeline("softmax_rows_inplace") catch @panic("pipeline softmax_rows_inplace");
        const n_rows: u32 = @intCast(n_heads * seq);
        const sm_tg = MetalEngine.Grid{ .x = @intCast(seq), .y = 1 };
        eng.encodeTgMem(sm_pipe, &.{ buf_sc },
            @ptrCast(&SoftmaxParams{ .n_rows = n_rows, .row_len = @intCast(seq) }),
            @sizeOf(SoftmaxParams),
            .{ .x = @intCast(seq), .y = n_rows }, sm_tg, seq * @sizeOf(f32));

        // context = attn × V
        const ctx_pipe = eng.getPipeline("attn_context_fwd") catch @panic("pipeline attn_context_fwd");
        const cp = AttnCtxParams{ .seq = @intCast(seq), .n_heads = @intCast(n_heads), .d_head = @intCast(d_head) };
        eng.encodeTyped(ctx_pipe, &.{ buf_sc, buf_v, buf_ctx }, cp,
            .{ .x = @intCast(d_head), .y = @intCast(seq), .z = @intCast(n_heads) }, null);

        // output = context × Wo^T
        encodeTiledMatmul(eng, bT, &.{ buf_ctx, buf_wo, buf_out }, mp, mp.M, mp.N);

        eng.commitAndWait();

        // save caches for backward
        eng.downloadTo(buf_q,   cache_q  [0..smd]);
        eng.downloadTo(buf_k,   cache_k  [0..smd]);
        eng.downloadTo(buf_v,   cache_v  [0..smd]);
        eng.downloadTo(buf_sc,  cache_attn[0..shh]);
        eng.downloadTo(buf_ctx, cache_concat[0..smd]);
        eng.downloadTo(buf_out, output[0..smd]);
    }

    pub fn backward(
        comptime d_model: usize,
        comptime n_heads: usize,
        comptime d_head:  usize,
        attn_scale: f32,
        seq: usize,
        input:        []const f32,
        grad_out:     []const f32,
        grad_in:      []f32,
        wq_f: []const f32,
        wk_f: []const f32,
        wv_f: []const f32,
        wo_f: []const f32,
        cache_q:      []const f32,
        cache_k:      []const f32,
        cache_v:      []const f32,
        cache_attn:   []const f32,
        cache_concat: []const f32,
        grad_wq: []f32,
        grad_wk: []f32,
        grad_wv: []f32,
        grad_wo: []f32,
        grad_q:  []f32,
        grad_k:  []f32,
        grad_v:  []f32,
        grad_concat: []f32,
        _scratch: []f32,
    ) void {
        _ = _scratch;
        const eng = getEngine() catch @panic("Metal init failed");
        eng.waitIfPending();
        const smd = seq * d_model;
        const shh = n_heads * seq * seq;

        const buf_x      = eng.getOrUpload(input);
        const buf_go     = eng.getOrUpload(grad_out);
        const buf_wq_b   = eng.getOrUpload(wq_f);
        const buf_wk_b   = eng.getOrUpload(wk_f);
        const buf_wv_b   = eng.getOrUpload(wv_f);
        const buf_wo_b   = eng.getOrUpload(wo_f);
        const buf_cq     = eng.getOrUpload(cache_q);
        const buf_ck     = eng.getOrUpload(cache_k);
        const buf_cv     = eng.getOrUpload(cache_v);
        const buf_ca     = eng.getOrUpload(cache_attn);
        const buf_cc     = eng.getOrUpload(cache_concat);

        // weight gradient accumulators (upload current gradient values)
        const buf_gwq    = eng.getOrUploadMut(grad_wq);
        const buf_gwk    = eng.getOrUploadMut(grad_wk);
        const buf_gwv    = eng.getOrUploadMut(grad_wv);
        const buf_gwo    = eng.getOrUploadMut(grad_wo);

        // intermediate grad_q, grad_k, grad_v (start from zero)
        const buf_gq     = eng.getOrUploadMut(grad_q);
        const buf_gk     = eng.getOrUploadMut(grad_k);
        const buf_gv     = eng.getOrUploadMut(grad_v);

        // scratch: grad_concat, grad_in from Q/K/V projections (separate buffers to avoid overwrite)
        const buf_gc     = eng.getScratch(f32, smd, 20);   // grad_concat
        const buf_gp     = eng.getScratch(f32, shh, 21);   // grad_attn_pre

        // Three separate buffers for grad_in contributions (matmul_bT overwrites, not accumulates)
        const buf_gi_q   = eng.getScratch(f32, smd, 22);   // grad_in from Q projection
        const buf_gi_k   = eng.getScratch(f32, smd, 23);   // grad_in from K projection
        const buf_gi_v   = eng.getScratch(f32, smd, 24);   // grad_in from V projection

        // Zero grad_q, grad_k, grad_v before atomic accumulation
        @memset(buf_gq.asSlice(f32)[0..smd], 0);
        @memset(buf_gk.asSlice(f32)[0..smd], 0);
        @memset(buf_gv.asSlice(f32)[0..smd], 0);

        eng.beginRecording();

        const bT      = eng.getPipeline("matmul_bT_tiled_f32")      catch @panic("pipeline matmul_bT_tiled_f32");
        const bwB     = eng.getPipeline("matmul_backward_b_tiled") catch @panic("pipeline matmul_backward_b_tiled");
        const gv_pipe = eng.getPipeline("attn_grad_v")        catch @panic("pipeline attn_grad_v");
        const gp_pipe = eng.getPipeline("attn_grad_attn_pre") catch @panic("pipeline attn_grad_attn_pre");
        const sb_pipe = eng.getPipeline("softmax_bwd_rows")   catch @panic("pipeline softmax_bwd_rows");
        const gq_pipe = eng.getPipeline("attn_grad_q")        catch @panic("pipeline attn_grad_q");
        const gk_pipe = eng.getPipeline("attn_grad_k")        catch @panic("pipeline attn_grad_k");
        const add3_pipe = eng.getPipeline("add3_f32")          catch @panic("pipeline add3_f32");

        const dm32: u32 = @intCast(d_model);
        const sq32: u32 = @intCast(seq);
        const nh32: u32 = @intCast(n_heads);
        const dh32: u32 = @intCast(d_head);

        const mp = MatmulParams{ .M = sq32, .K = dm32, .N = dm32 };

        // grad_wo += concat^T @ grad_out  (matmul_backward_b_tiled accumulates +=)
        encodeTiledMatmul(eng, bwB, &.{ buf_cc, buf_go, buf_gwo }, mp, mp.K, mp.N);

        // grad_concat = grad_out × Wo^T  (overwrites buf_gc)
        encodeTiledMatmul(eng, bT, &.{ buf_go, buf_wo_b, buf_gc }, mp, mp.M, mp.N);

        const ctx_p = AttnCtxParams{ .seq = sq32, .n_heads = nh32, .d_head = dh32 };

        // grad_v (atomic accumulate from all heads)
        eng.encodeTyped(gv_pipe, &.{ buf_ca, buf_gc, buf_gv }, ctx_p,
            .{ .x = dh32, .y = sq32, .z = nh32 }, null);

        // grad_attn_pre  (overwrites buf_gp)
        eng.encodeTyped(gp_pipe, &.{ buf_gc, buf_cv, buf_gp }, ctx_p,
            .{ .x = sq32, .y = sq32, .z = nh32 }, null);

        // softmax backward in-place on buf_gp (needs threadgroup memory = seq floats)
        const n_rows: u32 = nh32 * sq32;
        const sm_tg = MetalEngine.Grid{ .x = sq32, .y = 1 };
        eng.encodeTgMem(sb_pipe, &.{ buf_ca, buf_gp },
            @ptrCast(&SoftmaxBwdParams{ .n_rows = n_rows, .row_len = sq32 }),
            @sizeOf(SoftmaxBwdParams),
            .{ .x = sq32, .y = n_rows }, sm_tg, seq * @sizeOf(f32));

        // grad_q (atomic accumulate)
        const qk_p = AttnGradQKParams{ .seq = sq32, .n_heads = nh32, .d_head = dh32, .scale = attn_scale };
        eng.encodeTyped(gq_pipe, &.{ buf_gp, buf_ck, buf_gq }, qk_p,
            .{ .x = dh32, .y = sq32, .z = nh32 }, null);

        // grad_k (atomic accumulate)
        eng.encodeTyped(gk_pipe, &.{ buf_gp, buf_cq, buf_gk }, qk_p,
            .{ .x = dh32, .y = sq32, .z = nh32 }, null);

        // grad_wq += input^T @ grad_q  (accumulates +=)
        encodeTiledMatmul(eng, bwB, &.{ buf_x, buf_gq, buf_gwq }, mp, mp.K, mp.N);
        // grad_in contribution from Q projection (overwrites buf_gi_q)
        encodeTiledMatmul(eng, bT, &.{ buf_gq, buf_wq_b, buf_gi_q }, mp, mp.M, mp.N);

        // grad_wk += input^T @ grad_k  (accumulates +=)
        encodeTiledMatmul(eng, bwB, &.{ buf_x, buf_gk, buf_gwk }, mp, mp.K, mp.N);
        // grad_in contribution from K projection (overwrites buf_gi_k)
        encodeTiledMatmul(eng, bT, &.{ buf_gk, buf_wk_b, buf_gi_k }, mp, mp.M, mp.N);

        // grad_wv += input^T @ grad_v  (accumulates +=)
        encodeTiledMatmul(eng, bwB, &.{ buf_x, buf_gv, buf_gwv }, mp, mp.K, mp.N);
        // grad_in contribution from V projection (overwrites buf_gi_v)
        encodeTiledMatmul(eng, bT, &.{ buf_gv, buf_wv_b, buf_gi_v }, mp, mp.M, mp.N);

        // Sum three grad_in contributions on GPU (instead of CPU)
        const buf_gi = eng.getScratch(f32, smd, 25);
        eng.encodeTyped(add3_pipe, &.{ buf_gi_q, buf_gi_k, buf_gi_v, buf_gi }, ElementwiseParams{
            .length = @intCast(smd),
        }, .{ .x = @intCast(smd) }, null);

        eng.commitAndWait();

        eng.downloadTo(buf_gwo, grad_wo);
        eng.downloadTo(buf_gwq, grad_wq);
        eng.downloadTo(buf_gwk, grad_wk);
        eng.downloadTo(buf_gwv, grad_wv);
        eng.downloadTo(buf_gq,  grad_q);
        eng.downloadTo(buf_gk,  grad_k);
        eng.downloadTo(buf_gv,  grad_v);
        eng.downloadTo(buf_gc,  grad_concat[0..smd]);
        eng.downloadTo(buf_gi,  grad_in[0..smd]);
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

        const buf_a   = eng.getOrUpload(a);
        const buf_b   = eng.getOrUpload(b);
        const buf_out = eng.getScratch(f32, m * n, 0);

        eng.beginRecording();
        const pipe = eng.getPipeline("matmul_tiled_f32") catch @panic("pipeline");
        const mp = MatmulParams{ .M = @intCast(m), .K = @intCast(k), .N = @intCast(n) };
        encodeTiledMatmul(eng, pipe, &.{ buf_a, buf_b, buf_out }, mp, mp.M, mp.N);
        eng.commitAndWait();

        eng.downloadTo(buf_out, out);
    }

    pub fn backwardA(grad_out: []const f32, b: []const f32, grad_a: []f32, m: usize, k: usize, n: usize) void {
        const eng = getEngine() catch @panic("Metal init failed");
        eng.waitIfPending();

        const buf_gc = eng.getOrUpload(grad_out);
        const buf_b  = eng.getOrUpload(b);
        const buf_ga = eng.getOrUploadMut(grad_a);

        eng.beginRecording();
        const pipe = eng.getPipeline("matmul_backward_a_tiled") catch @panic("pipeline");
        const mp = MatmulParams{ .M = @intCast(m), .K = @intCast(k), .N = @intCast(n) };
        encodeTiledMatmul(eng, pipe, &.{ buf_gc, buf_b, buf_ga }, mp, mp.M, mp.K);
        eng.commitAndWait();

        eng.downloadTo(buf_ga, grad_a);
    }

    pub fn backwardB(a: []const f32, grad_out: []const f32, grad_b: []f32, m: usize, k: usize, n: usize) void {
        const eng = getEngine() catch @panic("Metal init failed");
        eng.waitIfPending();

        const buf_a  = eng.getOrUpload(a);
        const buf_gc = eng.getOrUpload(grad_out);
        const buf_gb = eng.getOrUploadMut(grad_b);

        eng.beginRecording();
        const pipe = eng.getPipeline("matmul_backward_b_tiled") catch @panic("pipeline");
        const mp = MatmulParams{ .M = @intCast(m), .K = @intCast(k), .N = @intCast(n) };
        encodeTiledMatmul(eng, pipe, &.{ buf_a, buf_gc, buf_gb }, mp, mp.K, mp.N);
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

const LossFusedParams = extern struct { length: u32, inv: f32 };

pub const MetalLoss = struct {
    pub fn mse(output: []const f32, target: []const f32, grad_out: []f32, inv: f32) f32 {
        return lossDispatch("mse_loss", output, target, grad_out, inv);
    }
    pub fn cross_entropy(output: []const f32, target: []const f32, grad_out: []f32, inv: f32) f32 {
        return lossDispatch("cross_entropy_loss", output, target, grad_out, inv);
    }
};

fn lossDispatch(comptime kernel: [:0]const u8, output: []const f32, target: []const f32, grad_out: []f32, inv: f32) f32 {
    const eng = getEngine() catch @panic("Metal init failed");
    const len = output.len;

    // Threadgroup size = next power-of-2 >= len (max 1024)
    var tg: u32 = 1;
    while (tg < len) tg <<= 1;

    const buf_out  = eng.getOrUpload(output);
    const buf_tgt  = eng.getOrUpload(target);
    const buf_grad = eng.getScratch(f32, len, 40);
    const buf_loss = eng.getScratch(f32, 1,   41);
    // Zero the loss accumulator before dispatch
    buf_loss.asSlice(f32)[0] = 0;

    eng.beginRecording();
    const pipe = eng.getPipeline(kernel) catch @panic("pipeline loss");
    eng.encodeTgMem(pipe, &.{ buf_out, buf_tgt, buf_grad, buf_loss },
        @ptrCast(&LossFusedParams{ .length = @intCast(len), .inv = inv }),
        @sizeOf(LossFusedParams),
        .{ .x = tg }, .{ .x = tg }, tg * @sizeOf(f32));
    eng.commitAndWait();

    eng.downloadTo(buf_grad, grad_out);
    return buf_loss.asSlice(f32)[0];
}

// Norms — single-position wrappers delegating to batch GPU kernels (seq=1)

pub const MetalLayerNorm = struct {
    pub fn forward(input: []const f32, out: []f32, gamma: []const f32, beta: []const f32, eps: f32, x_norm_out: []f32, mean_out: *f32, rstd_out: *f32) void {
        _ = mean_out;
        var rstd_buf = [1]f32{0};
        MetalBatchNorm.layernorm_fwd(input, gamma, beta, eps, out, x_norm_out, &rstd_buf, 1, input.len);
        rstd_out.* = rstd_buf[0];
    }
    pub fn backward(grad_out: []const f32, grad_in: []f32, grad_gamma: []f32, grad_beta: []f32, x_norm: []const f32, gamma: []const f32, rstd: f32) void {
        var rstd_buf = [1]f32{rstd};
        MetalBatchNorm.layernorm_bwd(grad_out, x_norm, gamma, &rstd_buf, grad_gamma, grad_beta, grad_in, 1, grad_out.len);
    }
};

pub const MetalRMSNorm = struct {
    pub fn forward(input: []const f32, out: []f32, gamma: []const f32, eps: f32, x_norm_out: []f32, rstd_out: *f32) void {
        var rms_buf = [1]f32{0};
        MetalBatchNorm.rmsnorm_fwd(input, gamma, eps, out, x_norm_out, &rms_buf, 1, input.len);
        rstd_out.* = rms_buf[0];
    }
    pub fn backward(grad_out: []const f32, grad_in: []f32, grad_gamma: []f32, x_norm: []const f32, gamma: []const f32, rstd: f32) void {
        var rms_buf = [1]f32{rstd};
        MetalBatchNorm.rmsnorm_bwd(grad_out, x_norm, gamma, &rms_buf, grad_gamma, grad_in, 1, grad_out.len);
    }
};

pub const MetalAttention = struct {
    pub fn forward(comptime d_model: usize, comptime n_heads: usize, comptime d_head: usize, comptime max_seq: usize, attn_scale: f32, seq: usize, input: []const f32, output: []f32, wq_f: []const f32, wk_f: []const f32, wv_f: []const f32, wo_f: []const f32, cache_q: []f32, cache_k: []f32, cache_v: []f32, cache_attn: []f32, cache_concat: []f32, _scratch: []f32, causal: bool) void {
        _ = max_seq; _ = _scratch;
        MetalBatchAttn.forward(d_model, n_heads, d_head, attn_scale, seq, input, output, wq_f, wk_f, wv_f, wo_f, cache_q, cache_k, cache_v, cache_attn, cache_concat, causal);
    }
    pub fn backward(comptime d_model: usize, comptime n_heads: usize, comptime d_head: usize, attn_scale: f32, seq: usize, input: []const f32, grad_out: []const f32, grad_in: []f32, wq_f: []const f32, wk_f: []const f32, wv_f: []const f32, wo_f: []const f32, cache_q: []const f32, cache_k: []const f32, cache_v: []const f32, cache_attn: []const f32, cache_concat: []const f32, grad_wq: []f32, grad_wk: []f32, grad_wv: []f32, grad_wo: []f32, grad_q: []f32, grad_k: []f32, grad_v: []f32, grad_concat: []f32, scratch: []f32) void {
        MetalBatchAttn.backward(d_model, n_heads, d_head, attn_scale, seq, input, grad_out, grad_in, wq_f, wk_f, wv_f, wo_f, cache_q, cache_k, cache_v, cache_attn, cache_concat, grad_wq, grad_wk, grad_wv, grad_wo, grad_q, grad_k, grad_v, grad_concat, scratch);
    }
};

pub const MetalPositionalEmbedding = struct {
    pub fn forward(input: []const f32, output: []f32, embed: []const f32) void {
        MetalBatchPE.forward(input, embed, output);
    }
    pub fn backward(grad_out: []const f32, grad_in: []f32, grad_embed: []f32) void {
        // grad_in = grad_out (identity pass-through)
        @memcpy(grad_in, grad_out);
        // grad_embed += grad_out  (GPU atomic accumulate)
        MetalBatchPE.backward_embed(grad_out, grad_embed);
    }
};

const EmbedParams   = extern struct { d_model: u32 };
const DropoutFwdParams = extern struct { length: u32, scale: f32, training: u32 };
const DropoutBwdParams = extern struct { length: u32, scale: f32 };

pub const MetalEmbedding = struct {
    pub fn forward(d_model: usize, indices: []const u32, output: []f32, table: []const f32) void {
        const eng  = getEngine() catch @panic("Metal init failed");
        const ntok = indices.len;

        // Scratch slot 30: indices (u32), slot 31: output (f32)
        const buf_idx = eng.getScratch(u32, ntok, 30);
        @memcpy(buf_idx.asSlice(u32)[0..ntok], indices);
        const buf_tab = eng.getOrUpload(table);
        const buf_out = eng.getScratch(f32, ntok * d_model, 31);

        eng.beginRecording();
        const pipe = eng.getPipeline("embedding_fwd") catch @panic("pipeline embedding_fwd");
        eng.encodeTyped(pipe, &.{ buf_idx, buf_tab, buf_out },
            EmbedParams{ .d_model = @intCast(d_model) },
            .{ .x = @intCast(d_model), .y = @intCast(ntok) }, null);
        eng.commitAndWait();
        eng.downloadTo(buf_out, output);
    }

    pub fn backward(d_model: usize, indices: []const u32, grad_out: []const f32, grad_table: []f32) void {
        const eng  = getEngine() catch @panic("Metal init failed");
        const ntok = indices.len;

        const buf_idx = eng.getScratch(u32, ntok, 30);
        @memcpy(buf_idx.asSlice(u32)[0..ntok], indices);
        const buf_go  = eng.getOrUpload(grad_out);
        // Upload current grad_table so atomic adds start from the right base
        const buf_gt  = eng.getOrUploadMut(grad_table);

        eng.beginRecording();
        const pipe = eng.getPipeline("embedding_bwd") catch @panic("pipeline embedding_bwd");
        eng.encodeTyped(pipe, &.{ buf_idx, buf_go, buf_gt },
            EmbedParams{ .d_model = @intCast(d_model) },
            .{ .x = @intCast(d_model), .y = @intCast(ntok) }, null);
        eng.commitAndWait();
        eng.downloadTo(buf_gt, grad_table);
    }
};

pub const MetalDropout = struct {
    pub fn forward(input: []const f32, output: []f32, mask: []bool, rate: f32, training: bool, rng: *std.Random.DefaultPrng) void {
        // Generate mask on CPU (needs rng state), then apply on GPU.
        if (training) {
            for (mask) |*m| m.* = rng.random().float(f32) >= rate;
        }
        const eng  = getEngine() catch @panic("Metal init failed");
        const len  = input.len;
        const scale: f32 = if (rate < 1.0) 1.0 / (1.0 - rate) else 1.0;

        const buf_in   = eng.getOrUpload(input);
        // Slot 32: mask (bool = u8), slot 33: output
        const buf_mask = eng.getScratch(u8, len, 32);
        @memcpy(buf_mask.asSlice(u8)[0..len], @as([]const u8, @ptrCast(mask)));
        const buf_out  = eng.getScratch(f32, len, 33);

        eng.beginRecording();
        const pipe = eng.getPipeline("dropout_fwd") catch @panic("pipeline dropout_fwd");
        eng.encodeTyped(pipe, &.{ buf_in, buf_mask, buf_out },
            DropoutFwdParams{ .length = @intCast(len), .scale = scale, .training = if (training) 1 else 0 },
            .{ .x = @intCast(len) }, null);
        eng.commitAndWait();
        eng.downloadTo(buf_out, output);
    }

    pub fn backward(grad_out: []const f32, grad_in: []f32, mask: []const bool, rate: f32) void {
        const eng  = getEngine() catch @panic("Metal init failed");
        const len  = grad_out.len;
        const scale: f32 = if (rate < 1.0) 1.0 / (1.0 - rate) else 1.0;

        const buf_go   = eng.getOrUpload(grad_out);
        const buf_mask = eng.getScratch(u8, len, 32);
        @memcpy(buf_mask.asSlice(u8)[0..len], @as([]const u8, @ptrCast(mask)));
        const buf_gi   = eng.getScratch(f32, len, 33);

        eng.beginRecording();
        const pipe = eng.getPipeline("dropout_bwd") catch @panic("pipeline dropout_bwd");
        eng.encodeTyped(pipe, &.{ buf_go, buf_mask, buf_gi },
            DropoutBwdParams{ .length = @intCast(len), .scale = scale },
            .{ .x = @intCast(len) }, null);
        eng.commitAndWait();
        eng.downloadTo(buf_gi, grad_in);
    }
};

// ── FusedOps: encode-only helpers for recording within an active command buffer ──
// These do NOT call beginRecording/commitAndWait/downloadTo.
// Caller must manage the recording lifecycle.

pub const NormResult = struct { out: GpuBuffer, x_norm: GpuBuffer, rstd: GpuBuffer };
pub const FFNResult = struct { pre_act: GpuBuffer, output: GpuBuffer };
pub const AttnFwdResult = struct {
    q: GpuBuffer, k: GpuBuffer, v: GpuBuffer,
    scores: GpuBuffer, ctx: GpuBuffer, output: GpuBuffer,
};

const LNFwdParams = extern struct { d: u32, eps: f32 };
const LNBwdParams = extern struct { d: u32 };
const RMSFwdParams = extern struct { d: u32, eps: f32 };
const RMSBwdParams = extern struct { d: u32 };

pub const FusedOps = struct {

    pub fn encodeLayernormFwd(
        eng: *MetalEngine, buf_x: GpuBuffer, buf_gamma: GpuBuffer, buf_beta: GpuBuffer,
        eps: f32, seq: usize, d: usize, slot_base: u8,
    ) NormResult {
        const buf_out = eng.getScratch(f32, seq * d, slot_base);
        const buf_xn = eng.getScratch(f32, seq * d, slot_base + 1);
        const buf_rs = eng.getScratch(f32, seq, slot_base + 2);
        const pipe = eng.getPipeline("layernorm_fwd_seq") catch @panic("pipeline");
        const tg = Grid{ .x = @intCast(d), .y = 1 };
        eng.encodeTgMem(pipe, &.{ buf_x, buf_gamma, buf_beta, buf_out, buf_xn, buf_rs },
            @ptrCast(&LNFwdParams{ .d = @intCast(d), .eps = eps }), @sizeOf(LNFwdParams),
            .{ .x = @intCast(d), .y = @intCast(seq) }, tg, d * @sizeOf(f32));
        return .{ .out = buf_out, .x_norm = buf_xn, .rstd = buf_rs };
    }

    pub fn encodeRmsnormFwd(
        eng: *MetalEngine, buf_x: GpuBuffer, buf_gamma: GpuBuffer,
        eps: f32, seq: usize, d: usize, slot_base: u8,
    ) NormResult {
        const buf_out = eng.getScratch(f32, seq * d, slot_base);
        const buf_xn = eng.getScratch(f32, seq * d, slot_base + 1);
        const buf_rms = eng.getScratch(f32, seq, slot_base + 2);
        const pipe = eng.getPipeline("rmsnorm_fwd_seq") catch @panic("pipeline");
        const tg = Grid{ .x = @intCast(d), .y = 1 };
        eng.encodeTgMem(pipe, &.{ buf_x, buf_gamma, buf_out, buf_xn, buf_rms },
            @ptrCast(&RMSFwdParams{ .d = @intCast(d), .eps = eps }), @sizeOf(RMSFwdParams),
            .{ .x = @intCast(d), .y = @intCast(seq) }, tg, d * @sizeOf(f32));
        return .{ .out = buf_out, .x_norm = buf_xn, .rstd = buf_rms };
    }

    pub fn encodeFFNFwd(
        eng: *MetalEngine, buf_input: GpuBuffer, buf_w: GpuBuffer, buf_b: GpuBuffer,
        activation: Activation, n_in: usize, n_out: usize, seq: usize, slot_base: u8,
    ) FFNResult {
        const total = seq * n_out;
        const buf_z = eng.getScratch(f32, total, slot_base);
        const buf_za = eng.getScratch(f32, total, slot_base + 1);

        // z = input × W^T
        const bT_pipe = eng.getPipeline("matmul_bT_tiled_f32") catch @panic("pipeline");
        const mp = MatmulParams{ .M = @intCast(seq), .K = @intCast(n_in), .N = @intCast(n_out) };
        encodeTiledMatmul(eng, bT_pipe, &.{ buf_input, buf_w, buf_z }, mp, mp.M, mp.N);

        // za = z + bias
        const bias_pipe = eng.getPipeline("add_bias_batch_f32") catch @panic("pipeline");
        eng.encodeTyped(bias_pipe, &.{ buf_z, buf_b, buf_za }, AddBiasBatchParams{
            .seq = @intCast(seq), .n_out = @intCast(n_out),
        }, .{ .x = @intCast(n_out), .y = @intCast(seq) }, null);

        // activation
        const act_kernel = activationKernelName(activation);
        if (act_kernel) |name| {
            const buf_act = eng.getScratch(f32, total, slot_base + 2);
            const act_pipe = eng.getPipeline(name) catch @panic("pipeline");
            eng.encodeTyped(act_pipe, &.{ buf_za, buf_act }, ElementwiseParams{
                .length = @intCast(total),
            }, .{ .x = @intCast(total) }, null);
            return .{ .pre_act = buf_za, .output = buf_act };
        }
        return .{ .pre_act = buf_za, .output = buf_za };
    }

    pub fn encodeAttnFwd(
        eng: *MetalEngine,
        comptime d_model: usize, comptime n_heads: usize, comptime d_head: usize,
        attn_scale: f32, seq: usize, causal: bool,
        buf_x: GpuBuffer, buf_wq: GpuBuffer, buf_wk: GpuBuffer,
        buf_wv: GpuBuffer, buf_wo: GpuBuffer,
        slot_base: u8,
    ) AttnFwdResult {
        const smd = seq * d_model;
        const shh = n_heads * seq * seq;

        const buf_q = eng.getScratch(f32, smd, slot_base);
        const buf_k = eng.getScratch(f32, smd, slot_base + 1);
        const buf_v = eng.getScratch(f32, smd, slot_base + 2);
        const buf_sc = eng.getScratch(f32, shh, slot_base + 3);
        const buf_ctx = eng.getScratch(f32, smd, slot_base + 4);
        const buf_out = eng.getScratch(f32, smd, slot_base + 5);

        // Q, K, V projections
        const bT = eng.getPipeline("matmul_bT_tiled_f32") catch @panic("pipeline");
        const mp = MatmulParams{ .M = @intCast(seq), .K = @intCast(d_model), .N = @intCast(d_model) };
        encodeTiledMatmul(eng, bT, &.{ buf_x, buf_wq, buf_q }, mp, mp.M, mp.N);
        encodeTiledMatmul(eng, bT, &.{ buf_x, buf_wk, buf_k }, mp, mp.M, mp.N);
        encodeTiledMatmul(eng, bT, &.{ buf_x, buf_wv, buf_v }, mp, mp.M, mp.N);

        // attention scores
        const sp = AttnScoreParams{
            .seq = @intCast(seq), .n_heads = @intCast(n_heads),
            .d_head = @intCast(d_head), .scale = attn_scale,
            .causal = if (causal) 1 else 0,
        };
        const sc_pipe = eng.getPipeline("attn_scores_fwd") catch @panic("pipeline");
        eng.encodeTyped(sc_pipe, &.{ buf_q, buf_k, buf_sc }, sp,
            .{ .x = @intCast(seq), .y = @intCast(seq), .z = @intCast(n_heads) }, null);

        // softmax
        const sm_pipe = eng.getPipeline("softmax_rows_inplace") catch @panic("pipeline");
        const n_rows: u32 = @intCast(n_heads * seq);
        eng.encodeTgMem(sm_pipe, &.{buf_sc},
            @ptrCast(&SoftmaxParams{ .n_rows = n_rows, .row_len = @intCast(seq) }),
            @sizeOf(SoftmaxParams),
            .{ .x = @intCast(seq), .y = n_rows }, .{ .x = @intCast(seq), .y = 1 }, seq * @sizeOf(f32));

        // context
        const ctx_pipe = eng.getPipeline("attn_context_fwd") catch @panic("pipeline");
        const cp = AttnCtxParams{ .seq = @intCast(seq), .n_heads = @intCast(n_heads), .d_head = @intCast(d_head) };
        eng.encodeTyped(ctx_pipe, &.{ buf_sc, buf_v, buf_ctx }, cp,
            .{ .x = @intCast(d_head), .y = @intCast(seq), .z = @intCast(n_heads) }, null);

        // output projection
        encodeTiledMatmul(eng, bT, &.{ buf_ctx, buf_wo, buf_out }, mp, mp.M, mp.N);

        return .{ .q = buf_q, .k = buf_k, .v = buf_v, .scores = buf_sc, .ctx = buf_ctx, .output = buf_out };
    }

    /// Encode: out[i] = a[i] + b[i]
    pub fn encodeAdd(eng: *MetalEngine, buf_a: GpuBuffer, buf_b: GpuBuffer, len: usize, slot: u8) GpuBuffer {
        const buf_out = eng.getScratch(f32, len, slot);
        const pipe = eng.getPipeline("add_f32") catch @panic("pipeline");
        eng.encodeTyped(pipe, &.{ buf_a, buf_b, buf_out }, ElementwiseParams{
            .length = @intCast(len),
        }, .{ .x = @intCast(len) }, null);
        return buf_out;
    }

    // ── Backward encode-only helpers ──

    pub fn encodeFFNBwd(
        eng: *MetalEngine, buf_pre_act: GpuBuffer, buf_grad_out: GpuBuffer,
        buf_input: GpuBuffer, buf_w: GpuBuffer, buf_gw: GpuBuffer, buf_gb: GpuBuffer,
        activation: Activation, n_in: usize, n_out: usize, seq: usize, slot_base: u8,
    ) struct { grad_in: GpuBuffer } {
        const total = seq * n_out;

        // delta = grad_out * act'(pre_act) — GPU
        const buf_delta = eng.getScratch(f32, total, slot_base);
        const act_bwd_pipe = eng.getPipeline("fused_act_backward") catch @panic("pipeline");
        eng.encodeTyped(act_bwd_pipe, &.{ buf_pre_act, buf_grad_out, buf_delta }, ActBwdParams{
            .length = @intCast(total), .act_type = activationTypeId(activation),
        }, .{ .x = @intCast(total) }, null);

        // grad_w += delta^T @ input
        const bw_b_pipe = eng.getPipeline("matmul_backward_b_tiled") catch @panic("pipeline");
        const mp = MatmulParams{ .M = @intCast(seq), .K = @intCast(n_out), .N = @intCast(n_in) };
        encodeTiledMatmul(eng, bw_b_pipe, &.{ buf_delta, buf_input, buf_gw }, mp, mp.K, mp.N);

        // grad_in = delta @ W
        const buf_gi = eng.getScratch(f32, seq * n_in, slot_base + 1);
        const mm_pipe = eng.getPipeline("matmul_tiled_f32") catch @panic("pipeline");
        const mp_gi = MatmulParams{ .M = @intCast(seq), .K = @intCast(n_out), .N = @intCast(n_in) };
        encodeTiledMatmul(eng, mm_pipe, &.{ buf_delta, buf_w, buf_gi }, mp_gi, mp_gi.M, mp_gi.N);

        // grad_b += sum of delta over seq
        const rb_pipe = eng.getPipeline("reduce_bias_grad") catch @panic("pipeline");
        eng.encodeTyped(rb_pipe, &.{ buf_delta, buf_gb }, ReduceBiasParams{
            .seq = @intCast(seq), .n_out = @intCast(n_out),
        }, .{ .x = @intCast(n_out) }, null);

        return .{ .grad_in = buf_gi };
    }

    pub fn encodeLayernormBwd(
        eng: *MetalEngine, buf_go: GpuBuffer, buf_xn: GpuBuffer,
        buf_gamma: GpuBuffer, buf_rstd: GpuBuffer,
        buf_gg: GpuBuffer, buf_gb: GpuBuffer,
        seq: usize, d: usize, slot: u8,
    ) GpuBuffer {
        const buf_gi = eng.getScratch(f32, seq * d, slot);
        const pipe = eng.getPipeline("layernorm_bwd_seq") catch @panic("pipeline");
        const tg = Grid{ .x = @intCast(d), .y = 1 };
        eng.encodeTgMem(pipe, &.{ buf_go, buf_xn, buf_gamma, buf_rstd, buf_gg, buf_gb, buf_gi },
            @ptrCast(&LNBwdParams{ .d = @intCast(d) }), @sizeOf(LNBwdParams),
            .{ .x = @intCast(d), .y = @intCast(seq) }, tg, d * @sizeOf(f32));
        return buf_gi;
    }

    pub fn encodeRmsnormBwd(
        eng: *MetalEngine, buf_go: GpuBuffer, buf_xn: GpuBuffer,
        buf_gamma: GpuBuffer, buf_rms: GpuBuffer, buf_gg: GpuBuffer,
        seq: usize, d: usize, slot: u8,
    ) GpuBuffer {
        const buf_gi = eng.getScratch(f32, seq * d, slot);
        const pipe = eng.getPipeline("rmsnorm_bwd_seq") catch @panic("pipeline");
        const tg = Grid{ .x = @intCast(d), .y = 1 };
        eng.encodeTgMem(pipe, &.{ buf_go, buf_xn, buf_gamma, buf_rms, buf_gg, buf_gi },
            @ptrCast(&RMSBwdParams{ .d = @intCast(d) }), @sizeOf(RMSBwdParams),
            .{ .x = @intCast(d), .y = @intCast(seq) }, tg, d * @sizeOf(f32));
        return buf_gi;
    }

    pub fn encodeAttnBwd(
        eng: *MetalEngine,
        comptime d_model: usize, comptime n_heads: usize, comptime d_head: usize,
        attn_scale: f32, seq: usize,
        buf_x: GpuBuffer, buf_go: GpuBuffer,
        buf_wq: GpuBuffer, buf_wk: GpuBuffer, buf_wv: GpuBuffer, buf_wo: GpuBuffer,
        buf_cq: GpuBuffer, buf_ck: GpuBuffer, buf_cv: GpuBuffer, buf_ca: GpuBuffer, buf_cc: GpuBuffer,
        buf_gwq: GpuBuffer, buf_gwk: GpuBuffer, buf_gwv: GpuBuffer, buf_gwo: GpuBuffer,
        slot_base: u8,
    ) GpuBuffer {
        const smd = seq * d_model;
        const shh = n_heads * seq * seq;
        const dm32: u32 = @intCast(d_model);
        const sq32: u32 = @intCast(seq);
        const nh32: u32 = @intCast(n_heads);
        const dh32: u32 = @intCast(d_head);

        const buf_gq = eng.getScratch(f32, smd, slot_base);
        const buf_gk = eng.getScratch(f32, smd, slot_base + 1);
        const buf_gv = eng.getScratch(f32, smd, slot_base + 2);
        const buf_gc = eng.getScratch(f32, smd, slot_base + 3);
        const buf_gp = eng.getScratch(f32, shh, slot_base + 4);
        const buf_gi_q = eng.getScratch(f32, smd, slot_base + 5);
        const buf_gi_k = eng.getScratch(f32, smd, slot_base + 6);
        const buf_gi_v = eng.getScratch(f32, smd, slot_base + 7);

        // Zero accumulators
        @memset(buf_gq.asSlice(f32)[0..smd], 0);
        @memset(buf_gk.asSlice(f32)[0..smd], 0);
        @memset(buf_gv.asSlice(f32)[0..smd], 0);

        const mm = eng.getPipeline("matmul_tiled_f32") catch @panic("pipeline");
        const bwB = eng.getPipeline("matmul_backward_b_tiled") catch @panic("pipeline");
        const mp = MatmulParams{ .M = sq32, .K = dm32, .N = dm32 };

        // Forward used concat @ Wo^T, so: grad_wo[m,n] = sum_t(grad_out[t,m] * concat[t,n])
        // => A=grad_out, grad_c=concat (swapped from naive A @ B formula)
        encodeTiledMatmul(eng, bwB, &.{ buf_go, buf_cc, buf_gwo }, mp, mp.K, mp.N);
        // Forward used concat @ Wo^T, so grad_concat = grad_out @ Wo (not Wo^T)
        encodeTiledMatmul(eng, mm, &.{ buf_go, buf_wo, buf_gc }, mp, mp.M, mp.N);

        const ctx_p = AttnCtxParams{ .seq = sq32, .n_heads = nh32, .d_head = dh32 };

        // grad_v
        const gv_pipe = eng.getPipeline("attn_grad_v") catch @panic("pipeline");
        eng.encodeTyped(gv_pipe, &.{ buf_ca, buf_gc, buf_gv }, ctx_p,
            .{ .x = dh32, .y = sq32, .z = nh32 }, null);

        // grad_attn_pre
        const gp_pipe = eng.getPipeline("attn_grad_attn_pre") catch @panic("pipeline");
        eng.encodeTyped(gp_pipe, &.{ buf_gc, buf_cv, buf_gp }, ctx_p,
            .{ .x = sq32, .y = sq32, .z = nh32 }, null);

        // softmax backward
        const n_rows: u32 = nh32 * sq32;
        const sb_pipe = eng.getPipeline("softmax_bwd_rows") catch @panic("pipeline");
        eng.encodeTgMem(sb_pipe, &.{ buf_ca, buf_gp },
            @ptrCast(&SoftmaxBwdParams{ .n_rows = n_rows, .row_len = sq32 }),
            @sizeOf(SoftmaxBwdParams),
            .{ .x = sq32, .y = n_rows }, .{ .x = sq32, .y = 1 }, seq * @sizeOf(f32));

        // grad_q, grad_k
        const qk_p = AttnGradQKParams{ .seq = sq32, .n_heads = nh32, .d_head = dh32, .scale = attn_scale };
        const gq_pipe = eng.getPipeline("attn_grad_q") catch @panic("pipeline");
        eng.encodeTyped(gq_pipe, &.{ buf_gp, buf_ck, buf_gq }, qk_p,
            .{ .x = dh32, .y = sq32, .z = nh32 }, null);
        const gk_pipe = eng.getPipeline("attn_grad_k") catch @panic("pipeline");
        eng.encodeTyped(gk_pipe, &.{ buf_gp, buf_cq, buf_gk }, qk_p,
            .{ .x = dh32, .y = sq32, .z = nh32 }, null);

        // Forward used X @ W^T, so: grad_w[m,n] = sum_t(grad[t,m] * X[t,n]) => A=grad, grad_c=X
        // grad_in = grad @ W (not W^T) since forward was X @ W^T
        encodeTiledMatmul(eng, bwB, &.{ buf_gq, buf_x, buf_gwq }, mp, mp.K, mp.N);
        encodeTiledMatmul(eng, mm, &.{ buf_gq, buf_wq, buf_gi_q }, mp, mp.M, mp.N);
        encodeTiledMatmul(eng, bwB, &.{ buf_gk, buf_x, buf_gwk }, mp, mp.K, mp.N);
        encodeTiledMatmul(eng, mm, &.{ buf_gk, buf_wk, buf_gi_k }, mp, mp.M, mp.N);
        encodeTiledMatmul(eng, bwB, &.{ buf_gv, buf_x, buf_gwv }, mp, mp.K, mp.N);
        encodeTiledMatmul(eng, mm, &.{ buf_gv, buf_wv, buf_gi_v }, mp, mp.M, mp.N);

        // Sum three grad_in contributions on GPU
        const buf_gi = eng.getScratch(f32, smd, slot_base + 8);
        const add3_pipe = eng.getPipeline("add3_f32") catch @panic("pipeline");
        eng.encodeTyped(add3_pipe, &.{ buf_gi_q, buf_gi_k, buf_gi_v, buf_gi }, ElementwiseParams{
            .length = @intCast(smd),
        }, .{ .x = @intCast(smd) }, null);

        return buf_gi;
    }
};
