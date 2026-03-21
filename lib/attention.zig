const std = @import("std");
const backend_mod = @import("backend.zig");
const Backend = backend_mod.Backend;
const optimizer_mod = @import("optimizer.zig");
const Optimizer = optimizer_mod.Optimizer;

/// Scaled Dot-Product Multi-Head Attention.
///
///   head_h = softmax(Q_h @ K_h^T / sqrt(d_head)) @ V_h
///   output  = concat(heads) @ W_o
///
/// Compile-time parameters:
///   backend  -- compute backend
///   d_model  -- model dimension
///   n_heads  -- number of heads (d_model % n_heads == 0)
///   max_seq  -- maximum sequence length
///   causal   -- if true, apply causal (lower-triangular) mask before softmax
///
/// The struct lives on the heap (init returns *Self).
pub fn MultiHeadAttention(
    comptime backend: Backend,
    comptime d_model: usize,
    comptime n_heads: usize,
    comptime max_seq: usize,
    comptime causal:  bool,
) type {
    comptime {
        if (d_model % n_heads != 0)
            @compileError("d_model must be divisible by n_heads");
    }
    const d_head = d_model / n_heads;
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    return struct {
        // ── weights (f16 storage, f32 compute) ─────────────────────────
        wq: [d_model * d_model]f16,
        wk: [d_model * d_model]f16,
        wv: [d_model * d_model]f16,
        wo: [d_model * d_model]f16,

        wq_f: [d_model * d_model]f32,
        wk_f: [d_model * d_model]f32,
        wv_f: [d_model * d_model]f32,
        wo_f: [d_model * d_model]f32,

        // ── weight gradients ────────────────────────────────────────────
        grad_wq: [d_model * d_model]f32,
        grad_wk: [d_model * d_model]f32,
        grad_wv: [d_model * d_model]f32,
        grad_wo: [d_model * d_model]f32,

        // ── Adam moment buffers for wq, wk, wv, wo ─────────────────────
        m_wq: [d_model * d_model]f32,
        v_wq: [d_model * d_model]f32,
        m_wk: [d_model * d_model]f32,
        v_wk: [d_model * d_model]f32,
        m_wv: [d_model * d_model]f32,
        v_wv: [d_model * d_model]f32,
        m_wo: [d_model * d_model]f32,
        v_wo: [d_model * d_model]f32,

        // ── forward cache (needed in backward) ──────────────────────────
        cache_q:      [max_seq * d_model]f32,          // Q = X @ W_q
        cache_k:      [max_seq * d_model]f32,          // K = X @ W_k
        cache_v:      [max_seq * d_model]f32,          // V = X @ W_v
        cache_attn:   [n_heads * max_seq * max_seq]f32, // attention weights per head
        cache_concat: [max_seq * d_model]f32,          // head concat before W_o

        // ── backward buffers ────────────────────────────────────────────
        grad_q:      [max_seq * d_model]f32,
        grad_k:      [max_seq * d_model]f32,
        grad_v:      [max_seq * d_model]f32,
        grad_concat: [max_seq * d_model]f32,

        // ── scratch (reused across calls) ───────────────────────────────
        scratch: [max_seq * max_seq]f32,

        last_seq: usize,

        const Self    = @This();
        const Impl    = backend_mod.AttentionImpl(backend);
        const OptImpl = backend_mod.OptimizerImpl(backend);

        // ── init / deinit ───────────────────────────────────────────────

        /// Allocate and initialize on the heap. Xavier init for weights.
        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);

            var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
            // Xavier: std = sqrt(2 / (fan_in + fan_out))
            const xavier: f32 = @sqrt(2.0 / @as(f32, @floatFromInt(d_model + d_model)));

            for (&self.wq) |*w| w.* = @floatCast(rng.random().floatNorm(f32) * xavier);
            for (&self.wk) |*w| w.* = @floatCast(rng.random().floatNorm(f32) * xavier);
            for (&self.wv) |*w| w.* = @floatCast(rng.random().floatNorm(f32) * xavier);
            for (&self.wo) |*w| w.* = @floatCast(rng.random().floatNorm(f32) * xavier);

            @memset(&self.grad_wq, 0);
            @memset(&self.grad_wk, 0);
            @memset(&self.grad_wv, 0);
            @memset(&self.grad_wo, 0);

            @memset(&self.m_wq, 0);
            @memset(&self.v_wq, 0);
            @memset(&self.m_wk, 0);
            @memset(&self.v_wk, 0);
            @memset(&self.m_wv, 0);
            @memset(&self.v_wv, 0);
            @memset(&self.m_wo, 0);
            @memset(&self.v_wo, 0);

            self.last_seq = 0;

            return self;
        }

        /// Destroy the heap allocation.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.destroy(self);
        }

        // ── forward ─────────────────────────────────────────────────────

        /// Forward pass.
        /// input:  [seq * d_model]  (flat, row-major)
        /// output: [seq * d_model]
        pub fn forward(self: *Self, input: []const f32, output: []f32, seq: usize) void {
            std.debug.assert(seq <= max_seq);

            self.last_seq = seq;
            // f16 -> f32
            for (self.wq, &self.wq_f) |w, *wf| wf.* = @floatCast(w);
            for (self.wk, &self.wk_f) |w, *wf| wf.* = @floatCast(w);
            for (self.wv, &self.wv_f) |w, *wf| wf.* = @floatCast(w);
            for (self.wo, &self.wo_f) |w, *wf| wf.* = @floatCast(w);

            Impl.forward(d_model, n_heads, d_head, max_seq, attn_scale, seq, input, output,
                &self.wq_f, &self.wk_f, &self.wv_f, &self.wo_f,
                &self.cache_q, &self.cache_k, &self.cache_v, &self.cache_attn, &self.cache_concat,
                &self.scratch, causal);
        }

        // ── backward ────────────────────────────────────────────────────

        /// Backward pass.
        /// input:    original forward input [seq * d_model]
        /// grad_out: [seq * d_model]
        /// grad_in:  [seq * d_model] (written)
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) void {
            const seq = self.last_seq;

            Impl.backward(d_model, n_heads, d_head, attn_scale, seq, input, grad_out, grad_in,
                &self.wq_f, &self.wk_f, &self.wv_f, &self.wo_f,
                &self.cache_q, &self.cache_k, &self.cache_v, &self.cache_attn, &self.cache_concat,
                &self.grad_wq, &self.grad_wk, &self.grad_wv, &self.grad_wo,
                &self.grad_q, &self.grad_k, &self.grad_v, &self.grad_concat, &self.scratch);
        }

        // ── weight update ────────────────────────────────────────────────

        /// Update weights using the given optimizer. t is the 1-based step counter.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            // Convert f16 -> f32
            for (self.wq, &self.wq_f) |w, *wf| wf.* = @floatCast(w);
            for (self.wk, &self.wk_f) |w, *wf| wf.* = @floatCast(w);
            for (self.wv, &self.wv_f) |w, *wf| wf.* = @floatCast(w);
            for (self.wo, &self.wo_f) |w, *wf| wf.* = @floatCast(w);

            OptImpl.update(opt, t, lr, &self.wq_f, &self.grad_wq, &self.m_wq, &self.v_wq);
            OptImpl.update(opt, t, lr, &self.wk_f, &self.grad_wk, &self.m_wk, &self.v_wk);
            OptImpl.update(opt, t, lr, &self.wv_f, &self.grad_wv, &self.m_wv, &self.v_wv);
            OptImpl.update(opt, t, lr, &self.wo_f, &self.grad_wo, &self.m_wo, &self.v_wo);

            // Sync f32 -> f16
            for (&self.wq, self.wq_f) |*w, wf| w.* = @floatCast(wf);
            for (&self.wk, self.wk_f) |*w, wf| w.* = @floatCast(wf);
            for (&self.wv, self.wv_f) |*w, wf| w.* = @floatCast(wf);
            for (&self.wo, self.wo_f) |*w, wf| w.* = @floatCast(wf);
        }

        /// Returns the sum of squared gradients.
        pub fn gradNormSq(self: *const Self) f32 {
            var sum: f32 = 0;
            for (self.grad_wq) |g| sum += g * g;
            for (self.grad_wk) |g| sum += g * g;
            for (self.grad_wv) |g| sum += g * g;
            for (self.grad_wo) |g| sum += g * g;
            return sum;
        }

        /// Multiply all gradients by s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            for (&self.grad_wq) |*g| g.* *= s;
            for (&self.grad_wk) |*g| g.* *= s;
            for (&self.grad_wv) |*g| g.* *= s;
            for (&self.grad_wo) |*g| g.* *= s;
        }
    };
}
