const std = @import("std");

const backend_mod = @import("backend.zig");
const Backend     = backend_mod.Backend;

const LayerNormT = @import("layernorm.zig").LayerNorm;
const RMSNormT   = @import("rmsnorm.zig").RMSNorm;
const LayerT     = @import("layer.zig").Layer;
const MhaT       = @import("attention.zig").MultiHeadAttention;
const Activation = @import("activation.zig").Activation;
const Optimizer  = @import("optimizer.zig").Optimizer;

/// Compile-time configuration for a TransformerBlock.
pub const BlockConfig = struct {
    /// Which normalization layer to use before each sub-layer.
    norm:           NormType,
    /// Activation function for the first FFN linear layer.
    ffn_activation: Activation,
    /// If true, attention uses a causal (lower-triangular) mask.
    causal:         bool,

    pub const NormType = enum { layer_norm, rms_norm };
};

/// Pre-LN Transformer Block:
///
///   out = FFN(Norm2(h)) + h
///   h   = Attn(Norm1(x)) + x
///
/// Compile-time parameters:
///   backend  -- compute backend
///   d_model  -- model dimension
///   n_heads  -- number of attention heads (d_model % n_heads == 0)
///   d_ff     -- FFN hidden size (typically 4 * d_model)
///   max_seq  -- maximum sequence length
///   cfg      -- BlockConfig: norm type, FFN activation, causal flag
///
/// The struct lives on the heap (init returns *Self).
/// input/output: flat slices [seq * d_model], row-major.
pub fn TransformerBlock(
    comptime backend: Backend,
    comptime d_model:  usize,
    comptime n_heads:  usize,
    comptime d_ff:     usize,
    comptime max_seq:  usize,
    comptime cfg:      BlockConfig,
) type {
    const Norm = switch (cfg.norm) {
        .layer_norm => LayerNormT(backend, d_model),
        .rms_norm   => RMSNormT(backend, d_model),
    };
    const Fc  = LayerT(backend);
    const Mha = MhaT(backend, d_model, n_heads, max_seq, cfg.causal);

    return struct {
        pub const backend_tag = backend;

        // --- Metal-only GPU buffer cache (intermediates that survive fwd->bwd without CPU roundtrip) ---
        const MetalCaches = if (backend == .metal) struct {
            norm1_out:    @import("backend/metal.zig").GpuBuffer,
            norm1_xn:     @import("backend/metal.zig").GpuBuffer,
            norm1_rstd:   @import("backend/metal.zig").GpuBuffer,
            h:            @import("backend/metal.zig").GpuBuffer,
            norm2_xn:     @import("backend/metal.zig").GpuBuffer,
            norm2_rstd:   @import("backend/metal.zig").GpuBuffer,
            norm2_out:    @import("backend/metal.zig").GpuBuffer,
            ffn1_pre:     @import("backend/metal.zig").GpuBuffer,
            ffn1_out:     @import("backend/metal.zig").GpuBuffer,
            ffn2_pre:     @import("backend/metal.zig").GpuBuffer,
            attn_q:       @import("backend/metal.zig").GpuBuffer,
            attn_k:       @import("backend/metal.zig").GpuBuffer,
            attn_v:       @import("backend/metal.zig").GpuBuffer,
            attn_scores:  @import("backend/metal.zig").GpuBuffer,
            attn_concat:  @import("backend/metal.zig").GpuBuffer,
        } else struct {};

        const Self = @This();

        //  modules
        norm1: Norm,
        norm2: Norm,
        attn:  *Mha,
        ffn1:  Fc,    // d_model -> d_ff,    cfg.ffn_activation
        ffn2:  Fc,    // d_ff    -> d_model,  linear

        //  intermediate forward activations
        norm1_out: [max_seq * d_model]f32,
        h:         [max_seq * d_model]f32,
        norm2_out: [max_seq * d_model]f32,
        ffn1_out:  [max_seq * d_ff]f32,

        //  per-position norm caches (needed in backward)
        // Both LayerNorm and RMSNorm expose x_norm_buf and last_rstd.
        norm1_xn:   [max_seq * d_model]f32,
        norm1_rstd: [max_seq]f32,
        norm2_xn:   [max_seq * d_model]f32,
        norm2_rstd: [max_seq]f32,

        //  per-position pre-activation Layer caches (needed in backward) ─
        ffn1_pre: [max_seq * d_ff]f32,
        ffn2_pre: [max_seq * d_model]f32,

        //  backward gradient buffers
        grad_ffn1_out:  [max_seq * d_ff]f32,
        grad_norm2_out: [max_seq * d_model]f32,
        grad_h:         [max_seq * d_model]f32,
        grad_norm1_out: [max_seq * d_model]f32,

        //  scratch: temporary buffer for per-position norm backward
        scratch: [d_model]f32,

        last_seq:      usize,
        fwd_slot_base: u16,
        bwd_slot_base: u16,
        allocator: std.mem.Allocator,
        metal_caches: MetalCaches,

        //  init / deinit

        /// Allocate and initialize a TransformerBlock on the heap.
        pub fn init(allocator: std.mem.Allocator) !*Self {
            return initWithSlots(allocator, 100, 300);
        }

        pub fn initWithSlots(allocator: std.mem.Allocator, fwd_slot_base: u16, bwd_slot_base: u16) !*Self {
            const self = try allocator.create(Self);
            self.allocator = allocator;
            self.last_seq  = 0;
            self.fwd_slot_base = fwd_slot_base;
            self.bwd_slot_base = bwd_slot_base;
            self.metal_caches = std.mem.zeroes(MetalCaches);

            self.norm1 = Norm.init(1e-5);
            self.norm2 = Norm.init(1e-5);
            self.attn  = try Mha.init(allocator);
            self.ffn1  = try Fc.init(allocator, .{
                .n_in = d_model, .n_out = d_ff,    .activation = cfg.ffn_activation,
            });
            self.ffn2  = try Fc.init(allocator, .{
                .n_in = d_ff,    .n_out = d_model, .activation = .linear,
            });
            return self;
        }

        /// Free all resources.
        pub fn deinit(self: *Self) void {
            self.attn.deinit(self.allocator);
            self.ffn1.deinit();
            self.ffn2.deinit();
            self.allocator.destroy(self);
        }

        //  forward

        /// Generic forward: seq is inferred from input.len / d_model.
        pub fn forward(self: *Self, input: []const f32, output: []f32) void {
            self.forwardSeq(input, output, input.len / d_model);
        }

        /// Metal-only: encode the full block forward into the active command buffer recording.
        /// Returns the output GpuBuffer. Stores intermediate GPU handles in self.metal_caches.
        /// Caller is responsible for beginRecording/commitAndWait.
        pub fn encodeForward(self: *Self, eng: *@import("backend/metal.zig").MetalEngine, buf_input: @import("backend/metal.zig").GpuBuffer, seq: usize) @import("backend/metal.zig").GpuBuffer {
            comptime std.debug.assert(backend == .metal);
            const mb = @import("backend/metal.zig");
            const FO = mb.FusedOps;
            const smd = seq * d_model;
            self.last_seq = seq;
            const fwd: u16 = self.fwd_slot_base;

            // Sync f16→f32 only on first use; after that the f32 shadows hold Adam-updated values.
            if (eng.getCached(self.ffn1.weights_compute_buffer) == null) self.ffn1.syncWeights();
            if (eng.getCached(self.ffn2.weights_compute_buffer) == null) self.ffn2.syncWeights();
            if (eng.getCached(&self.attn.wq_f) == null) self.attn.syncWeightsF32();

            const buf_gamma1 = eng.getOrUpload(self.norm1.gamma[0..d_model]);
            const buf_gamma2 = eng.getOrUpload(self.norm2.gamma[0..d_model]);
            const buf_wq = eng.getOrUpload(&self.attn.wq_f);
            const buf_wk = eng.getOrUpload(&self.attn.wk_f);
            const buf_wv = eng.getOrUpload(&self.attn.wv_f);
            const buf_wo = eng.getOrUpload(&self.attn.wo_f);
            const buf_f1w = eng.getOrUpload(self.ffn1.weights_compute_buffer);
            const buf_f1b = eng.getOrUpload(self.ffn1.biases);
            const buf_f2w = eng.getOrUpload(self.ffn2.weights_compute_buffer);
            const buf_f2b = eng.getOrUpload(self.ffn2.biases);

            const n1 = switch (cfg.norm) {
                .layer_norm => FO.encodeLayernormFwd(eng, buf_input, buf_gamma1,
                    eng.getOrUpload(self.norm1.beta[0..d_model]), self.norm1.eps, seq, d_model, fwd),
                .rms_norm   => FO.encodeRmsnormFwd(eng, buf_input, buf_gamma1,
                    self.norm1.eps, seq, d_model, fwd),
            };

            const attn_r = FO.encodeAttnFwd(eng, d_model, n_heads, d_model / n_heads,
                Mha.attn_scale_val, seq, cfg.causal,
                n1.out, buf_wq, buf_wk, buf_wv, buf_wo, fwd + 3);

            const buf_h = FO.encodeAdd(eng, attn_r.output, buf_input, smd, fwd + 9);

            const n2 = switch (cfg.norm) {
                .layer_norm => FO.encodeLayernormFwd(eng, buf_h, buf_gamma2,
                    eng.getOrUpload(self.norm2.beta[0..d_model]), self.norm2.eps, seq, d_model, fwd + 10),
                .rms_norm   => FO.encodeRmsnormFwd(eng, buf_h, buf_gamma2,
                    self.norm2.eps, seq, d_model, fwd + 10),
            };

            const f1 = FO.encodeFFNFwd(eng, n2.out, buf_f1w, buf_f1b, cfg.ffn_activation, d_model, d_ff, seq, fwd + 13);
            const f2 = FO.encodeFFNFwd(eng, f1.output, buf_f2w, buf_f2b, .linear, d_ff, d_model, seq, fwd + 16);
            const buf_final = FO.encodeAdd(eng, f2.output, buf_h, smd, fwd + 19);

            self.metal_caches = .{
                .norm1_out   = n1.out,
                .norm1_xn    = n1.x_norm,
                .norm1_rstd  = n1.rstd,
                .h           = buf_h,
                .norm2_xn    = n2.x_norm,
                .norm2_rstd  = n2.rstd,
                .norm2_out   = n2.out,
                .ffn1_pre    = f1.pre_act,
                .ffn1_out    = f1.output,
                .ffn2_pre    = f2.pre_act,
                .attn_q      = attn_r.q,
                .attn_k      = attn_r.k,
                .attn_v      = attn_r.v,
                .attn_scores = attn_r.scores,
                .attn_concat = attn_r.ctx,
            };

            return buf_final;
        }

        /// Explicit-seq forward.
        /// input  -- [seq * d_model], row-major
        /// output -- [seq * d_model]
        pub fn forwardSeq(self: *Self, input: []const f32, output: []f32, seq: usize) void {
            std.debug.assert(seq <= max_seq);
            std.debug.assert(input.len  == seq * d_model);
            std.debug.assert(output.len == seq * d_model);
            self.last_seq = seq;
            const smd = seq * d_model;

            if (comptime backend == .metal) { // fused forward
                const mb = @import("backend/metal.zig");
                const eng = mb.getEngine() catch @panic("Metal init failed");
                eng.waitIfPending();

                const buf_input = eng.getOrUpload(input);
                eng.beginRecording();
                const buf_out = self.encodeForward(eng, buf_input, seq);
                eng.commitAndWait();
                eng.downloadTo(buf_out, output[0..smd]);
            } else {
                // sublayer 1: Norm1 -> Attn -> +x
                for (0..seq) |t| {
                    const x  = input[t * d_model ..][0..d_model];
                    const no = self.norm1_out[t * d_model ..][0..d_model];
                    self.norm1.forward(x, no);
                    @memcpy(self.norm1_xn[t * d_model ..][0..d_model], &self.norm1.x_norm_buf);
                    self.norm1_rstd[t] = self.norm1.last_rstd;
                }

                self.attn.forward(
                    self.norm1_out[0..smd],
                    self.h[0..smd],
                    seq,
                );

                for (self.h[0..smd], input) |*hi, xi| hi.* += xi;

                // sublayer 2: Norm2 -> FFN -> +h
                for (0..seq) |t| {
                    const x  = self.h[t * d_model ..][0..d_model];
                    const no = self.norm2_out[t * d_model ..][0..d_model];
                    self.norm2.forward(x, no);
                    @memcpy(self.norm2_xn[t * d_model ..][0..d_model], &self.norm2.x_norm_buf);
                    self.norm2_rstd[t] = self.norm2.last_rstd;
                }

                for (0..seq) |t| {
                    const n2o = self.norm2_out[t * d_model ..][0..d_model];
                    const f1o = self.ffn1_out[t * d_ff ..][0..d_ff];
                    self.ffn1.forward(n2o, f1o);
                    @memcpy(self.ffn1_pre[t * d_ff ..][0..d_ff], self.ffn1.pre_activation_buffer);

                    const f2o = output[t * d_model ..][0..d_model];
                    self.ffn2.forward(f1o, f2o);
                    @memcpy(self.ffn2_pre[t * d_model ..][0..d_model], self.ffn2.pre_activation_buffer);
                }
            }

            // CPU path: residual add (Metal path handles this in the fused GPU recording)
            if (comptime backend != .metal) {
                for (output[0..smd], self.h[0..smd]) |*oi, hi| oi.* += hi;
            }
        }

        //  backward

        /// Metal-only: encode the full block backward into the active command buffer recording.
        /// Uses self.metal_caches set during encodeForward. Returns grad_in GpuBuffer.
        /// All gradient accumulators are uploaded before encoding and must be downloaded after commitAndWait.
        /// Caller is responsible for beginRecording/commitAndWait/downloadGrads.
        pub fn encodeBackward(self: *Self, eng: *@import("backend/metal.zig").MetalEngine, buf_go: @import("backend/metal.zig").GpuBuffer, seq: usize) @import("backend/metal.zig").GpuBuffer {
            comptime std.debug.assert(backend == .metal);
            const mb = @import("backend/metal.zig");
            const FO = mb.FusedOps;
            const smd = seq * d_model;
            const bwd: u16 = self.bwd_slot_base;

            // Upload params
            const buf_gamma1 = eng.getOrUpload(self.norm1.gamma[0..d_model]);
            const buf_gamma2 = eng.getOrUpload(self.norm2.gamma[0..d_model]);
            const buf_gg1 = eng.getOrUploadMut(self.norm1.grad_gamma[0..d_model]);
            const buf_gg2 = eng.getOrUploadMut(self.norm2.grad_gamma[0..d_model]);
            const buf_wq = eng.getOrUpload(&self.attn.wq_f);
            const buf_wk = eng.getOrUpload(&self.attn.wk_f);
            const buf_wv = eng.getOrUpload(&self.attn.wv_f);
            const buf_wo = eng.getOrUpload(&self.attn.wo_f);
            const buf_gwq = eng.getOrUploadMut(&self.attn.grad_wq);
            const buf_gwk = eng.getOrUploadMut(&self.attn.grad_wk);
            const buf_gwv = eng.getOrUploadMut(&self.attn.grad_wv);
            const buf_gwo = eng.getOrUploadMut(&self.attn.grad_wo);
            const buf_f2w = eng.getOrUpload(self.ffn2.weights_compute_buffer);
            const buf_f1w = eng.getOrUpload(self.ffn1.weights_compute_buffer);
            const buf_f2_gw = eng.getOrUploadMut(self.ffn2.grad_w);
            const buf_f1_gw = eng.getOrUploadMut(self.ffn1.grad_w);
            const buf_f2_gb = eng.getOrUploadMut(self.ffn2.grad_b);
            const buf_f1_gb = eng.getOrUploadMut(self.ffn1.grad_b);

            // FFN2 backward
            const f2_bwd = FO.encodeFFNBwd(eng, self.metal_caches.ffn2_pre, buf_go,
                self.metal_caches.ffn1_out, buf_f2w, buf_f2_gw, buf_f2_gb, .linear, d_ff, d_model, seq, bwd);

            // FFN1 backward
            const f1_bwd = FO.encodeFFNBwd(eng, self.metal_caches.ffn1_pre, f2_bwd.grad_in,
                self.metal_caches.norm2_out, buf_f1w, buf_f1_gw, buf_f1_gb, cfg.ffn_activation, d_model, d_ff, seq, bwd + 3);

            // Norm2 backward
            const buf_n2_gi = switch (cfg.norm) {
                .layer_norm => FO.encodeLayernormBwd(eng, f1_bwd.grad_in, self.metal_caches.norm2_xn,
                    buf_gamma2, self.metal_caches.norm2_rstd, buf_gg2,
                    eng.getOrUploadMut(self.norm2.grad_beta[0..d_model]), seq, d_model, bwd + 6),
                .rms_norm   => FO.encodeRmsnormBwd(eng, f1_bwd.grad_in, self.metal_caches.norm2_xn,
                    buf_gamma2, self.metal_caches.norm2_rstd, buf_gg2, seq, d_model, bwd + 6),
            };

            // Residual: grad_h = grad_out + norm2_grad_in
            const buf_grad_h = FO.encodeAdd(eng, buf_go, buf_n2_gi, smd, bwd + 7);

            // Attention backward
            const buf_attn_gi = FO.encodeAttnBwd(eng, d_model, n_heads, d_model / n_heads,
                Mha.attn_scale_val, seq,
                self.metal_caches.norm1_out, buf_grad_h,
                buf_wq, buf_wk, buf_wv, buf_wo,
                self.metal_caches.attn_q, self.metal_caches.attn_k, self.metal_caches.attn_v,
                self.metal_caches.attn_scores, self.metal_caches.attn_concat,
                buf_gwq, buf_gwk, buf_gwv, buf_gwo, bwd + 8);

            // Norm1 backward
            const buf_n1_gi = switch (cfg.norm) {
                .layer_norm => FO.encodeLayernormBwd(eng, buf_attn_gi, self.metal_caches.norm1_xn,
                    buf_gamma1, self.metal_caches.norm1_rstd, buf_gg1,
                    eng.getOrUploadMut(self.norm1.grad_beta[0..d_model]), seq, d_model, bwd + 20),
                .rms_norm   => FO.encodeRmsnormBwd(eng, buf_attn_gi, self.metal_caches.norm1_xn,
                    buf_gamma1, self.metal_caches.norm1_rstd, buf_gg1, seq, d_model, bwd + 20),
            };

            // Final residual: grad_in = norm1_grad_in + grad_h
            return FO.encodeAdd(eng, buf_n1_gi, buf_grad_h, smd, bwd + 21);
        }

        /// Metal-only: download all gradient accumulators from GPU to CPU.
        /// Call after commitAndWait following encodeBackward.
        pub fn downloadGrads(self: *Self, eng: *@import("backend/metal.zig").MetalEngine) void {
            comptime std.debug.assert(backend == .metal);
            if (eng.getCached(self.ffn2.grad_w)) |buf| eng.downloadTo(buf, self.ffn2.grad_w);
            if (eng.getCached(self.ffn2.grad_b)) |buf| eng.downloadTo(buf, self.ffn2.grad_b);
            if (eng.getCached(self.ffn1.grad_w)) |buf| eng.downloadTo(buf, self.ffn1.grad_w);
            if (eng.getCached(self.ffn1.grad_b)) |buf| eng.downloadTo(buf, self.ffn1.grad_b);
            if (eng.getCached(self.norm1.grad_gamma[0..d_model])) |buf| eng.downloadTo(buf, self.norm1.grad_gamma[0..d_model]);
            if (eng.getCached(self.norm2.grad_gamma[0..d_model])) |buf| eng.downloadTo(buf, self.norm2.grad_gamma[0..d_model]);
            if (comptime cfg.norm == .layer_norm) {
                if (eng.getCached(self.norm1.grad_beta[0..d_model])) |buf| eng.downloadTo(buf, self.norm1.grad_beta[0..d_model]);
                if (eng.getCached(self.norm2.grad_beta[0..d_model])) |buf| eng.downloadTo(buf, self.norm2.grad_beta[0..d_model]);
            }
            if (eng.getCached(@as([]const f32, &self.attn.grad_wo))) |buf| eng.downloadTo(buf, &self.attn.grad_wo);
            if (eng.getCached(@as([]const f32, &self.attn.grad_wq))) |buf| eng.downloadTo(buf, &self.attn.grad_wq);
            if (eng.getCached(@as([]const f32, &self.attn.grad_wk))) |buf| eng.downloadTo(buf, &self.attn.grad_wk);
            if (eng.getCached(@as([]const f32, &self.attn.grad_wv))) |buf| eng.downloadTo(buf, &self.attn.grad_wv);
        }

        /// Metal-only: upload all gradient accumulators and weight params without encoding any kernels.
        /// Call before beginRecording to ensure all data is on GPU before the recording starts.
        pub fn prepareBackwardUploads(self: *Self, eng: *@import("backend/metal.zig").MetalEngine, seq: usize) void {
            _ = seq;
            comptime std.debug.assert(backend == .metal);
            // Params (read-only in backward)
            _ = eng.getOrUpload(self.norm1.gamma[0..d_model]);
            _ = eng.getOrUpload(self.norm2.gamma[0..d_model]);
            _ = eng.getOrUpload(&self.attn.wq_f);
            _ = eng.getOrUpload(&self.attn.wk_f);
            _ = eng.getOrUpload(&self.attn.wv_f);
            _ = eng.getOrUpload(&self.attn.wo_f);
            _ = eng.getOrUpload(self.ffn2.weights_compute_buffer);
            _ = eng.getOrUpload(self.ffn1.weights_compute_buffer);
            // Gradient accumulators (read-write in backward)
            _ = eng.getOrUploadMut(self.norm1.grad_gamma[0..d_model]);
            _ = eng.getOrUploadMut(self.norm2.grad_gamma[0..d_model]);
            _ = eng.getOrUploadMut(&self.attn.grad_wq);
            _ = eng.getOrUploadMut(&self.attn.grad_wk);
            _ = eng.getOrUploadMut(&self.attn.grad_wv);
            _ = eng.getOrUploadMut(&self.attn.grad_wo);
            _ = eng.getOrUploadMut(self.ffn2.grad_w);
            _ = eng.getOrUploadMut(self.ffn2.grad_b);
            _ = eng.getOrUploadMut(self.ffn1.grad_w);
            _ = eng.getOrUploadMut(self.ffn1.grad_b);
            if (comptime cfg.norm == .layer_norm) {
                _ = eng.getOrUploadMut(self.norm1.grad_beta[0..d_model]);
                _ = eng.getOrUploadMut(self.norm2.grad_beta[0..d_model]);
            }
        }

        /// Backward pass.
        /// input    -- original forward input [seq * d_model]
        /// grad_out -- [seq * d_model]
        /// grad_in  -- [seq * d_model] (written)
        pub fn backward(
            self:     *Self,
            input:    []const f32,
            grad_out: []const f32,
            grad_in:  []f32,
        ) !void {
            const seq = self.last_seq;
            const smd = seq * d_model;

            @memcpy(self.grad_h[0..smd], grad_out[0..smd]);

            if (comptime backend == .metal) {
                const mb = @import("backend/metal.zig");
                const eng = mb.getEngine() catch @panic("Metal init failed");
                eng.waitIfPending();

                const buf_go = eng.getOrUpload(grad_out[0..smd]);
                eng.beginRecording();
                const buf_final_gi = self.encodeBackward(eng, buf_go, seq);
                eng.commitAndWait();

                self.downloadGrads(eng);
                eng.downloadTo(buf_final_gi, grad_in[0..smd]);

            } else {
                for (0..seq) |t| {
                    @memcpy(self.ffn2.pre_activation_buffer, self.ffn2_pre[t * d_model ..][0..d_model]);
                    try self.ffn2.backward(
                        self.ffn1_out[t * d_ff ..][0..d_ff],
                        grad_out[t * d_model ..][0..d_model],
                        self.grad_ffn1_out[t * d_ff ..][0..d_ff],
                    );
                }

                for (0..seq) |t| {
                    @memcpy(self.ffn1.pre_activation_buffer, self.ffn1_pre[t * d_ff ..][0..d_ff]);
                    try self.ffn1.backward(
                        self.norm2_out[t * d_model ..][0..d_model],
                        self.grad_ffn1_out[t * d_ff ..][0..d_ff],
                        self.grad_norm2_out[t * d_model ..][0..d_model],
                    );
                }

                for (0..seq) |t| {
                    @memcpy(&self.norm2.x_norm_buf, self.norm2_xn[t * d_model ..][0..d_model]);
                    self.norm2.last_rstd = self.norm2_rstd[t];
                    self.norm2.backward(
                        self.h[t * d_model ..][0..d_model],
                        self.grad_norm2_out[t * d_model ..][0..d_model],
                        &self.scratch,
                    );
                    for (self.grad_h[t * d_model ..][0..d_model], &self.scratch) |*gh, s| gh.* += s;
                }

                self.attn.backward(
                    self.norm1_out[0..smd],
                    self.grad_h[0..smd],
                    self.grad_norm1_out[0..smd],
                );

                @memcpy(grad_in[0..smd], self.grad_h[0..smd]);

                for (0..seq) |t| {
                    @memcpy(&self.norm1.x_norm_buf, self.norm1_xn[t * d_model ..][0..d_model]);
                    self.norm1.last_rstd = self.norm1_rstd[t];
                    self.norm1.backward(
                        input[t * d_model ..][0..d_model],
                        self.grad_norm1_out[t * d_model ..][0..d_model],
                        &self.scratch,
                    );
                    for (grad_in[t * d_model ..][0..d_model], &self.scratch) |*gi, s| gi.* += s;
                }
            }
        }

        //  weight update

        /// Update all submodule weights using the given optimizer.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            if (comptime backend == .metal) {
                const mb = @import("backend/metal.zig");
                const FO = mb.FusedOps;
                const eng = mb.getEngine() catch @panic("Metal init failed");
                eng.waitIfPending();

                eng.beginRecording();

                switch (opt.kind) {
                    .adam => |cfg_| {
                        const base = mb.AdamParams{ .length = 0, .lr = lr, .beta1 = cfg_.beta1, .beta2 = cfg_.beta2, .eps = cfg_.eps, .weight_decay = 0, .t = @intCast(t) };
                        const mk = struct {
                            fn p(b: mb.AdamParams, n: u32) mb.AdamParams {
                                var r = b; r.length = n; return r;
                            }
                        };
                        // norm1 gamma, beta
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.norm1.gamma[0..d_model]), eng.getOrUpload(self.norm1.grad_gamma[0..d_model]), eng.getOrUploadMut(self.norm1.m_gamma[0..d_model]), eng.getOrUploadMut(self.norm1.v_gamma[0..d_model]), mk.p(base, d_model));
                        if (comptime cfg.norm == .layer_norm) FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.norm1.beta[0..d_model]), eng.getOrUpload(self.norm1.grad_beta[0..d_model]), eng.getOrUploadMut(self.norm1.m_beta[0..d_model]), eng.getOrUploadMut(self.norm1.v_beta[0..d_model]), mk.p(base, d_model));
                        // norm2 gamma, beta
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.norm2.gamma[0..d_model]), eng.getOrUpload(self.norm2.grad_gamma[0..d_model]), eng.getOrUploadMut(self.norm2.m_gamma[0..d_model]), eng.getOrUploadMut(self.norm2.v_gamma[0..d_model]), mk.p(base, d_model));
                        if (comptime cfg.norm == .layer_norm) FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.norm2.beta[0..d_model]), eng.getOrUpload(self.norm2.grad_beta[0..d_model]), eng.getOrUploadMut(self.norm2.m_beta[0..d_model]), eng.getOrUploadMut(self.norm2.v_beta[0..d_model]), mk.p(base, d_model));
                        // attn wq, wk, wv, wo
                        const dm2: u32 = @intCast(d_model * d_model);
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(&self.attn.wq_f), eng.getOrUpload(&self.attn.grad_wq), eng.getOrUploadMut(&self.attn.m_wq), eng.getOrUploadMut(&self.attn.v_wq), mk.p(base, dm2));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(&self.attn.wk_f), eng.getOrUpload(&self.attn.grad_wk), eng.getOrUploadMut(&self.attn.m_wk), eng.getOrUploadMut(&self.attn.v_wk), mk.p(base, dm2));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(&self.attn.wv_f), eng.getOrUpload(&self.attn.grad_wv), eng.getOrUploadMut(&self.attn.m_wv), eng.getOrUploadMut(&self.attn.v_wv), mk.p(base, dm2));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(&self.attn.wo_f), eng.getOrUpload(&self.attn.grad_wo), eng.getOrUploadMut(&self.attn.m_wo), eng.getOrUploadMut(&self.attn.v_wo), mk.p(base, dm2));
                        // ffn1 weights, biases
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.ffn1.weights_compute_buffer), eng.getOrUpload(self.ffn1.grad_w), eng.getOrUploadMut(self.ffn1.m_w), eng.getOrUploadMut(self.ffn1.v_w), mk.p(base, @intCast(d_model * d_ff)));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.ffn1.biases), eng.getOrUpload(self.ffn1.grad_b), eng.getOrUploadMut(self.ffn1.m_b), eng.getOrUploadMut(self.ffn1.v_b), mk.p(base, @intCast(d_ff)));
                        // ffn2 weights, biases
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.ffn2.weights_compute_buffer), eng.getOrUpload(self.ffn2.grad_w), eng.getOrUploadMut(self.ffn2.m_w), eng.getOrUploadMut(self.ffn2.v_w), mk.p(base, @intCast(d_ff * d_model)));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.ffn2.biases), eng.getOrUpload(self.ffn2.grad_b), eng.getOrUploadMut(self.ffn2.m_b), eng.getOrUploadMut(self.ffn2.v_b), mk.p(base, @intCast(d_model)));
                    },
                    .adamw => |cfg_| {
                        const base = mb.AdamParams{ .length = 0, .lr = lr, .beta1 = cfg_.beta1, .beta2 = cfg_.beta2, .eps = cfg_.eps, .weight_decay = cfg_.weight_decay, .t = @intCast(t) };
                        const mk = struct {
                            fn p(b: mb.AdamParams, n: u32) mb.AdamParams { var r = b; r.length = n; return r; }
                        };
                        const dm2: u32 = @intCast(d_model * d_model);
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.norm1.gamma[0..d_model]), eng.getOrUpload(self.norm1.grad_gamma[0..d_model]), eng.getOrUploadMut(self.norm1.m_gamma[0..d_model]), eng.getOrUploadMut(self.norm1.v_gamma[0..d_model]), mk.p(base, d_model));
                        if (comptime cfg.norm == .layer_norm) FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.norm1.beta[0..d_model]), eng.getOrUpload(self.norm1.grad_beta[0..d_model]), eng.getOrUploadMut(self.norm1.m_beta[0..d_model]), eng.getOrUploadMut(self.norm1.v_beta[0..d_model]), mk.p(base, d_model));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.norm2.gamma[0..d_model]), eng.getOrUpload(self.norm2.grad_gamma[0..d_model]), eng.getOrUploadMut(self.norm2.m_gamma[0..d_model]), eng.getOrUploadMut(self.norm2.v_gamma[0..d_model]), mk.p(base, d_model));
                        if (comptime cfg.norm == .layer_norm) FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.norm2.beta[0..d_model]), eng.getOrUpload(self.norm2.grad_beta[0..d_model]), eng.getOrUploadMut(self.norm2.m_beta[0..d_model]), eng.getOrUploadMut(self.norm2.v_beta[0..d_model]), mk.p(base, d_model));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(&self.attn.wq_f), eng.getOrUpload(&self.attn.grad_wq), eng.getOrUploadMut(&self.attn.m_wq), eng.getOrUploadMut(&self.attn.v_wq), mk.p(base, dm2));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(&self.attn.wk_f), eng.getOrUpload(&self.attn.grad_wk), eng.getOrUploadMut(&self.attn.m_wk), eng.getOrUploadMut(&self.attn.v_wk), mk.p(base, dm2));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(&self.attn.wv_f), eng.getOrUpload(&self.attn.grad_wv), eng.getOrUploadMut(&self.attn.m_wv), eng.getOrUploadMut(&self.attn.v_wv), mk.p(base, dm2));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(&self.attn.wo_f), eng.getOrUpload(&self.attn.grad_wo), eng.getOrUploadMut(&self.attn.m_wo), eng.getOrUploadMut(&self.attn.v_wo), mk.p(base, dm2));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.ffn1.weights_compute_buffer), eng.getOrUpload(self.ffn1.grad_w), eng.getOrUploadMut(self.ffn1.m_w), eng.getOrUploadMut(self.ffn1.v_w), mk.p(base, @intCast(d_model * d_ff)));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.ffn1.biases), eng.getOrUpload(self.ffn1.grad_b), eng.getOrUploadMut(self.ffn1.m_b), eng.getOrUploadMut(self.ffn1.v_b), mk.p(base, @intCast(d_ff)));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.ffn2.weights_compute_buffer), eng.getOrUpload(self.ffn2.grad_w), eng.getOrUploadMut(self.ffn2.m_w), eng.getOrUploadMut(self.ffn2.v_w), mk.p(base, @intCast(d_ff * d_model)));
                        FO.encodeAdamUpdate(eng, eng.getOrUploadMut(self.ffn2.biases), eng.getOrUpload(self.ffn2.grad_b), eng.getOrUploadMut(self.ffn2.m_b), eng.getOrUploadMut(self.ffn2.v_b), mk.p(base, @intCast(d_model)));
                    },
                    .sgd => |_| {
                        // Fall through to default (SGD is rarely used, not worth special casing here)
                        eng.commitAndWait(); // close the recording first
                        self.norm1.updateWeights(opt, lr, t);
                        self.norm2.updateWeights(opt, lr, t);
                        self.attn.updateWeights(opt, lr, t);
                        self.ffn1.updateWeights(opt, lr, t);
                        self.ffn2.updateWeights(opt, lr, t);
                        return;
                    },
                }

                eng.commitAndWait();

                // Download updated params + Adam moments (use getCached — do NOT re-upload stale CPU values)
                if (eng.getCached(self.norm1.gamma[0..d_model])) |buf| eng.downloadTo(buf, self.norm1.gamma[0..d_model]);
                if (eng.getCached(self.norm1.m_gamma[0..d_model])) |buf| eng.downloadTo(buf, self.norm1.m_gamma[0..d_model]);
                if (eng.getCached(self.norm1.v_gamma[0..d_model])) |buf| eng.downloadTo(buf, self.norm1.v_gamma[0..d_model]);
                if (eng.getCached(self.norm2.gamma[0..d_model])) |buf| eng.downloadTo(buf, self.norm2.gamma[0..d_model]);
                if (eng.getCached(self.norm2.m_gamma[0..d_model])) |buf| eng.downloadTo(buf, self.norm2.m_gamma[0..d_model]);
                if (eng.getCached(self.norm2.v_gamma[0..d_model])) |buf| eng.downloadTo(buf, self.norm2.v_gamma[0..d_model]);
                if (comptime cfg.norm == .layer_norm) {
                    if (eng.getCached(self.norm1.beta[0..d_model])) |buf| eng.downloadTo(buf, self.norm1.beta[0..d_model]);
                    if (eng.getCached(self.norm1.m_beta[0..d_model])) |buf| eng.downloadTo(buf, self.norm1.m_beta[0..d_model]);
                    if (eng.getCached(self.norm1.v_beta[0..d_model])) |buf| eng.downloadTo(buf, self.norm1.v_beta[0..d_model]);
                    if (eng.getCached(self.norm2.beta[0..d_model])) |buf| eng.downloadTo(buf, self.norm2.beta[0..d_model]);
                    if (eng.getCached(self.norm2.m_beta[0..d_model])) |buf| eng.downloadTo(buf, self.norm2.m_beta[0..d_model]);
                    if (eng.getCached(self.norm2.v_beta[0..d_model])) |buf| eng.downloadTo(buf, self.norm2.v_beta[0..d_model]);
                }
                if (eng.getCached(&self.attn.wq_f)) |buf| eng.downloadTo(buf, &self.attn.wq_f);
                if (eng.getCached(&self.attn.m_wq)) |buf| eng.downloadTo(buf, &self.attn.m_wq);
                if (eng.getCached(&self.attn.v_wq)) |buf| eng.downloadTo(buf, &self.attn.v_wq);
                if (eng.getCached(&self.attn.wk_f)) |buf| eng.downloadTo(buf, &self.attn.wk_f);
                if (eng.getCached(&self.attn.m_wk)) |buf| eng.downloadTo(buf, &self.attn.m_wk);
                if (eng.getCached(&self.attn.v_wk)) |buf| eng.downloadTo(buf, &self.attn.v_wk);
                if (eng.getCached(&self.attn.wv_f)) |buf| eng.downloadTo(buf, &self.attn.wv_f);
                if (eng.getCached(&self.attn.m_wv)) |buf| eng.downloadTo(buf, &self.attn.m_wv);
                if (eng.getCached(&self.attn.v_wv)) |buf| eng.downloadTo(buf, &self.attn.v_wv);
                if (eng.getCached(&self.attn.wo_f)) |buf| eng.downloadTo(buf, &self.attn.wo_f);
                if (eng.getCached(&self.attn.m_wo)) |buf| eng.downloadTo(buf, &self.attn.m_wo);
                if (eng.getCached(&self.attn.v_wo)) |buf| eng.downloadTo(buf, &self.attn.v_wo);
                if (eng.getCached(self.ffn1.weights_compute_buffer)) |buf| eng.downloadTo(buf, self.ffn1.weights_compute_buffer);
                if (eng.getCached(self.ffn1.m_w)) |buf| eng.downloadTo(buf, self.ffn1.m_w);
                if (eng.getCached(self.ffn1.v_w)) |buf| eng.downloadTo(buf, self.ffn1.v_w);
                if (eng.getCached(self.ffn1.biases)) |buf| eng.downloadTo(buf, self.ffn1.biases);
                if (eng.getCached(self.ffn1.m_b)) |buf| eng.downloadTo(buf, self.ffn1.m_b);
                if (eng.getCached(self.ffn1.v_b)) |buf| eng.downloadTo(buf, self.ffn1.v_b);
                if (eng.getCached(self.ffn2.weights_compute_buffer)) |buf| eng.downloadTo(buf, self.ffn2.weights_compute_buffer);
                if (eng.getCached(self.ffn2.m_w)) |buf| eng.downloadTo(buf, self.ffn2.m_w);
                if (eng.getCached(self.ffn2.v_w)) |buf| eng.downloadTo(buf, self.ffn2.v_w);
                if (eng.getCached(self.ffn2.biases)) |buf| eng.downloadTo(buf, self.ffn2.biases);
                if (eng.getCached(self.ffn2.m_b)) |buf| eng.downloadTo(buf, self.ffn2.m_b);
                if (eng.getCached(self.ffn2.v_b)) |buf| eng.downloadTo(buf, self.ffn2.v_b);

                // Zero gradients on CPU
                @memset(&self.attn.grad_wq, 0); @memset(&self.attn.grad_wk, 0);
                @memset(&self.attn.grad_wv, 0); @memset(&self.attn.grad_wo, 0);
                @memset(self.ffn1.grad_w, 0); @memset(self.ffn1.grad_b, 0);
                @memset(self.ffn2.grad_w, 0); @memset(self.ffn2.grad_b, 0);
                @memset(self.norm1.grad_gamma[0..d_model], 0);
                @memset(self.norm2.grad_gamma[0..d_model], 0);
                if (comptime cfg.norm == .layer_norm) {
                    @memset(self.norm1.grad_beta[0..d_model], 0);
                    @memset(self.norm2.grad_beta[0..d_model], 0);
                }

                // f32 -> f16 sync for attention
                for (&self.attn.wq, self.attn.wq_f) |*w, wf| w.* = @floatCast(wf);
                for (&self.attn.wk, self.attn.wk_f) |*w, wf| w.* = @floatCast(wf);
                for (&self.attn.wv, self.attn.wv_f) |*w, wf| w.* = @floatCast(wf);
                for (&self.attn.wo, self.attn.wo_f) |*w, wf| w.* = @floatCast(wf);
                // f32 -> f16 for layer weights
                for (self.ffn1.weights, self.ffn1.weights_compute_buffer) |*w, wf| w.* = @floatCast(wf);
                for (self.ffn2.weights, self.ffn2.weights_compute_buffer) |*w, wf| w.* = @floatCast(wf);

                return;
            }

            // CPU path (existing):
            self.norm1.updateWeights(opt, lr, t);
            self.norm2.updateWeights(opt, lr, t);
            self.attn.updateWeights(opt, lr, t);
            self.ffn1.updateWeights(opt, lr, t);
            self.ffn2.updateWeights(opt, lr, t);
        }

        /// Sum of squared gradients across all submodules.
        pub fn gradNormSq(self: *const Self) f32 {
            return self.norm1.gradNormSq()
                 + self.norm2.gradNormSq()
                 + self.attn.gradNormSq()
                 + self.ffn1.gradNormSq()
                 + self.ffn2.gradNormSq();
        }

        /// Scale gradients of all submodules by s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            self.norm1.scaleGrads(s);
            self.norm2.scaleGrads(s);
            self.attn.scaleGrads(s);
            self.ffn1.scaleGrads(s);
            self.ffn2.scaleGrads(s);
        }
    };
}
