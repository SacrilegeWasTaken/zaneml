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

        last_seq:  usize,
        allocator: std.mem.Allocator,

        const Self = @This();

        //  init / deinit 

        /// Allocate and initialize a TransformerBlock on the heap.
        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            self.allocator = allocator;
            self.last_seq  = 0;

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
                const FO = mb.FusedOps;
                const eng = mb.getEngine() catch @panic("Metal init failed");
                eng.waitIfPending();

                // Sync and upload all weights once
                self.ffn1.syncWeights();
                self.ffn2.syncWeights();
                self.attn.syncWeightsF32();

                const buf_input  = eng.getOrUpload(input);
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

                eng.beginRecording();

                // Norm1
                const n1 = switch (cfg.norm) {
                    .layer_norm => FO.encodeLayernormFwd(eng, buf_input, buf_gamma1, eng.getOrUpload(self.norm1.beta[0..d_model]), self.norm1.eps, seq, d_model, 100),
                    .rms_norm => FO.encodeRmsnormFwd(eng, buf_input, buf_gamma1, self.norm1.eps, seq, d_model, 100),
                };

                // Attention
                const attn_r = FO.encodeAttnFwd(eng, d_model, n_heads, d_model / n_heads,
                    Mha.attn_scale_val, seq, cfg.causal,
                    n1.out, buf_wq, buf_wk, buf_wv, buf_wo, 103);

                // Residual: h = attn_out + input
                const buf_h = FO.encodeAdd(eng, attn_r.output, buf_input, smd, 109);

                // Norm2
                const n2 = switch (cfg.norm) {
                    .layer_norm => FO.encodeLayernormFwd(eng, buf_h, buf_gamma2, eng.getOrUpload(self.norm2.beta[0..d_model]), self.norm2.eps, seq, d_model, 110),
                    .rms_norm => FO.encodeRmsnormFwd(eng, buf_h, buf_gamma2, self.norm2.eps, seq, d_model, 110),
                };

                // FFN1
                const f1 = FO.encodeFFNFwd(eng, n2.out, buf_f1w, buf_f1b, cfg.ffn_activation, d_model, d_ff, seq, 113);

                // FFN2
                const f2 = FO.encodeFFNFwd(eng, f1.output, buf_f2w, buf_f2b, .linear, d_ff, d_model, seq, 116);

                // Residual: output = ffn2_out + h
                const buf_final = FO.encodeAdd(eng, f2.output, buf_h, smd, 119);

                // Single GPU submission for entire forward
                eng.commitAndWait();

                // Download caches needed for backward
                eng.downloadTo(n1.x_norm, self.norm1_xn[0..smd]);
                eng.downloadTo(n1.rstd, self.norm1_rstd[0..seq]);
                eng.downloadTo(attn_r.q, self.attn.cache_q[0..smd]);
                eng.downloadTo(attn_r.k, self.attn.cache_k[0..smd]);
                eng.downloadTo(attn_r.v, self.attn.cache_v[0..smd]);
                eng.downloadTo(attn_r.scores, self.attn.cache_attn[0..n_heads * seq * seq]);
                eng.downloadTo(attn_r.ctx, self.attn.cache_concat[0..smd]);
                eng.downloadTo(buf_h, self.h[0..smd]);
                eng.downloadTo(n1.out, self.norm1_out[0..smd]);
                eng.downloadTo(n2.x_norm, self.norm2_xn[0..smd]);
                eng.downloadTo(n2.rstd, self.norm2_rstd[0..seq]);
                eng.downloadTo(n2.out, self.norm2_out[0..smd]);
                eng.downloadTo(f1.pre_act, self.ffn1_pre[0..seq * d_ff]);
                eng.downloadTo(f1.output, self.ffn1_out[0..seq * d_ff]);
                eng.downloadTo(f2.pre_act, self.ffn2_pre[0..smd]);
                eng.downloadTo(buf_final, output[0..smd]);
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
                const FO = mb.FusedOps;
                const eng = mb.getEngine() catch @panic("Metal init failed");
                eng.waitIfPending();

                // Upload all data needed for backward
                const buf_go = eng.getOrUpload(grad_out[0..smd]);
                const buf_n1_out = eng.getOrUpload(self.norm1_out[0..smd]);
                const buf_n1_xn = eng.getOrUpload(self.norm1_xn[0..smd]);
                const buf_n1_rstd = eng.getOrUpload(self.norm1_rstd[0..seq]);
                const buf_n2_xn = eng.getOrUpload(self.norm2_xn[0..smd]);
                const buf_n2_rstd = eng.getOrUpload(self.norm2_rstd[0..seq]);
                const buf_n2_out = eng.getOrUpload(self.norm2_out[0..smd]);
                const buf_f1_pre = eng.getOrUpload(self.ffn1_pre[0..seq * d_ff]);
                const buf_f1_out = eng.getOrUpload(self.ffn1_out[0..seq * d_ff]);
                const buf_f2_pre = eng.getOrUpload(self.ffn2_pre[0..smd]);

                const buf_gamma1 = eng.getOrUpload(self.norm1.gamma[0..d_model]);
                const buf_gamma2 = eng.getOrUpload(self.norm2.gamma[0..d_model]);
                const buf_gg1 = eng.getOrUploadMut(self.norm1.grad_gamma[0..d_model]);
                const buf_gg2 = eng.getOrUploadMut(self.norm2.grad_gamma[0..d_model]);

                const buf_wq = eng.getOrUpload(&self.attn.wq_f);
                const buf_wk = eng.getOrUpload(&self.attn.wk_f);
                const buf_wv = eng.getOrUpload(&self.attn.wv_f);
                const buf_wo = eng.getOrUpload(&self.attn.wo_f);
                const buf_cq = eng.getOrUpload(self.attn.cache_q[0..smd]);
                const buf_ck = eng.getOrUpload(self.attn.cache_k[0..smd]);
                const buf_cv = eng.getOrUpload(self.attn.cache_v[0..smd]);
                const buf_ca = eng.getOrUpload(self.attn.cache_attn[0..n_heads * seq * seq]);
                const buf_cc = eng.getOrUpload(self.attn.cache_concat[0..smd]);
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

                eng.beginRecording();

                // FFN2 backward
                const f2_bwd = FO.encodeFFNBwd(eng, buf_f2_pre, buf_go, buf_f1_out, buf_f2w,
                    buf_f2_gw, buf_f2_gb, .linear, d_ff, d_model, seq, 200);

                // FFN1 backward
                const f1_bwd = FO.encodeFFNBwd(eng, buf_f1_pre, f2_bwd.grad_in, buf_n2_out, buf_f1w,
                    buf_f1_gw, buf_f1_gb, cfg.ffn_activation, d_model, d_ff, seq, 203);

                // Norm2 backward
                const buf_n2_gi = switch (cfg.norm) {
                    .layer_norm => FO.encodeLayernormBwd(eng, f1_bwd.grad_in, buf_n2_xn,
                        buf_gamma2, buf_n2_rstd, buf_gg2, eng.getOrUploadMut(self.norm2.grad_beta[0..d_model]), seq, d_model, 206),
                    .rms_norm => FO.encodeRmsnormBwd(eng, f1_bwd.grad_in, buf_n2_xn,
                        buf_gamma2, buf_n2_rstd, buf_gg2, seq, d_model, 206),
                };

                // Residual: grad_h = grad_out + norm2_grad_in
                const buf_grad_h = FO.encodeAdd(eng, buf_go, buf_n2_gi, smd, 207);

                // Attention backward
                const buf_attn_gi = FO.encodeAttnBwd(eng, d_model, n_heads, d_model / n_heads,
                    Mha.attn_scale_val, seq,
                    buf_n1_out, buf_grad_h,
                    buf_wq, buf_wk, buf_wv, buf_wo,
                    buf_cq, buf_ck, buf_cv, buf_ca, buf_cc,
                    buf_gwq, buf_gwk, buf_gwv, buf_gwo,
                    208);

                // Norm1 backward
                const buf_n1_gi = switch (cfg.norm) {
                    .layer_norm => FO.encodeLayernormBwd(eng, buf_attn_gi, buf_n1_xn,
                        buf_gamma1, buf_n1_rstd, buf_gg1, eng.getOrUploadMut(self.norm1.grad_beta[0..d_model]), seq, d_model, 220),
                    .rms_norm => FO.encodeRmsnormBwd(eng, buf_attn_gi, buf_n1_xn,
                        buf_gamma1, buf_n1_rstd, buf_gg1, seq, d_model, 220),
                };

                // Final residual: grad_in = norm1_grad_in + grad_h
                const buf_final_gi = FO.encodeAdd(eng, buf_n1_gi, buf_grad_h, smd, 221);

                eng.commitAndWait();

                // Download weight gradients
                eng.downloadTo(buf_f2_gw, self.ffn2.grad_w);
                eng.downloadTo(buf_f2_gb, self.ffn2.grad_b);
                eng.downloadTo(buf_f1_gw, self.ffn1.grad_w);
                eng.downloadTo(buf_f1_gb, self.ffn1.grad_b);
                eng.downloadTo(buf_gg1, self.norm1.grad_gamma[0..d_model]);
                eng.downloadTo(buf_gg2, self.norm2.grad_gamma[0..d_model]);
                if (comptime cfg.norm == .layer_norm) {
                    // Beta grad buffers were uploaded inline; read them back via getOrUploadMut cache
                    eng.downloadTo(eng.getOrUploadMut(self.norm1.grad_beta[0..d_model]), self.norm1.grad_beta[0..d_model]);
                    eng.downloadTo(eng.getOrUploadMut(self.norm2.grad_beta[0..d_model]), self.norm2.grad_beta[0..d_model]);
                }
                eng.downloadTo(buf_gwo, &self.attn.grad_wo);
                eng.downloadTo(buf_gwq, &self.attn.grad_wq);
                eng.downloadTo(buf_gwk, &self.attn.grad_wk);
                eng.downloadTo(buf_gwv, &self.attn.grad_wv);
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
