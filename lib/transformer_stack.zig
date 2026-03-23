const std = @import("std");

const backend_mod = @import("backend.zig");
const Backend     = backend_mod.Backend;

const BlockFn     = @import("transformer_block.zig").TransformerBlock;
pub const BlockConfig = @import("transformer_block.zig").BlockConfig;
const PEFn        = @import("positional_embedding.zig").PositionalEmbedding;
const Optimizer   = @import("optimizer.zig").Optimizer;

/// Stack of `n_layers` TransformerBlocks with learnable positional embeddings.
///
/// Compatible with Network(*TransformerStack(...)): implements
///   forward(input, output)  /  backward(input, grad_out, grad_in)  /  updateWeights(opt, lr, t)
///
/// input/output: flat slices [seq * d_model], seq inferred from input.len.
///
/// cfg controls norm type, FFN activation, and causal masking for every block.
pub fn TransformerStack(
    comptime backend:  Backend,
    comptime d_model:  usize,
    comptime n_heads:  usize,
    comptime d_ff:     usize,
    comptime max_seq:  usize,
    comptime n_layers: usize,
    comptime cfg:      BlockConfig,
) type {
    comptime std.debug.assert(n_layers >= 1);
    const Block = BlockFn(backend, d_model, n_heads, d_ff, max_seq, cfg);
    const PE    = PEFn(backend, d_model, max_seq);

    return struct {
        pub const backend_tag  = backend;
        pub const d_model_val  = d_model;
        pos_embed: PE,
        blocks:    [n_layers]*Block,

        /// Buffer for input + PE, used in backward as layer_input for block[0].
        pe_buf: [max_seq * d_model]f32,

        /// Intermediate activations and gradients between layers.
        /// When n_layers==1 these arrays have zero length -- no overhead.
        act_bufs:  [n_layers - 1][max_seq * d_model]f32,
        grad_bufs: [n_layers - 1][max_seq * d_model]f32,

        last_seq:  usize,
        allocator: std.mem.Allocator,

        /// Cached tiled PE buffer for batchForward (Metal only).
        /// Lazily allocated on first batch call; reused when batch shape is unchanged.
        tiled_pe:     ?[]f32,
        tiled_pe_len: usize,

        const Self = @This();

        //  init / deinit

        /// Allocate the stack and all its blocks on the heap.
        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.allocator    = allocator;
            self.last_seq     = 0;
            self.tiled_pe     = null;
            self.tiled_pe_len = 0;
            self.pos_embed    = PE.init();

            var n: usize = 0;
            errdefer for (0..n) |i| self.blocks[i].deinit();

            inline for (0..n_layers) |i| {
                self.blocks[i] = try Block.initWithSlots(allocator,
                    @as(u16, 100) + @as(u16, i) * 30,   // fwd_slot_base
                    @as(u16, 300) + @as(u16, i) * 30);  // bwd_slot_base
                n += 1;
            }

            return self;
        }

        /// Free all blocks and the stack struct itself.
        pub fn deinit(self: *Self) void {
            for (&self.blocks) |b| b.deinit();
            if (self.tiled_pe) |buf| self.allocator.free(buf);
            self.allocator.destroy(self);
        }

        //  forward

        /// Forward pass: applies positional embeddings then each block in order.
        pub fn forward(self: *Self, input: []const f32, output: []f32) void {
            const seq = input.len / d_model;
            const smd = seq * d_model;
            self.last_seq = seq;

            if (comptime backend == .metal) {
                const mb = @import("backend/metal.zig");
                const FO = mb.FusedOps;
                const eng = mb.getEngine() catch @panic("Metal init failed");
                eng.waitIfPending();

                // Upload PE embed
                const buf_input = eng.getOrUpload(input);
                const buf_embed = eng.getOrUpload(self.pos_embed.embed[0..smd]);

                eng.beginRecording();
                // PE addition (slot 50)
                var buf_cur = FO.encodeAdd(eng, buf_input, buf_embed, smd, 50);
                // All blocks in one recording
                inline for (0..n_layers) |i| {
                    buf_cur = self.blocks[i].encodeForward(eng, buf_cur, seq, 0);
                }
                eng.commitAndWait();
                eng.downloadTo(buf_cur, output[0..smd]);
                return;
            }

            // CPU path (existing)
            self.pos_embed.forward(input, self.pe_buf[0..smd]);

            inline for (0..n_layers) |i| {
                const cur_in: []const f32 = if (i == 0)
                    self.pe_buf[0..smd]
                else
                    self.act_bufs[i - 1][0..smd];

                const cur_out: []f32 = if (i == n_layers - 1)
                    output
                else
                    self.act_bufs[i][0..smd];

                self.blocks[i].forwardSeq(cur_in, cur_out, seq);
            }
        }

        //  batch forward/backward (Metal only)

        /// Metal-only batch forward: stacks N samples into one GPU pass using block-diagonal
        /// attention masking so each sample attends only to its own tokens.
        ///
        /// stacked_input  -- [n_samples * seq_per * d_model]
        /// stacked_output -- [n_samples * seq_per * d_model]
        pub fn batchForward(self: *Self, stacked_input: []const f32, stacked_output: []f32, n_samples: usize, seq_per: usize) void {
            comptime std.debug.assert(backend == .metal);
            const mb = @import("backend/metal.zig");
            const FO = mb.FusedOps;
            const eng = mb.getEngine() catch @panic("Metal init failed");
            eng.waitIfPending();

            std.debug.assert(n_samples > 0);
            std.debug.assert(seq_per > 0);
            const seq_total = n_samples * seq_per;
            const smd_total = seq_total * d_model;
            const smd_per   = seq_per * d_model;
            std.debug.assert(stacked_input.len == smd_total);
            std.debug.assert(stacked_output.len == smd_total);
            self.last_seq   = seq_total;

            // Build tiled PE: repeat pos_embed[0..smd_per] N times.
            // Reuse cached buffer when shape is unchanged (common in training loops).
            if (self.tiled_pe_len != smd_total) {
                if (self.tiled_pe) |old| self.allocator.free(old);
                self.tiled_pe     = self.allocator.alloc(f32, smd_total) catch @panic("batchForward PE alloc");
                self.tiled_pe_len = smd_total;
            }
            const tiled_pe = self.tiled_pe.?;
            for (0..n_samples) |s| {
                @memcpy(tiled_pe[s * smd_per ..][0..smd_per], self.pos_embed.embed[0..smd_per]);
            }

            const buf_input = eng.getOrUpload(stacked_input);
            const buf_embed = eng.getOrUpload(tiled_pe);

            eng.beginRecording();
            var buf_cur = FO.encodeAdd(eng, buf_input, buf_embed, smd_total, 50);
            inline for (0..n_layers) |i| {
                buf_cur = self.blocks[i].encodeForward(eng, buf_cur, seq_total, seq_per);
            }
            eng.commitAndWait();
            eng.downloadTo(buf_cur, stacked_output[0..smd_total]);
        }

        /// Metal-only batch backward: backward pass over stacked N-sample batch.
        ///
        /// stacked_grad_out -- [n_samples * seq_per * d_model] (loss gradient)
        /// stacked_grad_in  -- [n_samples * seq_per * d_model] (written: grad w.r.t. input)
        pub fn batchBackward(self: *Self, stacked_grad_out: []const f32, stacked_grad_in: []f32, n_samples: usize, seq_per: usize) void {
            comptime std.debug.assert(backend == .metal);
            const mb = @import("backend/metal.zig");
            const eng = mb.getEngine() catch @panic("Metal init failed");
            eng.waitIfPending();

            std.debug.assert(n_samples > 0);
            std.debug.assert(seq_per > 0);
            const seq_total = n_samples * seq_per;
            const smd_total = seq_total * d_model;
            const smd_per   = seq_per * d_model;
            std.debug.assert(stacked_grad_out.len == smd_total);
            std.debug.assert(stacked_grad_in.len  == smd_total);
            std.debug.assert(self.last_seq == seq_total);

            inline for (0..n_layers) |i| {
                self.blocks[i].prepareBackwardUploads(eng, seq_total);
            }

            const buf_go = eng.getOrUpload(stacked_grad_out[0..smd_total]);
            eng.beginRecording();
            var buf_gi = buf_go;
            inline for (0..n_layers) |rev| {
                const i = n_layers - 1 - rev;
                buf_gi = self.blocks[i].encodeBackward(eng, buf_gi, seq_total);
            }
            eng.commitAndWait();

            eng.downloadTo(buf_gi, stacked_grad_in[0..smd_total]);
            inline for (0..n_layers) |i| {
                self.blocks[i].downloadGrads(eng);
            }

            // PE grad: sum tile-wise (each sample contributes to same PE positions)
            for (0..n_samples) |s| {
                for (self.pos_embed.grad_embed[0..smd_per], stacked_grad_in[s * smd_per ..][0..smd_per]) |*ge, gi| ge.* += gi;
            }
        }

        //  backward

        /// Backward pass: propagates gradients through blocks then positional embeddings.
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) !void {
            _ = input;
            const seq = self.last_seq;
            const smd = seq * d_model;

            if (comptime backend == .metal) {
                const mb = @import("backend/metal.zig");
                const eng = mb.getEngine() catch @panic("Metal init failed");
                eng.waitIfPending();

                // Upload gradient accumulators and params for ALL blocks before recording
                inline for (0..n_layers) |i| {
                    self.blocks[i].prepareBackwardUploads(eng, seq);
                }

                const buf_go = eng.getOrUpload(grad_out[0..smd]);
                eng.beginRecording();
                var buf_gi = buf_go;
                // Reverse through all blocks in one recording
                inline for (0..n_layers) |rev| {
                    const i = n_layers - 1 - rev;
                    buf_gi = self.blocks[i].encodeBackward(eng, buf_gi, seq);
                }
                eng.commitAndWait();

                eng.downloadTo(buf_gi, grad_in[0..smd]);
                // Download all gradient accumulators
                inline for (0..n_layers) |i| {
                    self.blocks[i].downloadGrads(eng);
                }

                // PE grad: identity pass-through
                for (self.pos_embed.grad_embed[0..smd], grad_in[0..smd]) |*ge, gi| ge.* += gi;
                return;
            }

            // CPU path (existing)
            inline for (0..n_layers) |rev| {
                const i = n_layers - 1 - rev;

                // layer_input for block[0] is pe_buf (after PE), not the original input
                const layer_input: []const f32 = if (i == 0)
                    self.pe_buf[0..smd]
                else
                    self.act_bufs[i - 1][0..smd];

                const layer_grad_out: []const f32 = if (i == n_layers - 1)
                    grad_out
                else
                    self.grad_bufs[i][0..smd];

                const layer_grad_in: []f32 = if (i == 0)
                    grad_in
                else
                    self.grad_bufs[i - 1][0..smd];

                try self.blocks[i].backward(layer_input, layer_grad_out, layer_grad_in);
            }

            // d(input + pe)/d(input) = 1 => grad_in is already correct.
            // Only accumulate grad_embed.
            for (self.pos_embed.grad_embed[0..smd], grad_in[0..smd]) |*ge, gi| ge.* += gi;
        }

        //  weight update

        /// Update all submodule weights using the given optimizer.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            if (comptime backend == .metal) {
                const mb = @import("backend/metal.zig");
                const eng = mb.getEngine() catch @panic("Metal init failed");
                eng.waitIfPending();
                switch (opt.kind) {
                    .sgd => |_| {
                        self.pos_embed.updateWeights(opt, lr, t);
                        for (&self.blocks) |b| b.updateWeights(opt, lr, t);
                        return;
                    },
                    else => {},
                }
                // Fused: all blocks' Adam in one command buffer
                eng.beginRecording();
                inline for (0..n_layers) |i| {
                    self.blocks[i].encodeAdamAll(eng, opt, lr, t);
                }
                eng.commitAndWait();
                for (&self.blocks) |b| b.zeroGrads();
                self.pos_embed.updateWeights(opt, lr, t);
                return;
            }
            // CPU path
            self.pos_embed.updateWeights(opt, lr, t);
            for (&self.blocks) |b| b.updateWeights(opt, lr, t);
        }

        /// Sum of squared gradients across all submodules.
        pub fn gradNormSq(self: *const Self) f32 {
            var sum: f32 = self.pos_embed.gradNormSq();
            for (&self.blocks) |b| sum += b.gradNormSq();
            return sum;
        }

        /// Scale gradients of all submodules by s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            self.pos_embed.scaleGrads(s);
            for (&self.blocks) |b| b.scaleGrads(s);
        }
    };
}
