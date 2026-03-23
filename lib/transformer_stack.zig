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
        pub const backend_tag = backend;
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

        const Self = @This();

        //  init / deinit

        /// Allocate the stack and all its blocks on the heap.
        pub fn init(allocator: std.mem.Allocator) !*Self {
            const self = try allocator.create(Self);
            errdefer allocator.destroy(self);

            self.allocator = allocator;
            self.last_seq  = 0;
            self.pos_embed = PE.init();

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
                    buf_cur = self.blocks[i].encodeForward(eng, buf_cur, seq);
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
