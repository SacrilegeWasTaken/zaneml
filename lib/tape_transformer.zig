const std         = @import("std");
const autograd    = @import("autograd.zig");
const Tensor      = autograd.Tensor;
const stack_mod   = @import("transformer_stack.zig");
const backend_mod = @import("backend.zig");
const Backend     = backend_mod.Backend;
const Optimizer   = @import("optimizer.zig").Optimizer;

pub const BlockConfig = @import("transformer_block.zig").BlockConfig;

/// TransformerStack wrapped as a tape-compatible module.
///
/// Drop-in replacement for TransformerStack when you need autograd composability:
/// mix transformer layers with custom tape ops (embeddings, projections, losses).
///
/// GPU performance is identical to TransformerStack — forward and backward passes
/// still execute in the same fused Metal command buffers; the tape records only
/// one op per stack, not one per layer.
///
/// Works with Network(*TapeTransformerStack): implements forward/backward/updateWeights
/// and batchForward/batchBackward for the Metal batch-stacking fast path.
pub fn TapeTransformerStack(
    comptime backend:  Backend,
    comptime d_model:  usize,
    comptime n_heads:  usize,
    comptime d_ff:     usize,
    comptime max_seq:  usize,
    comptime n_layers: usize,
    comptime cfg:      BlockConfig,
) type {
    const StackT = stack_mod.TransformerStack(backend, d_model, n_heads, d_ff, max_seq, n_layers, cfg);
    const TapeT  = autograd.Tape(backend);

    return struct {
        /// Exposed for Network's Metal batch-path detection.
        pub const backend_tag = backend;
        pub const d_model_val = d_model;

        stack:    *StackT,
        tape:     TapeT,
        last_out: *Tensor,   // set in forward, valid until tape.reset() in backward
        alloc:    std.mem.Allocator,

        const Self = @This();

        pub fn init(alloc: std.mem.Allocator) !*Self {
            const self = try alloc.create(Self);
            errdefer alloc.destroy(self);
            self.alloc    = alloc;
            self.stack    = try StackT.init(alloc);
            errdefer self.stack.deinit();
            self.tape     = TapeT.init(alloc);
            self.last_out = undefined;
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.stack.deinit();
            self.tape.deinit();
            self.alloc.destroy(self);
        }

        // ── Network interface ──────────────────────────────────────────────

        /// Forward pass: records the transformer stack as a single tape op.
        /// On Metal the underlying stack still runs all layers in one command buffer.
        pub fn forward(self: *Self, input: []const f32, output: []f32) void {
            // inputFrom creates an arena-owned no-grad tensor that is freed on tape.reset().
            const x   = self.tape.inputFrom(input)                       catch @panic("OOM");
            const out = self.tape.applyModule(self.stack, x, input.len)  catch @panic("OOM");
            self.last_out = out;
            @memcpy(output, out.data);
        }

        /// Backward pass: replays the tape (calls stack.backward) then resets the arena.
        /// last_out is arena-owned and becomes invalid after reset(); we set it to undefined.
        pub fn backward(self: *Self, _: []const f32, grad_out: []const f32, grad_in: []f32) void {
            @memcpy(self.last_out.grad, grad_out);
            self.tape.backwardFrom(self.last_out);
            self.last_out = undefined;   // arena is about to be freed; prevent stale access
            self.tape.reset();
            @memset(grad_in, 0);
        }

        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            self.stack.updateWeights(opt, lr, t);
        }

        pub fn gradNormSq(self: *const Self) f32 { return self.stack.gradNormSq(); }
        pub fn scaleGrads(self: *Self, s: f32)  void { self.stack.scaleGrads(s); }

        // ── Metal batch path ───────────────────────────────────────────────
        // Delegated directly to the inner stack so the full batchForward/batchBackward
        // optimisation (single command buffer per pass) is preserved.

        pub fn batchForward(self: *Self, si: []const f32, so: []f32, n: usize, sp: usize) void {
            self.stack.batchForward(si, so, n, sp);
        }

        pub fn batchBackward(self: *Self, go: []const f32, gi: []f32, n: usize, sp: usize) void {
            self.stack.batchBackward(go, gi, n, sp);
        }
    };
}
