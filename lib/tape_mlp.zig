const std         = @import("std");
const backend_mod = @import("backend.zig");
const Backend     = backend_mod.Backend;
const autograd    = @import("autograd.zig");
const Tensor      = autograd.Tensor;
const Activation  = @import("activation.zig").Activation;
const Optimizer   = @import("optimizer.zig").Optimizer;

pub const TapeLayerConfig = struct {
    n_out:      usize,
    activation: Activation,
};

/// Fully-connected MLP of arbitrary depth whose gradients are computed
/// automatically by Tape (reverse-mode AD).
///
/// `n_in`   — input dimension
/// `layers` — comptime slice of TapeLayerConfig: one entry per layer
///
/// Example (2 → 16 → 8 → 1):
///   const Model = TapeMLP(.cpu, 2, &.{
///       .{ .n_out = 16, .activation = .relu },
///       .{ .n_out = 8,  .activation = .relu },
///       .{ .n_out = 1,  .activation = .sigmoid },
///   });
///   const Net = Network(*Model);
///   var model = try Model.init(allocator);
pub fn TapeMLP(
    comptime backend: Backend,
    comptime n_in:    usize,
    comptime layers:  []const TapeLayerConfig,
) type {
    comptime std.debug.assert(layers.len >= 1);

    const N        = layers.len;
    const TapeType = autograd.Tape(backend);
    const OptImpl  = backend_mod.OptimizerImpl(backend);

    // Precompute the width of each layer boundary: sizes[i] → sizes[i+1]
    const sizes: [N + 1]usize = comptime blk: {
        var s: [N + 1]usize = undefined;
        s[0] = n_in;
        for (0..N) |i| s[i + 1] = layers[i].n_out;
        break :blk s;
    };

    return struct {
        pub const backend_tag = backend;
        tape: TapeType,

        weights: [N]*Tensor,
        biases:  [N]*Tensor,
        m_w: [N][]f32, v_w: [N][]f32,
        m_b: [N][]f32, v_b: [N][]f32,

        x_t:      *Tensor, // pre-alloc input leaf; data overwritten each forward
        last_out: *Tensor, // tape-owned; valid between forward() and backward()

        alloc: std.mem.Allocator,

        const Self = @This();

        pub fn init(alloc: std.mem.Allocator) !Self {
            var rng  = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
            const rand = rng.random();

            // --- parameters -----------------------------------------------
            var weights: [N]*Tensor = undefined;
            var wi: usize = 0;
            errdefer for (0..wi) |i| weights[i].deinit(alloc);

            var biases: [N]*Tensor = undefined;
            var bi: usize = 0;
            errdefer for (0..bi) |i| biases[i].deinit(alloc);

            inline for (0..N) |i| {
                const ni = sizes[i];
                const no = sizes[i + 1];
                weights[i] = try Tensor.createLeaf(alloc, ni * no, true);
                wi += 1;
                biases[i] = try Tensor.createLeaf(alloc, no, true);
                bi += 1;
                // He init: scale = sqrt(2 / n_in_of_layer)
                const scale = @sqrt(2.0 / @as(f32, ni));
                for (weights[i].data) |*p| p.* = rand.floatNorm(f32) * scale;
            }

            //  input leaf 
            const x_t = try Tensor.createLeaf(alloc, n_in, false);
            errdefer x_t.deinit(alloc);

            //  moment buffers 
            var m_w: [N][]f32 = undefined;
            var v_w: [N][]f32 = undefined;
            var m_b: [N][]f32 = undefined;
            var v_b: [N][]f32 = undefined;
            var mi: usize = 0;
            errdefer for (0..mi) |i| {
                alloc.free(m_w[i]); alloc.free(v_w[i]);
                alloc.free(m_b[i]); alloc.free(v_b[i]);
            };

            inline for (0..N) |i| {
                const ni = sizes[i];
                const no = sizes[i + 1];
                m_w[i] = try alloc.alloc(f32, ni * no); @memset(m_w[i], 0);
                v_w[i] = try alloc.alloc(f32, ni * no); @memset(v_w[i], 0);
                m_b[i] = try alloc.alloc(f32, no);      @memset(m_b[i], 0);
                v_b[i] = try alloc.alloc(f32, no);      @memset(v_b[i], 0);
                mi += 1;
            }

            return .{
                .tape     = TapeType.init(alloc),
                .weights  = weights,
                .biases   = biases,
                .m_w = m_w, .v_w = v_w,
                .m_b = m_b, .v_b = v_b,
                .x_t      = x_t,
                .last_out = undefined,
                .alloc    = alloc,
            };
        }

        pub fn deinit(self: *Self) void {
            self.tape.deinit();
            for (0..N) |i| {
                self.weights[i].deinit(self.alloc);
                self.biases[i].deinit(self.alloc);
                self.alloc.free(self.m_w[i]); self.alloc.free(self.v_w[i]);
                self.alloc.free(self.m_b[i]); self.alloc.free(self.v_b[i]);
            }
            self.x_t.deinit(self.alloc);
        }

        pub fn forward(self: *Self, input: []const f32, output: []f32) void {
            @memcpy(self.x_t.data, input);
            @memset(self.x_t.grad, 0);

            var cur: *Tensor = self.x_t;
            inline for (0..N) |i| {
                const ni = sizes[i];
                const no = sizes[i + 1];
                const h  = self.tape.matmul(cur, self.weights[i], 1, ni, no) catch @panic("OOM");
                const hb = self.tape.add(h, self.biases[i])                   catch @panic("OOM");
                cur = applyActivation(&self.tape, hb, layers[i].activation);
            }

            self.last_out = cur;
            @memcpy(output, cur.data);
        }

        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) void {
            _ = input;
            @memcpy(self.last_out.grad, grad_out);
            self.tape.backwardFrom(self.last_out);
            self.tape.reset();
            @memset(grad_in, 0);
        }

        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            inline for (0..N) |i| {
                OptImpl.update(opt, t, lr, self.weights[i].data, self.weights[i].grad, self.m_w[i], self.v_w[i]);
                OptImpl.update(opt, t, lr, self.biases[i].data,  self.biases[i].grad,  self.m_b[i], self.v_b[i]);
            }
        }

        pub fn gradNormSq(self: *const Self) f32 {
            var sum: f32 = 0;
            for (0..N) |i| {
                for (self.weights[i].grad) |g| sum += g * g;
                for (self.biases[i].grad)  |g| sum += g * g;
            }
            return sum;
        }

        pub fn scaleGrads(self: *Self, s: f32) void {
            for (0..N) |i| {
                for (self.weights[i].grad) |*g| g.* *= s;
                for (self.biases[i].grad)  |*g| g.* *= s;
            }
        }
    };
}

//  activation dispatch 

fn applyActivation(
    tape: anytype,
    t:    *Tensor,
    comptime act: Activation,
) *Tensor {
    return switch (act) {
        .relu      => tape.relu(t)    catch @panic("OOM"),
        .sigmoid   => tape.sigmoid(t) catch @panic("OOM"),
        .tanh      => tape.tanhOp(t)  catch @panic("OOM"),
        .linear    => t,
        .silu      => tape.silu(t)    catch @panic("OOM"),
        .gelu      => tape.gelu(t)    catch @panic("OOM"),
        .leaky_relu => @compileError("leaky_relu not yet supported in TapeMLP"),
        .elu        => @compileError("elu not yet supported in TapeMLP"),
    };
}
