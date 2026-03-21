const std = @import("std");
const backend_mod = @import("backend.zig");
const Backend = backend_mod.Backend;

/// A gradient-carrying tensor.
///
/// Leaf tensors (parameters, inputs) are created with createLeaf() and
/// freed by the caller with deinit(). Intermediate tensors produced by
/// Tape ops are owned by the tape's arena and freed on tape.reset().
pub const Tensor = struct {
    data:          []f32,
    grad:          []f32,
    requires_grad: bool,

    /// Create a standalone leaf tensor. Caller must call deinit() when done.
    pub fn createLeaf(allocator: std.mem.Allocator, size: usize, requires_grad: bool) !*Tensor {
        const self = try allocator.create(Tensor);
        self.data = try allocator.alloc(f32, size);
        self.grad = try allocator.alloc(f32, size);
        @memset(self.data, 0);
        @memset(self.grad, 0);
        self.requires_grad = requires_grad;
        return self;
    }

    /// Free a leaf tensor created with createLeaf.
    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        allocator.free(self.grad);
        allocator.destroy(self);
    }

    /// Zero the gradient buffer.
    pub fn zeroGrad(self: *Tensor) void {
        @memset(self.grad, 0);
    }
};

/// Tape records forward computations and supports reverse-mode automatic
/// differentiation via a list of backward closures.
///
/// Usage:
///   var tape = Tape(.cpu).init(allocator);
///   defer tape.deinit();
///
///   const w = try Tensor.createLeaf(allocator, n, true);
///   defer w.deinit(allocator);
///
///   const out  = try tape.matmul(input, w, m, k, n);
///   const loss = try tape.mse(out, target_slice);
///   tape.backward(loss);
///   // w.grad now holds dL/dw
///
///   tape.reset();   // free intermediates, reset for next step
pub fn Tape(comptime backend: Backend) type {
    const MM = backend_mod.MatmulImpl(backend);

    return struct {
        ops:        std.ArrayList(Op),
        arena:      std.heap.ArenaAllocator,
        /// Stored separately so deinit/append can pass it without the arena
        outer_alloc: std.mem.Allocator,

        const Self = @This();

        const Op = struct {
            backward_fn: *const fn (*anyopaque) void,
            ctx:         *anyopaque,
        };

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .ops         = .empty,
                .arena       = std.heap.ArenaAllocator.init(allocator),
                .outer_alloc = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.ops.deinit(self.outer_alloc);
            self.arena.deinit();
        }

        /// Free all intermediates and recorded ops. Call between training steps.
        pub fn reset(self: *Self) void {
            self.ops.clearRetainingCapacity();
            _ = self.arena.reset(.retain_capacity);
        }

        // ── internal: allocate a tape-owned intermediate tensor ────────────

        fn newTensor(self: *Self, size: usize, requires_grad: bool) !*Tensor {
            const alloc = self.arena.allocator();
            const t     = try alloc.create(Tensor);
            t.data          = try alloc.alloc(f32, size);
            t.grad          = try alloc.alloc(f32, size);
            t.requires_grad = requires_grad;
            @memset(t.data, 0);
            @memset(t.grad, 0);
            return t;
        }

        fn recordOp(self: *Self, comptime Ctx: type, ctx: *Ctx,
                    comptime bwd: fn (*anyopaque) void) !void {
            try self.ops.append(self.outer_alloc, .{
                .backward_fn = bwd,
                .ctx         = ctx,
            });
        }

        // ── backward ──────────────────────────────────────────────────────

        /// Set loss.grad = 1 and replay all recorded ops in reverse.
        /// loss must be a scalar tensor (data.len == 1).
        pub fn backward(self: *Self, loss: *Tensor) void {
            std.debug.assert(loss.data.len == 1);
            loss.grad[0] = 1.0;
            var i = self.ops.items.len;
            while (i > 0) {
                i -= 1;
                const op = self.ops.items[i];
                op.backward_fn(op.ctx);
            }
        }

        // ── ops ───────────────────────────────────────────────────────────

        /// out = a + b  (element-wise broadcast not supported; sizes must match)
        pub fn add(self: *Self, a: *Tensor, b: *Tensor) !*Tensor {
            std.debug.assert(a.data.len == b.data.len);
            const rg  = a.requires_grad or b.requires_grad;
            const out = try self.newTensor(a.data.len, rg);
            for (out.data, a.data, b.data) |*o, av, bv| o.* = av + bv;

            if (rg) {
                const Ctx = struct { a: *Tensor, b: *Tensor, out: *Tensor };
                const ctx = try self.arena.allocator().create(Ctx);
                ctx.* = .{ .a = a, .b = b, .out = out };
                try self.recordOp(Ctx, ctx, struct {
                    fn bwd(ptr: *anyopaque) void {
                        const c: *Ctx = @ptrCast(@alignCast(ptr));
                        if (c.a.requires_grad) {
                            for (c.a.grad, c.out.grad) |*ag, og| ag.* += og;
                        }
                        if (c.b.requires_grad) {
                            for (c.b.grad, c.out.grad) |*bg, og| bg.* += og;
                        }
                    }
                }.bwd);
            }
            return out;
        }

        /// out = a * b  (element-wise)
        pub fn mul(self: *Self, a: *Tensor, b: *Tensor) !*Tensor {
            std.debug.assert(a.data.len == b.data.len);
            const rg  = a.requires_grad or b.requires_grad;
            const out = try self.newTensor(a.data.len, rg);
            for (out.data, a.data, b.data) |*o, av, bv| o.* = av * bv;

            if (rg) {
                const Ctx = struct { a: *Tensor, b: *Tensor, out: *Tensor };
                const ctx = try self.arena.allocator().create(Ctx);
                ctx.* = .{ .a = a, .b = b, .out = out };
                try self.recordOp(Ctx, ctx, struct {
                    fn bwd(ptr: *anyopaque) void {
                        const c: *Ctx = @ptrCast(@alignCast(ptr));
                        if (c.a.requires_grad) {
                            for (c.a.grad, c.b.data, c.out.grad) |*ag, bv, og| ag.* += og * bv;
                        }
                        if (c.b.requires_grad) {
                            for (c.b.grad, c.a.data, c.out.grad) |*bg, av, og| bg.* += og * av;
                        }
                    }
                }.bwd);
            }
            return out;
        }

        /// out[m*n] = a[m*k] @ b[k*n]  — dispatched through MatmulImpl(backend)
        pub fn matmul(self: *Self, a: *Tensor, b: *Tensor, m: usize, k: usize, n: usize) !*Tensor {
            std.debug.assert(a.data.len == m * k);
            std.debug.assert(b.data.len == k * n);
            const rg  = a.requires_grad or b.requires_grad;
            const out = try self.newTensor(m * n, rg);
            MM.forward(a.data, b.data, out.data, m, k, n);

            if (rg) {
                const Ctx = struct { a: *Tensor, b: *Tensor, out: *Tensor, m: usize, k: usize, n: usize };
                const ctx = try self.arena.allocator().create(Ctx);
                ctx.* = .{ .a = a, .b = b, .out = out, .m = m, .k = k, .n = n };
                try self.recordOp(Ctx, ctx, struct {
                    fn bwd(ptr: *anyopaque) void {
                        const c: *Ctx = @ptrCast(@alignCast(ptr));
                        if (c.a.requires_grad)
                            MM.backwardA(c.out.grad, c.b.data, c.a.grad, c.m, c.k, c.n);
                        if (c.b.requires_grad)
                            MM.backwardB(c.a.data, c.out.grad, c.b.grad, c.m, c.k, c.n);
                    }
                }.bwd);
            }
            return out;
        }

        /// out = relu(a)
        pub fn relu(self: *Self, a: *Tensor) !*Tensor {
            const out = try self.newTensor(a.data.len, a.requires_grad);
            for (out.data, a.data) |*o, av| o.* = @max(0.0, av);

            if (a.requires_grad) {
                const Ctx = struct { a: *Tensor, out: *Tensor };
                const ctx = try self.arena.allocator().create(Ctx);
                ctx.* = .{ .a = a, .out = out };
                try self.recordOp(Ctx, ctx, struct {
                    fn bwd(ptr: *anyopaque) void {
                        const c: *Ctx = @ptrCast(@alignCast(ptr));
                        for (c.a.grad, c.a.data, c.out.grad) |*ag, av, og|
                            ag.* += if (av > 0.0) og else 0.0;
                    }
                }.bwd);
            }
            return out;
        }

        /// out = sigmoid(a)
        pub fn sigmoid(self: *Self, a: *Tensor) !*Tensor {
            const out = try self.newTensor(a.data.len, a.requires_grad);
            for (out.data, a.data) |*o, av| o.* = 1.0 / (1.0 + @exp(-av));

            if (a.requires_grad) {
                const Ctx = struct { a: *Tensor, out: *Tensor };
                const ctx = try self.arena.allocator().create(Ctx);
                ctx.* = .{ .a = a, .out = out };
                try self.recordOp(Ctx, ctx, struct {
                    fn bwd(ptr: *anyopaque) void {
                        const c: *Ctx = @ptrCast(@alignCast(ptr));
                        // sigmoid'(x) = s*(1-s), s = out
                        for (c.a.grad, c.out.data, c.out.grad) |*ag, ov, og|
                            ag.* += og * ov * (1.0 - ov);
                    }
                }.bwd);
            }
            return out;
        }

        /// out = tanh(a)
        pub fn tanhOp(self: *Self, a: *Tensor) !*Tensor {
            const out = try self.newTensor(a.data.len, a.requires_grad);
            for (out.data, a.data) |*o, av| o.* = std.math.tanh(av);

            if (a.requires_grad) {
                const Ctx = struct { a: *Tensor, out: *Tensor };
                const ctx = try self.arena.allocator().create(Ctx);
                ctx.* = .{ .a = a, .out = out };
                try self.recordOp(Ctx, ctx, struct {
                    fn bwd(ptr: *anyopaque) void {
                        const c: *Ctx = @ptrCast(@alignCast(ptr));
                        // tanh'(x) = 1 - tanh(x)^2
                        for (c.a.grad, c.out.data, c.out.grad) |*ag, ov, og|
                            ag.* += og * (1.0 - ov * ov);
                    }
                }.bwd);
            }
            return out;
        }

        /// out = silu(a) = a * sigmoid(a)
        pub fn silu(self: *Self, a: *Tensor) !*Tensor {
            const out = try self.newTensor(a.data.len, a.requires_grad);
            for (out.data, a.data) |*o, av| {
                const s = 1.0 / (1.0 + @exp(-av));
                o.* = av * s;
            }

            if (a.requires_grad) {
                const Ctx = struct { a: *Tensor, out: *Tensor };
                const ctx = try self.arena.allocator().create(Ctx);
                ctx.* = .{ .a = a, .out = out };
                try self.recordOp(Ctx, ctx, struct {
                    fn bwd(ptr: *anyopaque) void {
                        const c: *Ctx = @ptrCast(@alignCast(ptr));
                        // silu'(x) = s(x)(1 + x(1-s(x)))
                        for (c.a.grad, c.a.data, c.out.grad) |*ag, av, og| {
                            const s = 1.0 / (1.0 + @exp(-av));
                            ag.* += og * s * (1.0 + av * (1.0 - s));
                        }
                    }
                }.bwd);
            }
            return out;
        }

        /// out = gelu(a)  — tanh approximation
        pub fn gelu(self: *Self, a: *Tensor) !*Tensor {
            const sqrt_2_over_pi: f32 = @sqrt(2.0 / std.math.pi);
            const out = try self.newTensor(a.data.len, a.requires_grad);
            for (out.data, a.data) |*o, av| {
                const inner = sqrt_2_over_pi * (av + 0.044715 * av * av * av);
                o.* = av * 0.5 * (1.0 + std.math.tanh(inner));
            }

            if (a.requires_grad) {
                const Ctx = struct { a: *Tensor, out: *Tensor };
                const ctx = try self.arena.allocator().create(Ctx);
                ctx.* = .{ .a = a, .out = out };
                try self.recordOp(Ctx, ctx, struct {
                    fn bwd(ptr: *anyopaque) void {
                        const sqrt2pi: f32 = @sqrt(2.0 / std.math.pi);
                        const c: *Ctx = @ptrCast(@alignCast(ptr));
                        for (c.a.grad, c.a.data, c.out.grad) |*ag, av, og| {
                            const inner = sqrt2pi * (av + 0.044715 * av * av * av);
                            const th    = std.math.tanh(inner);
                            const d     = 0.5 * th + 0.5 +
                                av * 0.5 * (1.0 - th * th) *
                                sqrt2pi * (1.0 + 3.0 * 0.044715 * av * av);
                            ag.* += og * d;
                        }
                    }
                }.bwd);
            }
            return out;
        }

        /// Scalar MSE loss: out[0] = mean((pred - target)^2)
        pub fn mse(self: *Self, pred: *Tensor, target: []const f32) !*Tensor {
            std.debug.assert(pred.data.len == target.len);
            const out = try self.newTensor(1, pred.requires_grad);
            const n: f32 = @floatFromInt(pred.data.len);
            var loss: f32 = 0;
            for (pred.data, target) |pv, tv| { const d = pv - tv; loss += d * d; }
            out.data[0] = loss / n;

            if (pred.requires_grad) {
                const Ctx = struct { pred: *Tensor, target: []const f32, n: f32, out: *Tensor };
                const alloc = self.arena.allocator();
                const ctx   = try alloc.create(Ctx);
                ctx.* = .{ .pred = pred, .target = try alloc.dupe(f32, target), .n = n, .out = out };
                try self.recordOp(Ctx, ctx, struct {
                    fn bwd(ptr: *anyopaque) void {
                        const c: *Ctx = @ptrCast(@alignCast(ptr));
                        const og = c.out.grad[0];
                        for (c.pred.grad, c.pred.data, c.target) |*pg, pv, tv|
                            pg.* += og * 2.0 * (pv - tv) / c.n;
                    }
                }.bwd);
            }
            return out;
        }

        /// Scalar cross-entropy loss with softmax: out[0] = -sum(target * log(softmax(pred)))
        pub fn crossEntropy(self: *Self, pred: *Tensor, target: []const f32) !*Tensor {
            std.debug.assert(pred.data.len == target.len);
            const out = try self.newTensor(1, pred.requires_grad);
            const n: f32 = @floatFromInt(pred.data.len);

            var max_v: f32 = pred.data[0];
            for (pred.data) |v| if (v > max_v) { max_v = v; };
            var sum_exp: f32 = 0;
            for (pred.data) |v| sum_exp += @exp(v - max_v);

            var loss: f32 = 0;
            for (pred.data, target) |pv, tv| {
                const soft = @exp(pv - max_v) / sum_exp;
                if (tv > 0) loss -= tv * @log(soft + 1e-9);
            }
            out.data[0] = loss;

            if (pred.requires_grad) {
                const Ctx = struct { pred: *Tensor, target: []const f32, n: f32, out: *Tensor };
                const alloc = self.arena.allocator();
                const ctx   = try alloc.create(Ctx);
                ctx.* = .{ .pred = pred, .target = try alloc.dupe(f32, target), .n = n, .out = out };
                try self.recordOp(Ctx, ctx, struct {
                    fn bwd(ptr: *anyopaque) void {
                        const c: *Ctx = @ptrCast(@alignCast(ptr));
                        const og = c.out.grad[0];
                        var max_vv: f32 = c.pred.data[0];
                        for (c.pred.data) |v| if (v > max_vv) { max_vv = v; };
                        var se: f32 = 0;
                        for (c.pred.data) |v| se += @exp(v - max_vv);
                        for (c.pred.grad, c.pred.data, c.target) |*pg, pv, tv| {
                            const soft = @exp(pv - max_vv) / se;
                            pg.* += og * (soft - tv) / c.n;
                        }
                    }
                }.bwd);
            }
            return out;
        }
    };
}

// ── unit tests ────────────────────────────────────────────────────────────────

const testing = std.testing;
const T = Tape(.cpu);

fn approxEq(actual: f32, expected: f32, tol: f32) !void {
    if (@abs(actual - expected) > tol) {
        std.debug.print("expected ~{d}, got {d}\n", .{ expected, actual });
        return error.TestExpectedApproxEqual;
    }
}

test "tape: add backward gives grad 1 to each input" {
    const alloc = testing.allocator;
    var tape = T.init(alloc);
    defer tape.deinit();
    const a = try Tensor.createLeaf(alloc, 1, true); defer a.deinit(alloc);
    const b = try Tensor.createLeaf(alloc, 1, true); defer b.deinit(alloc);
    a.data[0] = 3.0; b.data[0] = 5.0;
    const out = try tape.add(a, b);
    try testing.expectApproxEqAbs(@as(f32, 8.0), out.data[0], 1e-6);
    tape.backward(out);
    try approxEq(a.grad[0], 1.0, 1e-6);
    try approxEq(b.grad[0], 1.0, 1e-6);
}

test "tape: mul backward grad_a=b, grad_b=a" {
    const alloc = testing.allocator;
    var tape = T.init(alloc);
    defer tape.deinit();
    const a = try Tensor.createLeaf(alloc, 1, true); defer a.deinit(alloc);
    const b = try Tensor.createLeaf(alloc, 1, true); defer b.deinit(alloc);
    a.data[0] = 4.0; b.data[0] = 7.0;
    const out = try tape.mul(a, b);
    tape.backward(out);
    try approxEq(a.grad[0], 7.0, 1e-6);
    try approxEq(b.grad[0], 4.0, 1e-6);
}

test "tape: relu backward masks negatives" {
    const alloc = testing.allocator;
    var tape = T.init(alloc);
    defer tape.deinit();
    const a = try Tensor.createLeaf(alloc, 3, true); defer a.deinit(alloc);
    a.data[0] = -1.0; a.data[1] = 0.5; a.data[2] = 2.0;
    const out  = try tape.relu(a);
    const loss = try tape.mse(out, &.{ 0.0, 0.0, 0.0 });
    tape.backward(loss);
    try approxEq(a.grad[0], 0.0,          1e-5);
    try approxEq(a.grad[1], 2.0*0.5/3.0,  1e-5);
    try approxEq(a.grad[2], 2.0*2.0/3.0,  1e-5);
}

test "tape: sigmoid at 0 has derivative 0.25" {
    // sigmoid(0)=0.5; mse(0.5, 0)=0.25; d/dx = 2*0.5 * sigmoid'(0) = 0.25
    const alloc = testing.allocator;
    var tape = T.init(alloc);
    defer tape.deinit();
    const a = try Tensor.createLeaf(alloc, 1, true); defer a.deinit(alloc);
    a.data[0] = 0.0;
    const out  = try tape.sigmoid(a);
    const loss = try tape.mse(out, &.{0.0});
    try approxEq(loss.data[0], 0.25, 1e-6);
    tape.backward(loss);
    try approxEq(a.grad[0], 0.25, 1e-5);
}

test "tape: mse loss and gradient" {
    // pred=2, target=1 → loss=1, grad=2
    const alloc = testing.allocator;
    var tape = T.init(alloc);
    defer tape.deinit();
    const pred = try Tensor.createLeaf(alloc, 1, true); defer pred.deinit(alloc);
    pred.data[0] = 2.0;
    const loss = try tape.mse(pred, &.{1.0});
    try approxEq(loss.data[0], 1.0, 1e-6);
    tape.backward(loss);
    try approxEq(pred.grad[0], 2.0, 1e-6);
}

test "tape: matmul dot-product (1×2 @ 2×1) forward and backward" {
    // a=[3,4], b=[5,6]: out=39; da=b=[5,6], db=a=[3,4]
    const alloc = testing.allocator;
    var tape = T.init(alloc);
    defer tape.deinit();
    const a = try Tensor.createLeaf(alloc, 2, true); defer a.deinit(alloc);
    const b = try Tensor.createLeaf(alloc, 2, true); defer b.deinit(alloc);
    a.data[0] = 3.0; a.data[1] = 4.0;
    b.data[0] = 5.0; b.data[1] = 6.0;
    const out = try tape.matmul(a, b, 1, 2, 1);
    try approxEq(out.data[0], 39.0, 1e-5);
    tape.backward(out);
    try approxEq(a.grad[0], 5.0, 1e-5);
    try approxEq(a.grad[1], 6.0, 1e-5);
    try approxEq(b.grad[0], 3.0, 1e-5);
    try approxEq(b.grad[1], 4.0, 1e-5);
}

test "tape: chain rule through add and mul" {
    // f = (a + b) * b; at a=2,b=3: f=15, df/da=3, df/db=8
    // loss = mse(f, 0) = 225; d_loss/da = 30*3=90, d_loss/db = 30*8=240
    const alloc = testing.allocator;
    var tape = T.init(alloc);
    defer tape.deinit();
    const a = try Tensor.createLeaf(alloc, 1, true); defer a.deinit(alloc);
    const b = try Tensor.createLeaf(alloc, 1, true); defer b.deinit(alloc);
    a.data[0] = 2.0; b.data[0] = 3.0;
    const sum  = try tape.add(a, b);
    const prod = try tape.mul(sum, b);
    const loss = try tape.mse(prod, &.{0.0});
    try approxEq(loss.data[0], 225.0, 1e-3);
    tape.backward(loss);
    try approxEq(a.grad[0],  90.0, 1e-3);
    try approxEq(b.grad[0], 240.0, 1e-3);
}

test "tape: requires_grad=false input gets no gradient" {
    const alloc = testing.allocator;
    var tape = T.init(alloc);
    defer tape.deinit();
    const x = try Tensor.createLeaf(alloc, 1, false); defer x.deinit(alloc);
    const w = try Tensor.createLeaf(alloc, 1, true);  defer w.deinit(alloc);
    x.data[0] = 3.0; w.data[0] = 2.0;
    const out  = try tape.mul(x, w);
    const loss = try tape.mse(out, &.{0.0});
    tape.backward(loss);
    try approxEq(x.grad[0], 0.0, 1e-9); // not accumulated
    try testing.expect(w.grad[0] != 0.0);
}

test "tape: reset preserves leaf grads" {
    const alloc = testing.allocator;
    var tape = T.init(alloc);
    defer tape.deinit();
    const a = try Tensor.createLeaf(alloc, 1, true); defer a.deinit(alloc);
    a.data[0] = 5.0;
    const out = try tape.relu(a);
    _ = out;
    try testing.expect(tape.ops.items.len == 1);
    tape.reset();
    try testing.expect(tape.ops.items.len == 0);
    try approxEq(a.grad[0], 0.0, 1e-9); // leaf grad unchanged (was 0 to start)
}
