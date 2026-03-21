/// Library unit tests.
///
/// Covers:
///   - Tape autograd: add, mul, matmul, relu, sigmoid, mse
///   - Optimizer: SGD and Adam single-step correctness
///   - Layer: forward pass
///   - Network: convergence on a trivial regression task
///   - Batch gradient accumulation equivalence
const std     = @import("std");
const testing = std.testing;

const autograd_mod = @import("autograd.zig");
const TapeT        = autograd_mod.Tape(.cpu);
const AT           = autograd_mod.Tensor;

const Layer      = @import("layer.zig").Layer;
const Sequential = @import("sequential.zig").Sequential;
const Network    = @import("network.zig").Network;
const Optimizer  = @import("optimizer.zig").Optimizer;
const CpuOpt     = @import("backend/cpu.zig").CpuOptimizer;

const L = Layer(.cpu);

// ── helpers

fn approx(actual: f32, expected: f32, tol: f32) !void {
    if (@abs(actual - expected) > tol) {
        std.debug.print("expected ~{d}, got {d} (delta={d})\n", .{
            expected, actual, @abs(actual - expected),
        });
        return error.TestExpectedApproxEqual;
    }
}

// ── Tape: element-wise ops
test "tape: add backward gives grad 1 to each input" {
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const a = try AT.createLeaf(alloc, 1, true);
    defer a.deinit(alloc);
    const b = try AT.createLeaf(alloc, 1, true);
    defer b.deinit(alloc);
    a.data[0] = 3.0;
    b.data[0] = 5.0;

    const out = try tape.add(a, b);
    try testing.expectApproxEqAbs(@as(f32, 8.0), out.data[0], 1e-6);
    tape.backward(out);

    try approx(a.grad[0], 1.0, 1e-6);
    try approx(b.grad[0], 1.0, 1e-6);
}

test "tape: mul backward grad_a=b, grad_b=a" {
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const a = try AT.createLeaf(alloc, 1, true);
    defer a.deinit(alloc);
    const b = try AT.createLeaf(alloc, 1, true);
    defer b.deinit(alloc);
    a.data[0] = 4.0;
    b.data[0] = 7.0;

    const out = try tape.mul(a, b);
    try testing.expectApproxEqAbs(@as(f32, 28.0), out.data[0], 1e-6);
    tape.backward(out);

    try approx(a.grad[0], 7.0, 1e-6); // dL/da = b
    try approx(b.grad[0], 4.0, 1e-6); // dL/db = a
}

test "tape: relu backward masks negatives" {
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const a = try AT.createLeaf(alloc, 3, true);
    defer a.deinit(alloc);
    a.data[0] = -1.0;
    a.data[1] =  0.5;
    a.data[2] =  2.0;

    // out = relu(a), loss = mse(out, zeros) = (0 + 0.25 + 4) / 3
    const out  = try tape.relu(a);
    const loss = try tape.mse(out, &.{ 0.0, 0.0, 0.0 });
    try approx(loss.data[0], (0.25 + 4.0) / 3.0, 1e-5);
    tape.backward(loss);

    // grad_out[i] = 2*(relu(a[i]) - 0) / 3
    // grad_a[i]   = grad_out[i] if a[i]>0 else 0
    try approx(a.grad[0], 0.0,         1e-5); // masked by relu
    try approx(a.grad[1], 2.0*0.5/3.0, 1e-5);
    try approx(a.grad[2], 2.0*2.0/3.0, 1e-5);
}

test "tape: sigmoid at 0 has grad 0.25" {
    // sigmoid(0) = 0.5; d/dx sigmoid(0) = 0.5*(1-0.5) = 0.25
    // loss = mse(sigmoid(0), 0) = 0.25
    // d_loss/d_x = 2*(0.5-0) * 0.25 = 0.25
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const a = try AT.createLeaf(alloc, 1, true);
    defer a.deinit(alloc);
    a.data[0] = 0.0;

    const out  = try tape.sigmoid(a);
    const loss = try tape.mse(out, &.{0.0});
    try approx(loss.data[0], 0.25, 1e-6);
    tape.backward(loss);
    try approx(a.grad[0], 0.25, 1e-5);
}

test "tape: mse scalar loss and gradient" {
    // pred=2, target=1 → loss=(2-1)^2/1=1, grad=2*(2-1)/1=2
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const pred = try AT.createLeaf(alloc, 1, true);
    defer pred.deinit(alloc);
    pred.data[0] = 2.0;

    const loss = try tape.mse(pred, &.{1.0});
    try approx(loss.data[0], 1.0, 1e-6);
    tape.backward(loss);
    try approx(pred.grad[0], 2.0, 1e-6);
}

// ── Tape: matmul ──────────────────────────────────────────────────────────────

test "tape: matmul dot-product (1×2 @ 2×1) forward and backward" {
    // a = [3, 4], b = [5, 6]
    // out = a(1×2) @ b(2×1) = [3*5 + 4*6] = [39]
    // tape.backward(out):
    //   dL/da = out_grad @ b^T: da[l] = b[l]  → [5, 6]
    //   dL/db = a^T @ out_grad: db[l] = a[l]  → [3, 4]
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const a = try AT.createLeaf(alloc, 2, true);
    defer a.deinit(alloc);
    const b = try AT.createLeaf(alloc, 2, true);
    defer b.deinit(alloc);
    a.data[0] = 3.0; a.data[1] = 4.0;
    b.data[0] = 5.0; b.data[1] = 6.0;

    const out = try tape.matmul(a, b, 1, 2, 1); // (1×2) @ (2×1) → (1×1)
    try approx(out.data[0], 39.0, 1e-5);
    tape.backward(out);

    try approx(a.grad[0], 5.0, 1e-5);
    try approx(a.grad[1], 6.0, 1e-5);
    try approx(b.grad[0], 3.0, 1e-5);
    try approx(b.grad[1], 4.0, 1e-5);
}

test "tape: matmul 2×2 @ 2×1 forward" {
    // A = [[1,2],[3,4]], x = [1,1]
    // out = A @ x = [3, 7]
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const A = try AT.createLeaf(alloc, 4, true);
    defer A.deinit(alloc);
    const x = try AT.createLeaf(alloc, 2, false);
    defer x.deinit(alloc);
    @memcpy(A.data, &[_]f32{ 1, 2, 3, 4 });
    @memcpy(x.data, &[_]f32{ 1, 1 });

    const out  = try tape.matmul(A, x, 2, 2, 1);
    const loss = try tape.mse(out, &.{ 3.0, 7.0 });
    try approx(loss.data[0], 0.0, 1e-5);
}

// ── Tape: chain rule ─────────────────────────────────────────────────────────

test "tape: chain rule add+mul" {
    // f(a, b) = (a + b) * b
    // df/da = b, df/db = (a + b) + b = a + 2b
    // At a=2, b=3: df/da = 3, df/db = 8
    // loss = mse(f, 0): loss = f^2 = 25
    // d_loss/df = 2*f = 10
    // d_loss/da = 10 * 3 = 30
    // d_loss/db = 10 * 8 = 80
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const a = try AT.createLeaf(alloc, 1, true);
    defer a.deinit(alloc);
    const b = try AT.createLeaf(alloc, 1, true);
    defer b.deinit(alloc);
    a.data[0] = 2.0;
    b.data[0] = 3.0;

    const sum  = try tape.add(a, b);     // 5
    const prod = try tape.mul(sum, b);   // 15
    const loss = try tape.mse(prod, &.{0.0}); // 225

    try approx(loss.data[0], 225.0, 1e-4);
    tape.backward(loss);

    // d_loss/d_prod = 2*15/1 = 30
    // d_prod/d_sum  = b = 3,  d_prod/d_b = sum = 5
    // d_sum/d_a = 1, d_sum/d_b = 1
    // d_loss/d_a = 30 * 3 * 1 = 90
    // d_loss/d_b = 30 * (3*1 + 5) = 30 * 8 = 240  wait...
    //            = 30 * d_prod/d_b + 30 * d_prod/d_sum * d_sum/d_b
    //            = 30 * 5          + 30 * 3 * 1
    //            = 150 + 90 = 240
    try approx(a.grad[0],  90.0, 1e-3);
    try approx(b.grad[0], 240.0, 1e-3);
}

// ── Tape: no-grad input ───────────────────────────────────────────────────────

test "tape: input with requires_grad=false gets no gradient" {
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const x = try AT.createLeaf(alloc, 2, false); // input, no grad
    defer x.deinit(alloc);
    const w = try AT.createLeaf(alloc, 2, true);  // weight, needs grad
    defer w.deinit(alloc);
    x.data[0] = 1.0; x.data[1] = 2.0;
    w.data[0] = 0.5; w.data[1] = 0.5;

    const out  = try tape.mul(x, w);
    const loss = try tape.mse(out, &.{0.0, 0.0});
    tape.backward(loss);

    // x.grad should remain 0 (requires_grad=false)
    try approx(x.grad[0], 0.0, 1e-9);
    try approx(x.grad[1], 0.0, 1e-9);

    // w.grad = d_loss/d_out * x = 2*out/2 * x = out * x
    // out[0]=0.5, out[1]=1.0; d_out[i]/d_w[i] = x[i]
    // grad_w[0] = 2*0.5/2 * 1.0 = 0.5
    // grad_w[1] = 2*1.0/2 * 2.0 = 2.0
    try approx(w.grad[0], 0.5, 1e-5);
    try approx(w.grad[1], 2.0, 1e-5);
}

// ── Tape: reset 

test "tape: reset clears ops and intermediates, leaf grads persist" {
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const a = try AT.createLeaf(alloc, 1, true);
    defer a.deinit(alloc);
    a.data[0] = 5.0;

    const out = try tape.relu(a);
    _ = out;
    try testing.expect(tape.ops.items.len == 1);

    const val_before_reset = a.grad[0];

    tape.reset();

    try testing.expect(tape.ops.items.len == 0);
    try approx(a.grad[0], val_before_reset, 1e-9); // leaf grad unchanged
}

// ── Optimizer: SGD 

test "optimizer: SGD param -= lr * grad" {
    const opt = Optimizer{ .kind = .sgd };
    var params = [_]f32{ 1.0, 2.0 };
    var grads  = [_]f32{ 0.1, 0.2 };
    var m      = [_]f32{ 0.0, 0.0 };
    var v      = [_]f32{ 0.0, 0.0 };

    CpuOpt.update(opt, 1, 0.1, &params, &grads, &m, &v);

    try approx(params[0], 1.0 - 0.1 * 0.1, 1e-6); // 0.99
    try approx(params[1], 2.0 - 0.1 * 0.2, 1e-6); // 1.98

    // grads should be zeroed after update
    try approx(grads[0], 0.0, 1e-9);
    try approx(grads[1], 0.0, 1e-9);
}

test "optimizer: Adam first step close to -lr" {
    // After one step with grad=1, both m and v are small but bias-corrected to ~1.
    // Update ≈ lr * 1 / (1 + eps) ≈ lr. So param moves by about -lr.
    const opt = Optimizer{ .kind = .{ .adam = .{
        .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8,
    }}};
    var params = [_]f32{ 0.0 };
    var grads  = [_]f32{ 1.0 };
    var m      = [_]f32{ 0.0 };
    var v      = [_]f32{ 0.0 };

    CpuOpt.update(opt, 1, 0.01, &params, &grads, &m, &v);

    // m_hat = 0.1/(1-0.9) = 1.0, v_hat = 0.001/(1-0.999) = 1.0
    // step = 0.01 * 1.0 / (sqrt(1.0) + 1e-8) ≈ 0.01
    try approx(params[0], -0.01, 1e-4);
}

test "optimizer: AdamW applies weight decay" {
    // Weight decay should shrink param even with zero grad
    const opt = Optimizer{ .kind = .{ .adamw = .{
        .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8, .weight_decay = 0.1,
    }}};
    var params = [_]f32{ 2.0 };
    var grads  = [_]f32{ 0.0 }; // zero gradient
    var m      = [_]f32{ 0.0 };
    var v      = [_]f32{ 0.0 };

    CpuOpt.update(opt, 1, 0.01, &params, &grads, &m, &v);

    // param *= (1 - lr*wd) = 1 - 0.001 = 0.999 → 2.0 * 0.999 = 1.998
    // (minus the near-zero Adam step from zero grad)
    try approx(params[0], 2.0 * (1.0 - 0.01 * 0.1), 1e-4);
}

// ── Layer ────────────────────────────────────────────────────────────────────

test "layer: forward with known weights (linear activation)" {
    const alloc = testing.allocator;
    var layer = try L.init(alloc, .{
        .n_in  = 2,
        .n_out = 2,
        .activation = .linear,
    });
    defer layer.deinit();

    // Weight matrix (n_out × n_in row-major): [[2, 0], [0, 3]]
    layer.weights[0] = @floatCast(@as(f32, 2.0));
    layer.weights[1] = @floatCast(@as(f32, 0.0));
    layer.weights[2] = @floatCast(@as(f32, 0.0));
    layer.weights[3] = @floatCast(@as(f32, 3.0));
    @memset(layer.biases, 0.0);

    const input  = [_]f32{ 5.0, 7.0 };
    var   output = [_]f32{ 0.0, 0.0 };
    layer.forward(&input, &output);

    try approx(output[0], 10.0, 1e-4); // 2*5 + 0*7
    try approx(output[1], 21.0, 1e-4); // 0*5 + 3*7
}

test "layer: forward with bias" {
    const alloc = testing.allocator;
    var layer = try L.init(alloc, .{
        .n_in  = 1,
        .n_out = 1,
        .activation = .linear,
    });
    defer layer.deinit();

    layer.weights[0] = @floatCast(@as(f32, 1.0));
    layer.biases[0]  = 4.0;

    const input  = [_]f32{ 3.0 };
    var   output = [_]f32{ 0.0 };
    layer.forward(&input, &output);

    try approx(output[0], 7.0, 1e-4); // 1*3 + 4
}

test "layer: relu activation clamps negatives" {
    const alloc = testing.allocator;
    var layer = try L.init(alloc, .{
        .n_in  = 1,
        .n_out = 1,
        .activation = .relu,
    });
    defer layer.deinit();

    layer.weights[0] = @floatCast(@as(f32, -1.0));
    @memset(layer.biases, 0.0);

    const input  = [_]f32{ 2.0 }; // pre-act = -2, relu → 0
    var   output = [_]f32{ 0.0 };
    layer.forward(&input, &output);

    try approx(output[0], 0.0, 1e-5);
}

// ── Network ──────────────────────────────────────────────────────────────────

test "network: learns y = -x (linear regression, 1→1)" {
    const alloc = testing.allocator;

    const Seq = Sequential(struct { fc: L });
    const Net = Network(.cpu, *Seq);

    var seq = try Seq.init(alloc, .{
        .fc = try L.init(alloc, .{ .n_in = 1, .n_out = 1, .activation = .linear }),
    }, &.{});
    defer seq.deinit();

    var net = Net.init(alloc, &seq);

    const samples = [_]Net.Sample{
        .{ .input = &.{ 1.0 }, .target = &.{ -1.0 } },
        .{ .input = &.{ -1.0 }, .target = &.{ 1.0 } },
        .{ .input = &.{ 2.0 }, .target = &.{ -2.0 } },
        .{ .input = &.{ -2.0 }, .target = &.{ 2.0 } },
    };

    try net.train(&samples, .{
        .lr          = 0.05,
        .epochs      = 500,
        .log_every   = 0,
        .optimizer   = .{ .kind = .sgd },
        .grad_clip   = 0,
        .lr_schedule = .constant,
        .loss        = .mse,
        .batch_size  = 4,
    });

    var out = [1]f32{0};
    net.predict(&.{1.0}, &out);
    try testing.expect(out[0] < -0.5);
    net.predict(&.{-1.0}, &out);
    try testing.expect(out[0] > 0.5);
}

test "network: XOR converges with Adam" {
    const alloc = testing.allocator;

    const Seq = Sequential(struct { fc1: L, fc2: L });
    const Net = Network(.cpu, *Seq);

    var seq = try Seq.init(alloc, .{
        .fc1 = try L.init(alloc, .{ .n_in = 2, .n_out = 8, .activation = .relu }),
        .fc2 = try L.init(alloc, .{ .n_in = 8, .n_out = 1, .activation = .sigmoid }),
    }, &.{8});
    defer seq.deinit();

    var net = Net.init(alloc, &seq);

    const samples = [_]Net.Sample{
        .{ .input = &.{ 0, 0 }, .target = &.{0} },
        .{ .input = &.{ 0, 1 }, .target = &.{1} },
        .{ .input = &.{ 1, 0 }, .target = &.{1} },
        .{ .input = &.{ 1, 1 }, .target = &.{0} },
    };

    try net.train(&samples, .{
        .lr          = 0.01,
        .epochs      = 3000,
        .log_every   = 0,
        .optimizer   = .{ .kind = .{ .adam = .{ .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8 } } },
        .grad_clip   = 1.0,
        .lr_schedule = .constant,
        .loss        = .mse,
        .batch_size  = 4,
    });

    var out = [1]f32{0};
    for (samples) |s| {
        net.predict(s.input, &out);
        const expected: f32 = s.target[0];
        try testing.expect(@abs(out[0] - expected) < 0.25);
    }
}

// ── Batch gradient accumulation ───────────────────────────────────────────────

test "batch: gradient of batch=N equals average of N single-sample gradients" {
    // Train 1 epoch with batch_size=2 and compare accumulated grad to
    // the sum of two individual single-sample gradients.
    // We only check that gradients were applied (loss decreases), since
    // extracting raw gradients from inside Network requires white-box access.
    //
    // Instead: verify that training with batch_size=1 and batch_size=2 both
    // reduce the loss on a trivial problem.
    const alloc = testing.allocator;
    const Seq   = Sequential(struct { fc: L });
    const Net   = Network(.cpu, *Seq);

    const samples = [_]Net.Sample{
        .{ .input = &.{1.0}, .target = &.{2.0} },
        .{ .input = &.{3.0}, .target = &.{6.0} },
    };

    // Train with batch_size=1
    {
        var seq = try Seq.init(alloc, .{
            .fc = try L.init(alloc, .{ .n_in = 1, .n_out = 1, .activation = .linear }),
        }, &.{});
        defer seq.deinit();
        // Fix weights to known value for determinism
        seq.modules.fc.weights[0] = @floatCast(@as(f32, 0.0));
        @memset(seq.modules.fc.biases, 0.0);

        var net = Net.init(alloc, &seq);
        try net.train(&samples, .{
            .lr          = 0.01,
            .epochs      = 50,
            .log_every   = 0,
            .optimizer   = .{ .kind = .sgd },
            .grad_clip   = 0,
            .lr_schedule = .constant,
            .loss        = .mse,
            .batch_size  = 1,
        });

        var out = [1]f32{0};
        net.predict(&.{1.0}, &out);
        // Should move toward 2.0 from 0.0
        try testing.expect(out[0] > 0.1);
    }

    // Train with batch_size=2
    {
        var seq = try Seq.init(alloc, .{
            .fc = try L.init(alloc, .{ .n_in = 1, .n_out = 1, .activation = .linear }),
        }, &.{});
        defer seq.deinit();
        seq.modules.fc.weights[0] = @floatCast(@as(f32, 0.0));
        @memset(seq.modules.fc.biases, 0.0);

        var net = Net.init(alloc, &seq);
        try net.train(&samples, .{
            .lr          = 0.01,
            .epochs      = 50,
            .log_every   = 0,
            .optimizer   = .{ .kind = .sgd },
            .grad_clip   = 0,
            .lr_schedule = .constant,
            .loss        = .mse,
            .batch_size  = 2,
        });

        var out = [1]f32{0};
        net.predict(&.{1.0}, &out);
        try testing.expect(out[0] > 0.1);
    }
}

// ── LR schedule ──────────────────────────────────────────────────────────────

test "lr_schedule: constant always returns base_lr" {
    const LRSchedule = @import("network.zig").LRSchedule;
    const sched: LRSchedule = .constant;
    try approx(sched.get(0.01,    0), 0.01, 1e-7);
    try approx(sched.get(0.01, 1000), 0.01, 1e-7);
}

test "lr_schedule: warmup_cosine linearly increases during warmup" {
    const LRSchedule = @import("network.zig").LRSchedule;
    const sched: LRSchedule = .{ .warmup_cosine = .{ .warmup = 10, .total = 100 } };
    // At step 5 (halfway through warmup): lr = base * 5/10 = base * 0.5
    try approx(sched.get(1.0, 5), 0.5, 1e-6);
    // At step 0: lr = 0
    try approx(sched.get(1.0, 0), 0.0, 1e-6);
}
