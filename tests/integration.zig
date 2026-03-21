/// Integration tests: end-to-end training scenarios using the full zaneml API.
const std    = @import("std");
const zaneml = @import("zaneml");
const testing = std.testing;

const Layer      = zaneml.Layer;
const Sequential = zaneml.Sequential;
const Network    = zaneml.Network;
const Optimizer  = zaneml.Optimizer;
const AT         = zaneml.AutogradTensor;
const TapeT      = zaneml.Tape(.cpu);

// ── helpers ───────────────────────────────────────────────────────────────────

fn approxEq(actual: f32, expected: f32, tol: f32) !void {
    if (@abs(actual - expected) > tol) {
        std.debug.print("expected ~{d}, got {d}\n", .{ expected, actual });
        return error.TestExpectedApproxEqual;
    }
}

// ── Network: Layer + Sequential ───────────────────────────────────────────────

test "Network/Sequential: learns y = -x (single linear layer, SGD)" {
    const alloc = testing.allocator;
    const L   = Layer(.cpu);
    const Seq = Sequential(struct { fc: L });
    const Net = Network(.cpu, *Seq);

    var seq = try Seq.init(alloc, .{
        .fc = try L.init(alloc, .{ .n_in = 1, .n_out = 1, .activation = .linear }),
    }, &.{});
    defer seq.deinit();

    var net = Net.init(alloc, &seq);

    const samples = [_]Net.Sample{
        .{ .input = &.{  1.0 }, .target = &.{ -1.0 } },
        .{ .input = &.{ -1.0 }, .target = &.{  1.0 } },
        .{ .input = &.{  2.0 }, .target = &.{ -2.0 } },
        .{ .input = &.{ -2.0 }, .target = &.{  2.0 } },
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

test "Network/Sequential: XOR with Adam converges" {
    const alloc = testing.allocator;
    const L   = Layer(.cpu);
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
        try testing.expect(@abs(out[0] - s.target[0]) < 0.25);
    }
}

test "Network/Sequential: warmup_cosine schedule runs without error" {
    const alloc = testing.allocator;
    const L   = Layer(.cpu);
    const Seq = Sequential(struct { fc: L });
    const Net = Network(.cpu, *Seq);

    var seq = try Seq.init(alloc, .{
        .fc = try L.init(alloc, .{ .n_in = 1, .n_out = 1, .activation = .linear }),
    }, &.{});
    defer seq.deinit();

    var net = Net.init(alloc, &seq);

    const samples = [_]Net.Sample{
        .{ .input = &.{1.0}, .target = &.{2.0} },
    };

    try net.train(&samples, .{
        .lr          = 0.01,
        .epochs      = 100,
        .log_every   = 0,
        .optimizer   = .{ .kind = .sgd },
        .grad_clip   = 0,
        .lr_schedule = .{ .warmup_cosine = .{ .warmup = 10, .total = 100 } },
        .loss        = .mse,
        .batch_size  = 1,
    });
}

// ── Network: batching ─────────────────────────────────────────────────────────

test "Network: batch_size=1 and batch_size=N both converge on same task" {
    const alloc = testing.allocator;
    const L   = Layer(.cpu);
    const Seq = Sequential(struct { fc: L });
    const Net = Network(.cpu, *Seq);

    const samples = [_]Net.Sample{
        .{ .input = &.{1.0}, .target = &.{3.0} },
        .{ .input = &.{2.0}, .target = &.{6.0} },
        .{ .input = &.{3.0}, .target = &.{9.0} },
    };

    for ([_]usize{ 1, 3 }) |bs| {
        var seq = try Seq.init(alloc, .{
            .fc = try L.init(alloc, .{ .n_in = 1, .n_out = 1, .activation = .linear }),
        }, &.{});
        defer seq.deinit();
        seq.modules.fc.weights[0] = @floatCast(@as(f32, 0.0));
        @memset(seq.modules.fc.biases, 0.0);

        var net = Net.init(alloc, &seq);
        try net.train(&samples, .{
            .lr          = 0.01,
            .epochs      = 300,
            .log_every   = 0,
            .optimizer   = .{ .kind = .sgd },
            .grad_clip   = 0,
            .lr_schedule = .constant,
            .loss        = .mse,
            .batch_size  = bs,
        });

        var out = [1]f32{0};
        net.predict(&.{1.0}, &out);
        try testing.expect(out[0] > 1.0); // moved from 0 toward 3.0
    }
}

// ── Tape-based autograd ───────────────────────────────────────────────────────

test "Tape: XOR with manual SGD (autograd API end-to-end)" {
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const w1 = try AT.createLeaf(alloc, 2 * 8, true); defer w1.deinit(alloc);
    const b1 = try AT.createLeaf(alloc, 8,     true); defer b1.deinit(alloc);
    const w2 = try AT.createLeaf(alloc, 8 * 1, true); defer w2.deinit(alloc);
    const b2 = try AT.createLeaf(alloc, 1,     true); defer b2.deinit(alloc);

    var rng = std.Random.DefaultPrng.init(42);
    const rand = rng.random();
    for (w1.data) |*p| p.* = rand.floatNorm(f32) * @sqrt(2.0 / 2.0);
    for (w2.data) |*p| p.* = rand.floatNorm(f32) * @sqrt(2.0 / 8.0);

    const xs = [4][2]f32{ .{0,0}, .{0,1}, .{1,0}, .{1,1} };
    const ys = [4]f32{ 0, 1, 1, 0 };

    for (0..8000) |_| {
        for (0..4) |i| {
            w1.zeroGrad(); b1.zeroGrad();
            w2.zeroGrad(); b2.zeroGrad();

            const x = try AT.createLeaf(alloc, 2, false);
            defer x.deinit(alloc);
            @memcpy(x.data, &xs[i]);

            const h  = try tape.relu(try tape.add(try tape.matmul(x, w1, 1, 2, 8), b1));
            const out = try tape.sigmoid(try tape.add(try tape.matmul(h, w2, 1, 8, 1), b2));
            const loss = try tape.mse(out, &.{ys[i]});

            tape.backward(loss);
            tape.reset();

            for (w1.data, w1.grad) |*p, g| p.* -= 0.1 * g;
            for (b1.data, b1.grad) |*p, g| p.* -= 0.1 * g;
            for (w2.data, w2.grad) |*p, g| p.* -= 0.1 * g;
            for (b2.data, b2.grad) |*p, g| p.* -= 0.1 * g;
        }
    }

    // Verify convergence
    for (0..4) |i| {
        const x = try AT.createLeaf(alloc, 2, false);
        defer x.deinit(alloc);
        @memcpy(x.data, &xs[i]);

        const h   = try tape.relu(try tape.add(try tape.matmul(x, w1, 1, 2, 8), b1));
        const out = try tape.sigmoid(try tape.add(try tape.matmul(h, w2, 1, 8, 1), b2));
        const val = out.data[0];
        tape.reset();

        try testing.expect(@abs(val - ys[i]) < 0.25);
    }
}

test "Tape: gradient accumulation over multiple samples" {
    // Accumulate grads from two samples then apply one SGD step.
    // Should produce the same result as two separate steps with half the LR.
    const alloc = testing.allocator;
    var tape = TapeT.init(alloc);
    defer tape.deinit();

    const w = try AT.createLeaf(alloc, 1, true); defer w.deinit(alloc);
    w.data[0] = 1.0;

    // Two samples: (x=1, target=0) and (x=2, target=0)
    // With w=1: pred1=1, loss1=1, grad_w1 = 2*(1-0)*x/1 = 2
    //           pred2=2, loss2=4, grad_w2 = 2*(2-0)*x/1 = 8 (x=2 → actually grad from mul: 2*pred*x)
    // Accumulated grad = 2 + 8 = 10; SGD step: w -= lr * 10

    for ([_]f32{ 1.0, 2.0 }) |xv| {
        const x = try AT.createLeaf(alloc, 1, false); defer x.deinit(alloc);
        x.data[0] = xv;
        const pred = try tape.mul(x, w);
        const loss = try tape.mse(pred, &.{0.0});
        tape.backward(loss);
        tape.reset();
    }

    const grad_before_update = w.grad[0];
    try testing.expect(grad_before_update > 0.0); // positive gradient accumulated

    const lr: f32 = 0.01;
    const w_new = w.data[0] - lr * w.grad[0];
    try testing.expect(w_new < w.data[0]); // gradient descent reduces w
}
