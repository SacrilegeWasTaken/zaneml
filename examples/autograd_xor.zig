/// XOR solved end-to-end using the Tape-based autograd API.
///
/// Architecture: input(2) → matmul(2→8) → bias+relu → matmul(8→1) → bias+sigmoid → MSE
/// Parameters are leaf tensors managed by the caller.
/// Training uses online SGD: one forward+backward+update per sample.
///
/// This demonstrates the low-level Tape API without using Layer/Sequential/Network.
/// Run: zig build autograd
const std    = @import("std");
const zaneml = @import("zaneml");

const AT    = zaneml.AutogradTensor;
const TapeT = zaneml.Tape(.cpu);

const LR: f32       = 0.1;
const EPOCHS: usize = 5000;
const LOG_EVERY     = 500;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var tape = TapeT.init(alloc);
    defer tape.deinit();

    // ── parameters (leaf tensors, requires_grad = true) ──────────────────────
    // W1: shape [2, 8] (n_in=2 rows × n_out=8 cols, row-major for tape.matmul)
    const w1 = try AT.createLeaf(alloc, 2 * 8, true);
    defer w1.deinit(alloc);
    const b1 = try AT.createLeaf(alloc, 8, true);
    defer b1.deinit(alloc);

    // W2: shape [8, 1]
    const w2 = try AT.createLeaf(alloc, 8 * 1, true);
    defer w2.deinit(alloc);
    const b2 = try AT.createLeaf(alloc, 1, true);
    defer b2.deinit(alloc);

    // He initialisation
    var rng  = std.Random.DefaultPrng.init(42);
    const rand = rng.random();
    for (w1.data) |*p| p.* = rand.floatNorm(f32) * @sqrt(2.0 / 2.0);
    for (w2.data) |*p| p.* = rand.floatNorm(f32) * @sqrt(2.0 / 8.0);

    // XOR dataset
    const xs = [4][2]f32{
        .{ 0, 0 }, .{ 0, 1 }, .{ 1, 0 }, .{ 1, 1 },
    };
    const ys = [4]f32{ 0, 1, 1, 0 };

    std.debug.print(
        "Autograd XOR  (2→8→1, ReLU+Sigmoid, MSE, online SGD lr={d})\n\n",
        .{LR},
    );

    for (0..EPOCHS) |epoch| {
        var total_loss: f32 = 0;

        // Online SGD: one update per sample
        for (0..4) |i| {
            // Zero leaf grads before this sample's backward pass
            w1.zeroGrad(); b1.zeroGrad();
            w2.zeroGrad(); b2.zeroGrad();

            // Create a non-differentiable input leaf (freed at end of this block)
            const x = try AT.createLeaf(alloc, 2, false);
            defer x.deinit(alloc);
            @memcpy(x.data, &xs[i]);

            // Forward:
            //   h_pre  = x(1×2) @ W1(2×8) → (1×8)
            //   h_bias = h_pre + b1
            //   h_act  = relu(h_bias)
            //   logit  = h_act(1×8) @ W2(8×1) → (1×1)
            //   l_bias = logit + b2
            //   pred   = sigmoid(l_bias)
            const h_pre  = try tape.matmul(x,    w1, 1, 2, 8);
            const h_bias = try tape.add(h_pre, b1);
            const h_act  = try tape.relu(h_bias);
            const logit  = try tape.matmul(h_act, w2, 1, 8, 1);
            const l_bias = try tape.add(logit, b2);
            const pred   = try tape.sigmoid(l_bias);
            const loss   = try tape.mse(pred, &.{ys[i]});

            total_loss += loss.data[0];

            // Backward: fills w1.grad, b1.grad, w2.grad, b2.grad
            tape.backward(loss);

            // Free all tape-owned intermediates (leaf grads are preserved)
            tape.reset();

            // SGD update
            for (w1.data, w1.grad) |*p, g| p.* -= LR * g;
            for (b1.data, b1.grad) |*p, g| p.* -= LR * g;
            for (w2.data, w2.grad) |*p, g| p.* -= LR * g;
            for (b2.data, b2.grad) |*p, g| p.* -= LR * g;
        }

        if (LOG_EVERY > 0 and (epoch + 1) % LOG_EVERY == 0) {
            std.debug.print("epoch {d:>5}  loss = {d:.6}\n", .{
                epoch + 1, total_loss / 4.0,
            });
        }
    }

    // ── inference (no backward needed) ───────────────────────────────────────
    std.debug.print("\n--- predictions ---\n", .{});
    for (0..4) |i| {
        const x = try AT.createLeaf(alloc, 2, false);
        defer x.deinit(alloc);
        @memcpy(x.data, &xs[i]);

        const h_pre  = try tape.matmul(x,    w1, 1, 2, 8);
        const h_bias = try tape.add(h_pre, b1);
        const h_act  = try tape.relu(h_bias);
        const logit  = try tape.matmul(h_act, w2, 1, 8, 1);
        const l_bias = try tape.add(logit, b2);
        const pred   = try tape.sigmoid(l_bias);

        const out_val = pred.data[0];
        tape.reset();

        std.debug.print("  {d} XOR {d}  =>  {d:.4}  (expected {d:.0})\n", .{
            @as(u1, @intFromFloat(xs[i][0])),
            @as(u1, @intFromFloat(xs[i][1])),
            out_val,
            ys[i],
        });
    }
}
