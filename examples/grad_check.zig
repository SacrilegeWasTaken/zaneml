/// Gradient checker: verifies Metal backward via finite differences.
/// Uses a single TransformerBlock (1 layer) with small dimensions.
const std    = @import("std");
const zaneml = @import("zaneml");

const BACKEND  = .metal;
const D_MODEL  = 16;
const N_HEADS  = 2;
const D_FF     = 32;
const MAX_SEQ  = 4;
const SEQ_LEN  = 2;

const Block = zaneml.TransformerBlock(BACKEND, D_MODEL, N_HEADS, D_FF, MAX_SEQ, .{
    .norm = .layer_norm,
    .ffn_activation = .gelu,
    .causal = false,
});

const INPUT_SIZE = SEQ_LEN * D_MODEL;

fn mseLoss(output: []const f32, target: []const f32) f32 {
    var loss: f32 = 0;
    for (output, target) |o, t| {
        const d = o - t;
        loss += d * d;
    }
    return loss;
}

fn forwardLoss(block: *Block, input: []const f32, target: []const f32, output: []f32) f32 {
    block.forward(input, output);
    return mseLoss(output, target);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var block = try Block.init(allocator);
    defer block.deinit();

    // Generate deterministic test data
    var rng = std.Random.DefaultPrng.init(42);
    const rand = rng.random();

    var input: [INPUT_SIZE]f32 = undefined;
    var target: [INPUT_SIZE]f32 = undefined;
    for (0..INPUT_SIZE) |i| {
        input[i] = rand.float(f32) * 2.0 - 1.0;
        target[i] = -input[i];
    }

    var output: [INPUT_SIZE]f32 = undefined;
    var grad_out: [INPUT_SIZE]f32 = undefined;
    var grad_in: [INPUT_SIZE]f32 = undefined;

    // Forward + backward
    block.forward(&input, &output);

    // Compute grad_out = 2 * (output - target) — raw MSE gradient (no normalization)
    for (&grad_out, output, target) |*g, o, t| {
        g.* = 2.0 * (o - t);
    }

    // Zero all gradients before backward
    @memset(block.ffn1.grad_w, 0);
    @memset(block.ffn1.grad_b, 0);
    @memset(block.ffn2.grad_w, 0);
    @memset(block.ffn2.grad_b, 0);
    @memset(block.norm1.grad_gamma[0..D_MODEL], 0);
    @memset(block.norm1.grad_beta[0..D_MODEL], 0);
    @memset(block.norm2.grad_gamma[0..D_MODEL], 0);
    @memset(block.norm2.grad_beta[0..D_MODEL], 0);
    @memset(&block.attn.grad_wq, 0);
    @memset(&block.attn.grad_wk, 0);
    @memset(&block.attn.grad_wv, 0);
    @memset(&block.attn.grad_wo, 0);

    try block.backward(&input, &grad_out, &grad_in);

    // Now check gradients via finite differences
    const eps: f32 = 1e-3;
    var output_plus: [INPUT_SIZE]f32 = undefined;
    var output_minus: [INPUT_SIZE]f32 = undefined;

    std.debug.print("=== Gradient Check: TransformerBlock (Metal) ===\n", .{});
    std.debug.print("d_model={d}, n_heads={d}, d_ff={d}, seq={d}\n\n", .{ D_MODEL, N_HEADS, D_FF, SEQ_LEN });

    // Check FFN2 weight gradients (first 5 elements)
    {
        std.debug.print("--- FFN2 grad_w (first 10) ---\n", .{});
        const n_check = @min(10, block.ffn2.grad_w.len);
        for (0..n_check) |i| {
            const orig = block.ffn2.weights_compute_buffer[i];
            block.ffn2.weights_compute_buffer[i] = orig + eps;
            // Sync f32 -> f16 so forward uses perturbed weight
            block.ffn2.weights[i] = @floatCast(block.ffn2.weights_compute_buffer[i]);
            const loss_plus = forwardLoss(block, &input, &target, &output_plus);

            block.ffn2.weights_compute_buffer[i] = orig - eps;
            block.ffn2.weights[i] = @floatCast(block.ffn2.weights_compute_buffer[i]);
            const loss_minus = forwardLoss(block, &input, &target, &output_minus);

            block.ffn2.weights_compute_buffer[i] = orig;
            block.ffn2.weights[i] = @floatCast(orig);

            const numerical = (loss_plus - loss_minus) / (2.0 * eps);
            const analytical = block.ffn2.grad_w[i];
            const diff = @abs(numerical - analytical);
            const rel = if (@abs(numerical) + @abs(analytical) > 1e-7)
                diff / (@abs(numerical) + @abs(analytical))
            else
                diff;
            std.debug.print("  [{d:>3}] analytical={d:>12.8}, numerical={d:>12.8}, diff={d:.10}, rel={d:.10}\n",
                .{ i, analytical, numerical, diff, rel });
        }
    }

    // Check FFN2 bias gradients
    {
        std.debug.print("\n--- FFN2 grad_b (first 5) ---\n", .{});
        const n_check = @min(5, block.ffn2.grad_b.len);
        for (0..n_check) |i| {
            const orig = block.ffn2.biases[i];
            block.ffn2.biases[i] = orig + eps;
            const loss_plus = forwardLoss(block, &input, &target, &output_plus);
            block.ffn2.biases[i] = orig - eps;
            const loss_minus = forwardLoss(block, &input, &target, &output_minus);
            block.ffn2.biases[i] = orig;

            const numerical = (loss_plus - loss_minus) / (2.0 * eps);
            const analytical = block.ffn2.grad_b[i];
            const diff = @abs(numerical - analytical);
            const rel = if (@abs(numerical) + @abs(analytical) > 1e-7)
                diff / (@abs(numerical) + @abs(analytical))
            else
                diff;
            std.debug.print("  [{d:>3}] analytical={d:>12.8}, numerical={d:>12.8}, diff={d:.10}, rel={d:.10}\n",
                .{ i, analytical, numerical, diff, rel });
        }
    }

    // Check FFN1 weight gradients
    {
        std.debug.print("\n--- FFN1 grad_w (first 10) ---\n", .{});
        const n_check = @min(10, block.ffn1.grad_w.len);
        for (0..n_check) |i| {
            const orig = block.ffn1.weights_compute_buffer[i];
            block.ffn1.weights_compute_buffer[i] = orig + eps;
            block.ffn1.weights[i] = @floatCast(block.ffn1.weights_compute_buffer[i]);
            const loss_plus = forwardLoss(block, &input, &target, &output_plus);
            block.ffn1.weights_compute_buffer[i] = orig - eps;
            block.ffn1.weights[i] = @floatCast(block.ffn1.weights_compute_buffer[i]);
            const loss_minus = forwardLoss(block, &input, &target, &output_minus);
            block.ffn1.weights_compute_buffer[i] = orig;
            block.ffn1.weights[i] = @floatCast(orig);

            const numerical = (loss_plus - loss_minus) / (2.0 * eps);
            const analytical = block.ffn1.grad_w[i];
            const diff = @abs(numerical - analytical);
            const rel = if (@abs(numerical) + @abs(analytical) > 1e-7)
                diff / (@abs(numerical) + @abs(analytical))
            else
                diff;
            std.debug.print("  [{d:>3}] analytical={d:>12.8}, numerical={d:>12.8}, diff={d:.10}, rel={d:.10}\n",
                .{ i, analytical, numerical, diff, rel });
        }
    }

    // Check attention Wo gradients
    {
        std.debug.print("\n--- Attn grad_wo (first 10) ---\n", .{});
        const n_check = @min(10, block.attn.grad_wo.len);
        for (0..n_check) |i| {
            const orig_f32 = block.attn.wo_f[i];
            const orig_f16 = block.attn.wo[i];
            block.attn.wo_f[i] = orig_f32 + eps;
            block.attn.wo[i] = @floatCast(block.attn.wo_f[i]);
            const loss_plus = forwardLoss(block, &input, &target, &output_plus);
            block.attn.wo_f[i] = orig_f32 - eps;
            block.attn.wo[i] = @floatCast(block.attn.wo_f[i]);
            const loss_minus = forwardLoss(block, &input, &target, &output_minus);
            block.attn.wo_f[i] = orig_f32;
            block.attn.wo[i] = orig_f16;

            const numerical = (loss_plus - loss_minus) / (2.0 * eps);
            const analytical = block.attn.grad_wo[i];
            const diff = @abs(numerical - analytical);
            const rel = if (@abs(numerical) + @abs(analytical) > 1e-7)
                diff / (@abs(numerical) + @abs(analytical))
            else
                diff;
            std.debug.print("  [{d:>3}] analytical={d:>12.8}, numerical={d:>12.8}, diff={d:.10}, rel={d:.10}\n",
                .{ i, analytical, numerical, diff, rel });
        }
    }

    // Check attention Wq gradients
    {
        std.debug.print("\n--- Attn grad_wq (first 10) ---\n", .{});
        const n_check = @min(10, block.attn.grad_wq.len);
        for (0..n_check) |i| {
            const orig_f32 = block.attn.wq_f[i];
            const orig_f16 = block.attn.wq[i];
            block.attn.wq_f[i] = orig_f32 + eps;
            block.attn.wq[i] = @floatCast(block.attn.wq_f[i]);
            const loss_plus = forwardLoss(block, &input, &target, &output_plus);
            block.attn.wq_f[i] = orig_f32 - eps;
            block.attn.wq[i] = @floatCast(block.attn.wq_f[i]);
            const loss_minus = forwardLoss(block, &input, &target, &output_minus);
            block.attn.wq_f[i] = orig_f32;
            block.attn.wq[i] = orig_f16;

            const numerical = (loss_plus - loss_minus) / (2.0 * eps);
            const analytical = block.attn.grad_wq[i];
            const diff = @abs(numerical - analytical);
            const rel = if (@abs(numerical) + @abs(analytical) > 1e-7)
                diff / (@abs(numerical) + @abs(analytical))
            else
                diff;
            std.debug.print("  [{d:>3}] analytical={d:>12.8}, numerical={d:>12.8}, diff={d:.10}, rel={d:.10}\n",
                .{ i, analytical, numerical, diff, rel });
        }
    }

    // Check norm1 grad_gamma
    {
        std.debug.print("\n--- Norm1 grad_gamma (first 5) ---\n", .{});
        const n_check = @min(5, D_MODEL);
        for (0..n_check) |i| {
            const orig = block.norm1.gamma[i];
            block.norm1.gamma[i] = orig + eps;
            const loss_plus = forwardLoss(block, &input, &target, &output_plus);
            block.norm1.gamma[i] = orig - eps;
            const loss_minus = forwardLoss(block, &input, &target, &output_minus);
            block.norm1.gamma[i] = orig;

            const numerical = (loss_plus - loss_minus) / (2.0 * eps);
            const analytical = block.norm1.grad_gamma[i];
            const diff = @abs(numerical - analytical);
            const rel = if (@abs(numerical) + @abs(analytical) > 1e-7)
                diff / (@abs(numerical) + @abs(analytical))
            else
                diff;
            std.debug.print("  [{d:>3}] analytical={d:>12.8}, numerical={d:>12.8}, diff={d:.10}, rel={d:.10}\n",
                .{ i, analytical, numerical, diff, rel });
        }
    }

    // Check grad_in (input gradient)
    {
        std.debug.print("\n--- grad_in (first 10) ---\n", .{});
        const n_check = @min(10, INPUT_SIZE);
        for (0..n_check) |i| {
            var input_pert = input;
            input_pert[i] += eps;
            const loss_plus = forwardLoss(block, &input_pert, &target, &output_plus);
            input_pert[i] -= 2.0 * eps;
            const loss_minus = forwardLoss(block, &input_pert, &target, &output_minus);

            const numerical = (loss_plus - loss_minus) / (2.0 * eps);
            const analytical = grad_in[i];
            const diff = @abs(numerical - analytical);
            const rel = if (@abs(numerical) + @abs(analytical) > 1e-7)
                diff / (@abs(numerical) + @abs(analytical))
            else
                diff;
            std.debug.print("  [{d:>3}] analytical={d:>12.8}, numerical={d:>12.8}, diff={d:.10}, rel={d:.10}\n",
                .{ i, analytical, numerical, diff, rel });
        }
    }
}
