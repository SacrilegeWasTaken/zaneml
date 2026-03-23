/// Big Transformer benchmark — exercises the full training pipeline
/// on a model large enough that compute time dominates dispatch overhead.
///
/// Task: learn to negate an input sequence (output = -input).
/// d_model=128, n_heads=4, d_ff=512, 4 layers, seq_len=8.
///
/// Run:  zig build big-transformer
const std    = @import("std");
const zaneml = @import("zaneml");

const BACKEND  = .metal;
const D_MODEL  = 256;
const N_HEADS  = 4;
const D_FF     = 1024;
const MAX_SEQ  = 4;
const N_LAYERS = 4;
const SEQ_LEN  = 2;

const Stack = zaneml.TransformerStack(BACKEND, D_MODEL, N_HEADS, D_FF, MAX_SEQ, N_LAYERS, .{
    .norm = .layer_norm,
    .ffn_activation = .gelu,
    .causal = false,
});
const Net = zaneml.Network(*Stack);

const N_SAMPLES  = 8;
const EPOCHS     = 200;
const INPUT_SIZE = SEQ_LEN * D_MODEL;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Generate synthetic training data: output = -input
    var rng = std.Random.DefaultPrng.init(42);
    const rand = rng.random();

    var inputs:  [N_SAMPLES][INPUT_SIZE]f32 = undefined;
    var targets: [N_SAMPLES][INPUT_SIZE]f32 = undefined;
    for (0..N_SAMPLES) |s| {
        for (0..INPUT_SIZE) |i| {
            inputs[s][i] = rand.float(f32) * 2.0 - 1.0; // uniform [-1, 1]
            targets[s][i] = -inputs[s][i];
        }
    }

    var samples: [N_SAMPLES]Net.Sample = undefined;
    for (0..N_SAMPLES) |s| {
        samples[s] = .{ .input = &inputs[s], .target = &targets[s] };
    }

    // Init model
    var stack = try Stack.init(allocator);
    defer stack.deinit();
    var net = Net.init(allocator, stack);

    std.debug.print("Big Transformer: d_model={d}, n_heads={d}, d_ff={d}, layers={d}, seq={d}\n", .{
        D_MODEL, N_HEADS, D_FF, N_LAYERS, SEQ_LEN,
    });
    std.debug.print("Training: {d} samples x {d} epochs\n\n", .{ N_SAMPLES, EPOCHS });

    const start = std.time.nanoTimestamp();

    try net.train(&samples, .{
        .lr          = 1e-4,
        .epochs      = EPOCHS,
        .log_every   = 20,
        .optimizer   = .{ .kind = .{ .adam = .{ .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8 } } },
        .grad_clip   = 1.0,
        .lr_schedule = .{ .warmup_cosine = .{ .warmup = 20, .total = EPOCHS } },
        .loss        = .mse,
        .batch_size  = N_SAMPLES,
    });

    const end = std.time.nanoTimestamp();
    const elapsed_s = @as(f64, @floatFromInt(end - start)) / 1e9;
    std.debug.print("\nTraining time: {d:.2}s\n", .{elapsed_s});

    // Quick eval
    std.debug.print("\n--- sample predictions (first 4 values) ---\n", .{});
    var out: [INPUT_SIZE]f32 = undefined;
    for (0..2) |s| {
        net.predict(&inputs[s], &out);
        std.debug.print("  sample {d}:\n", .{s});
        std.debug.print("    target:  [{d:.4}, {d:.4}, {d:.4}, {d:.4}, ...]\n", .{
            targets[s][0], targets[s][1], targets[s][2], targets[s][3],
        });
        std.debug.print("    predict: [{d:.4}, {d:.4}, {d:.4}, {d:.4}, ...]\n", .{
            out[0], out[1], out[2], out[3],
        });
    }
}
