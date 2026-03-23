/// TransformerStack -- a stack of N blocks via the unified Network interface.
/// Task: learn to invert the input sequence (out = -in).
///
/// Run: zig build transformer
const std = @import("std");
const zaneml = @import("zaneml");

const BACKEND   = zaneml.Backend.cpu;
const D_MODEL   = 16;
const N_HEADS   = 2;
const D_FF      = 32;
const MAX_SEQ   = 4;
const N_LAYERS  = 3;
const SEQ       = 3;
const EPOCHS    = 2000;
const LR: f32   = 3e-4;
const LOG_EVERY = 200;

const Stack = zaneml.TransformerStack(
    BACKEND, 
    D_MODEL, 
    N_HEADS, 
    D_FF, 
    MAX_SEQ, 
    N_LAYERS, 
    .{ 
        .norm = .layer_norm, 
        .ffn_activation = .silu, 
        .causal = false 
    }
);

const Net   = zaneml.Network(*Stack);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stack = try Stack.init(allocator);
    defer stack.deinit();

    var net = Net.init(allocator, stack);

    // Input sequence: striped pattern
    var input  = [_]f32{0} ** (SEQ * D_MODEL);
    var target = [_]f32{0} ** (SEQ * D_MODEL);
    for (0..SEQ) |t| {
        for (0..D_MODEL) |d| {
            input[t * D_MODEL + d]  = if ((t + d) % 2 == 0) 0.5 else -0.5;
            target[t * D_MODEL + d] = -input[t * D_MODEL + d];
        }
    }

    const samples = [_]Net.Sample{
        .{ .input = &input, .target = &target },
    };

    std.debug.print(
        "TransformerStack  layers={d}  d_model={d}  n_heads={d}  d_ff={d}  seq={d}\n\n",
        .{ N_LAYERS, D_MODEL, N_HEADS, D_FF, SEQ },
    );

    try net.train(&samples, .{
        .lr          = LR,
        .epochs      = EPOCHS,
        .log_every   = LOG_EVERY,
        .optimizer   = .{ .kind = .{ .adam = .{ .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8 } } },
        .grad_clip   = 1.0,
        .lr_schedule = .constant,
        .loss        = .mse,
        .batch_size  = 1,
    });

    var output = [_]f32{0} ** (SEQ * D_MODEL);
    net.predict(&input, &output);

    std.debug.print("\n--- final output vs target (position 0) ---\n", .{});
    for (0..D_MODEL) |d| {
        std.debug.print("  [{d:>2}]  out={d:>8.4}  target={d:>8.4}\n", .{
            d, output[d], target[d],
        });
    }
}
