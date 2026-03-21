const std = @import("std");
const Activation = @import("../activation.zig").Activation;


const SIMD_WIDTH = std.simd.suggestVectorLength(f32) orelse 4;
const VecF32 = @Vector(SIMD_WIDTH, f32);


pub const CpuBackend = struct {
    pub fn forward(
        weights: []const f32,
        input: []const f32,
        biases: []const f32,
        pre_activation_out: []f32,
        out: []f32,
        activation: Activation,
    ) void {
        // z = W*x + b  →  pre_activation_out
        matmulSimd(pre_activation_out, weights, input, out.len, input.len);
        for (pre_activation_out, biases) |*z, b| z.* += b;
        // a = activation(z)  →  out
        for (out, pre_activation_out) |*o, z| o.* = applyActivation(activation, z);
    }

    // grad_out  — gradient from top
    // grad_in   — gradient going bottom
    // grad_w    — gradient weight for updates
    // grad_b    — gradient bias
    pub fn backward(
        allocator: std.mem.Allocator,
        weights: []const f32,
        input: []const f32,
        biases: []const f32,
        pre_activation: []const f32,
        grad_out: []const f32,
        grad_in: []f32,
        grad_w: []f32,
        grad_b: []f32,
        activation: Activation,
    ) !void {
        _ = biases;

        // аллоцируем delta нормально
        const delta = try allocator.alloc(f32, pre_activation.len);
        defer allocator.free(delta);

        // delta = grad_out * activation'(pre_activation)
        for (delta, pre_activation, grad_out) |*di, pre, go| {
            di.* = go * applyActivationBackward(activation, pre);
        }

        // grad_w = delta * input^T — тоже через SIMD
        for (0..delta.len) |i| {
            const dv: VecF32 = @splat(delta[i]);
            var j: usize = 0;
            while (j + SIMD_WIDTH <= input.len) : (j += SIMD_WIDTH) {
                var gv: VecF32 = grad_w[i * input.len + j ..][0..SIMD_WIDTH].*;
                const iv: VecF32 = input[j..][0..SIMD_WIDTH].*;
                gv += dv * iv;
                grad_w[i * input.len + j ..][0..SIMD_WIDTH].* = gv;
            }
            while (j < input.len) : (j += 1) {
                grad_w[i * input.len + j] += delta[i] * input[j];
            }
        }

        // grad_b += delta
        for (grad_b, delta) |*gb, di| {
            gb.* += di;
        }

        // grad_in = weights^T * delta
        matmulTransposedSimd(grad_in, weights, delta, input.len, delta.len);
    }
 };


pub fn applyActivation(activation: Activation, x: f32) f32 {
    return switch (activation) {
        .relu         => @max(0, x),
        .sigmoid      => 1.0 / (1.0 + @exp(-x)),
        .tanh         => std.math.tanh(x),
        .linear       => x,
        .leaky_relu   => |alpha| if (x > 0) x else alpha * x,
        .elu          => |alpha| if (x > 0) x else alpha * (@exp(x) - 1.0),
    };
}


pub fn applyActivationBackward(activation: Activation, x: f32) f32 {
    return switch (activation) {
        .relu         => if (x > 0) 1.0 else 0.0,
        .sigmoid      => blk: {
            const s = 1.0 / (1.0 + @exp(-x));
            break :blk s * (1.0 - s);
        },
        .tanh         => blk: {
            const t = std.math.tanh(x);
            break :blk 1.0 - t * t;
        },
        .linear       => 1.0,
        .leaky_relu   => |alpha| if (x > 0) 1.0 else alpha,
        .elu          => |alpha| if (x > 0) 1.0 else alpha * @exp(x),
    };
}


fn dotSimd(a: []const f32, b: []const f32) f32 {
    var sum: VecF32 = @splat(0.0);
    var i: usize = 0;

    while (i + SIMD_WIDTH <= a.len) : (i += SIMD_WIDTH) {
        const va: VecF32 = a[i..][0..SIMD_WIDTH].*;
        const vb: VecF32 = b[i..][0..SIMD_WIDTH].*;
        sum += va * vb;
    }

    var result = @reduce(.Add, sum);
    while (i < a.len) : (i += 1) {
        result += a[i] * b[i];
    }
    return result;
}


fn matmulTransposedSimd(
    out: []f32,
    weights: []const f32,
    delta: []const f32,
    rows: usize, // n_in
    cols: usize, // n_out
) void {
    @memset(out, 0.0);

    for (0..cols) |j| {
        const d = delta[j];
        const dv: VecF32 = @splat(d);
        const weight_row = weights[j * rows .. (j + 1) * rows];

        var i: usize = 0;
        while (i + SIMD_WIDTH <= rows) : (i += SIMD_WIDTH) {
            var ov: VecF32 = out[i..][0..SIMD_WIDTH].*;
            const wv: VecF32 = weight_row[i..][0..SIMD_WIDTH].*;
            ov += dv * wv;
            out[i..][0..SIMD_WIDTH].* = ov;
        }
        while (i < rows) : (i += 1) {
            out[i] += d * weight_row[i];
        }
    }
}


fn matmulSimd(
    out: []f32,
    weights: []const f32,
    input: []const f32,
    rows: usize,
    cols: usize,
) void {
    for (0..rows) |i| {
        const row = weights[i * cols .. (i + 1) * cols];
        out[i] = dotSimd(row, input);
    }
}



