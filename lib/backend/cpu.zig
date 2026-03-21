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
        // z = W*x + b  ->  pre_activation_out
        matmulSimd(pre_activation_out, weights, input, out.len, input.len);
        for (pre_activation_out, biases) |*z, b| z.* += b;
        // a = activation(z)  ->  out
        for (out, pre_activation_out) |*o, z| o.* = applyActivation(activation, z);
    }

    // grad_out  -- gradient from top
    // grad_in   -- gradient going bottom
    // grad_w    -- gradient weight for updates
    // grad_b    -- gradient bias
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

        const delta = try allocator.alloc(f32, pre_activation.len);
        defer allocator.free(delta);

        // delta = grad_out * activation'(pre_activation)
        for (delta, pre_activation, grad_out) |*di, pre, go| {
            di.* = go * applyActivationBackward(activation, pre);
        }

        // grad_w = delta * input^T
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
        .gelu         => geluForward(x),
        .silu         => siluForward(x),
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
        .gelu         => geluBackward(x),
        .silu         => siluBackward(x),
    };
}

/// GELU forward: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
fn geluForward(x: f32) f32 {
    const sqrt_2_over_pi: f32 = @sqrt(2.0 / std.math.pi);
    const inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    return x * 0.5 * (1.0 + std.math.tanh(inner));
}

/// GELU backward (approximate derivative)
fn geluBackward(x: f32) f32 {
    const sqrt_2_over_pi: f32 = @sqrt(2.0 / std.math.pi);
    const inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
    const tanh_val = std.math.tanh(inner);
    return 0.5 * tanh_val
        + 0.5
        + x * 0.5 * (1.0 - tanh_val * tanh_val) * sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
}

/// SiLU forward: x * sigmoid(x)
fn siluForward(x: f32) f32 {
    const s = 1.0 / (1.0 + @exp(-x));
    return x * s;
}

/// SiLU backward: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
fn siluBackward(x: f32) f32 {
    const s = 1.0 / (1.0 + @exp(-x));
    return s * (1.0 + x * (1.0 - s));
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


pub const CpuLayerNorm = struct {
    pub fn forward(
        input: []const f32,
        out: []f32,
        gamma: []const f32,
        beta: []const f32,
        eps: f32,
        x_norm_out: []f32,
        mean_out: *f32,
        rstd_out: *f32,
    ) void {
        const dim = input.len;
        var mean: f32 = 0;
        for (input) |x| mean += x;
        mean /= @as(f32, @floatFromInt(dim));

        var variance: f32 = 0;
        for (input) |x| { const d = x - mean; variance += d * d; }
        variance /= @as(f32, @floatFromInt(dim));

        const rstd = 1.0 / @sqrt(variance + eps);
        mean_out.* = mean;
        rstd_out.* = rstd;

        for (input, out, x_norm_out, gamma, beta) |x, *o, *xn, g, b| {
            xn.* = (x - mean) * rstd;
            o.* = xn.* * g + b;
        }
    }

    pub fn backward(
        grad_out: []const f32,
        grad_in: []f32,
        grad_gamma: []f32,
        grad_beta: []f32,
        x_norm: []const f32,
        gamma: []const f32,
        rstd: f32,
    ) void {
        const dim = grad_out.len;
        for (grad_out, x_norm, grad_gamma, grad_beta) |go, xn, *gg, *gb| {
            gg.* += go * xn;
            gb.* += go;
        }
        var sum1: f32 = 0;
        var sum2: f32 = 0;
        for (grad_out, x_norm, gamma) |go, xn, g| {
            sum1 += go * g;
            sum2 += go * g * xn;
        }
        const inv_dim = rstd / @as(f32, @floatFromInt(dim));
        for (grad_in, grad_out, x_norm, gamma) |*gi, go, xn, g| {
            gi.* = inv_dim * (@as(f32, @floatFromInt(dim)) * go * g - sum1 - xn * sum2);
        }
    }
};


pub const CpuAttention = struct {
    /// Forward pass for multi-head attention.
    /// When causal=true, applies causal mask (upper-triangular -1e9) before softmax.
    pub fn forward(
        comptime d_model: usize,
        comptime n_heads: usize,
        comptime d_head: usize,
        comptime max_seq: usize,
        attn_scale: f32,
        seq: usize,
        input: []const f32,
        output: []f32,
        wq_f: []const f32,
        wk_f: []const f32,
        wv_f: []const f32,
        wo_f: []const f32,
        cache_q: []f32,
        cache_k: []f32,
        cache_v: []f32,
        cache_attn: []f32,
        cache_concat: []f32,
        scratch: []f32,
        causal: bool,
    ) void {
        _ = max_seq;

        const mm = struct {
            fn call(out: []f32, a: []const f32, b: []const f32, m: usize, k: usize, n: usize) void {
                for (0..m) |i| {
                    for (0..n) |j| {
                        var s: f32 = 0;
                        for (0..k) |l| s += a[i * k + l] * b[l * n + j];
                        out[i * n + j] = s;
                    }
                }
            }
        }.call;

        const softmaxRow = struct {
            fn call(row: []f32) void {
                var max: f32 = row[0];
                for (row) |v| if (v > max) { max = v; };
                var sum: f32 = 0;
                for (row) |*v| { v.* = @exp(v.* - max); sum += v.*; }
                for (row) |*v| v.* /= sum;
            }
        }.call;

        mm(cache_q[0 .. seq * d_model], input, wq_f, seq, d_model, d_model);
        mm(cache_k[0 .. seq * d_model], input, wk_f, seq, d_model, d_model);
        mm(cache_v[0 .. seq * d_model], input, wv_f, seq, d_model, d_model);

        @memset(cache_concat[0 .. seq * d_model], 0);

        for (0..n_heads) |h| {
            const ho = h * d_head;
            const ao = h * seq * seq;
            for (0..seq) |i| {
                for (0..seq) |j| {
                    var dot: f32 = 0;
                    for (0..d_head) |dk| {
                        dot += cache_q[i * d_model + ho + dk] * cache_k[j * d_model + ho + dk];
                    }
                    scratch[i * seq + j] = dot * attn_scale;
                }
                // Apply causal mask: set future positions to -1e9
                if (causal) {
                    for (0..seq) |j| {
                        if (j > i) scratch[i * seq + j] = -1e9;
                    }
                }
                softmaxRow(scratch[i * seq .. (i + 1) * seq]);
                @memcpy(cache_attn[ao + i * seq .. ao + (i + 1) * seq], scratch[i * seq .. (i + 1) * seq]);
            }
            for (0..seq) |i| {
                for (0..d_head) |dk| {
                    var s: f32 = 0;
                    for (0..seq) |j| s += cache_attn[ao + i * seq + j] * cache_v[j * d_model + ho + dk];
                    cache_concat[i * d_model + ho + dk] = s;
                }
            }
        }
        mm(output[0 .. seq * d_model], cache_concat[0 .. seq * d_model], wo_f, seq, d_model, d_model);
    }

    pub fn backward(
        comptime d_model: usize,
        comptime n_heads: usize,
        comptime d_head: usize,
        attn_scale: f32,
        seq: usize,
        input: []const f32,
        grad_out: []const f32,
        grad_in: []f32,
        wq_f: []const f32,
        wk_f: []const f32,
        wv_f: []const f32,
        wo_f: []const f32,
        cache_q: []const f32,
        cache_k: []const f32,
        cache_v: []const f32,
        cache_attn: []const f32,
        cache_concat: []const f32,
        grad_wq: []f32,
        grad_wk: []f32,
        grad_wv: []f32,
        grad_wo: []f32,
        grad_q: []f32,
        grad_k: []f32,
        grad_v: []f32,
        grad_concat: []f32,
        scratch: []f32,
    ) void {
        const sm = seq * d_model;

        // grad_wo += concat^T @ grad_out
        @memset(grad_concat[0..sm], 0);
        for (0..d_model) |i| {
            for (0..d_model) |j| {
                var s: f32 = 0;
                for (0..seq) |t| s += cache_concat[t * d_model + i] * grad_out[t * d_model + j];
                grad_wo[i * d_model + j] += s;
            }
        }
        // grad_concat = grad_out @ W_o^T
        for (0..seq) |t| {
            for (0..d_model) |i| {
                var s: f32 = 0;
                for (0..d_model) |j| s += grad_out[t * d_model + j] * wo_f[i * d_model + j];
                grad_concat[t * d_model + i] = s;
            }
        }

        @memset(grad_q[0..sm], 0);
        @memset(grad_k[0..sm], 0);
        @memset(grad_v[0..sm], 0);

        for (0..n_heads) |h| {
            const ho = h * d_head;
            const ao = h * seq * seq;

            for (0..seq) |j| {
                for (0..d_head) |dk| {
                    var s: f32 = 0;
                    for (0..seq) |i| s += cache_attn[ao + i * seq + j] * grad_concat[i * d_model + ho + dk];
                    grad_v[j * d_model + ho + dk] += s;
                }
            }

            // grad_attn into scratch
            for (0..seq) |i| {
                for (0..seq) |j| {
                    var s: f32 = 0;
                    for (0..d_head) |dk| s += grad_concat[i * d_model + ho + dk] * cache_v[j * d_model + ho + dk];
                    scratch[i * seq + j] = s;
                }
            }

            // softmax backward
            for (0..seq) |i| {
                var dot: f32 = 0;
                for (0..seq) |j| dot += scratch[i * seq + j] * cache_attn[ao + i * seq + j];
                for (0..seq) |j| {
                    scratch[i * seq + j] = cache_attn[ao + i * seq + j] * (scratch[i * seq + j] - dot);
                }
            }

            for (0..seq) |i| {
                for (0..d_head) |dk| {
                    var s: f32 = 0;
                    for (0..seq) |j| s += scratch[i * seq + j] * cache_k[j * d_model + ho + dk];
                    grad_q[i * d_model + ho + dk] += s * attn_scale;
                }
            }
            for (0..seq) |j| {
                for (0..d_head) |dk| {
                    var s: f32 = 0;
                    for (0..seq) |i| s += scratch[i * seq + j] * cache_q[i * d_model + ho + dk];
                    grad_k[j * d_model + ho + dk] += s * attn_scale;
                }
            }
        }

        @memset(grad_in[0..sm], 0);
        // grad_wq += input^T @ grad_q, grad_in += grad_q @ W_q^T  (and same for K, V)
        const projs = [3]struct { gw: []f32, w: []const f32, gp: []f32 }{
            .{ .gw = grad_wq, .w = wq_f, .gp = grad_q },
            .{ .gw = grad_wk, .w = wk_f, .gp = grad_k },
            .{ .gw = grad_wv, .w = wv_f, .gp = grad_v },
        };
        for (projs) |p| {
            for (0..d_model) |i| {
                for (0..d_model) |j| {
                    var s: f32 = 0;
                    for (0..seq) |t| s += input[t * d_model + i] * p.gp[t * d_model + j];
                    p.gw[i * d_model + j] += s;
                }
            }
            for (0..seq) |t| {
                for (0..d_model) |i| {
                    var s: f32 = 0;
                    for (0..d_model) |j| s += p.gp[t * d_model + j] * p.w[i * d_model + j];
                    grad_in[t * d_model + i] += s;
                }
            }
        }
    }
};


pub const CpuOptimizer = struct {
    /// In-place parameter update. Zeroes grads after update.
    /// m/v may be empty slices for SGD (ignored).
    pub fn update(
        opt: @import("../optimizer.zig").Optimizer,
        t: usize,
        lr: f32,
        params: []f32,
        grads: []f32,
        m: []f32,
        v: []f32,
    ) void {
        switch (opt.kind) {
            .sgd => {
                for (params, grads) |*p, g| p.* -= lr * g;
            },
            .adam => |cfg| {
                const b1 = cfg.beta1;
                const b2 = cfg.beta2;
                const eps = cfg.eps;
                const t_f: f32 = @floatFromInt(t);
                const bc1 = 1.0 - std.math.pow(f32, b1, t_f);
                const bc2 = 1.0 - std.math.pow(f32, b2, t_f);
                for (params, grads, m, v) |*p, g, *mi, *vi| {
                    mi.* = b1 * mi.* + (1.0 - b1) * g;
                    vi.* = b2 * vi.* + (1.0 - b2) * g * g;
                    const m_hat = mi.* / bc1;
                    const v_hat = vi.* / bc2;
                    p.* -= lr * m_hat / (@sqrt(v_hat) + eps);
                }
            },
            .adamw => |cfg| {
                const b1 = cfg.beta1;
                const b2 = cfg.beta2;
                const eps = cfg.eps;
                const wd  = cfg.weight_decay;
                const t_f: f32 = @floatFromInt(t);
                const bc1 = 1.0 - std.math.pow(f32, b1, t_f);
                const bc2 = 1.0 - std.math.pow(f32, b2, t_f);
                for (params, grads, m, v) |*p, g, *mi, *vi| {
                    mi.* = b1 * mi.* + (1.0 - b1) * g;
                    vi.* = b2 * vi.* + (1.0 - b2) * g * g;
                    const m_hat = mi.* / bc1;
                    const v_hat = vi.* / bc2;
                    p.* = p.* * (1.0 - lr * wd) - lr * m_hat / (@sqrt(v_hat) + eps);
                }
            },
        }
        @memset(grads, 0);
    }
};

pub const CpuLoss = struct {
    /// MSE: returns raw loss sum, fills grad_out with normalized gradient.
    pub fn mse(output: []const f32, target: []const f32, grad_out: []f32, inv: f32) f32 {
        var loss: f32 = 0;
        for (grad_out, output, target) |*g, o, tgt| {
            const d = o - tgt;
            loss += d * d;
            g.* = 2.0 * d * inv;
        }
        return loss;
    }

    /// Cross-entropy with softmax: returns raw loss sum, fills grad_out.
    pub fn cross_entropy(output: []const f32, target: []const f32, grad_out: []f32, inv: f32) f32 {
        var max_val: f32 = output[0];
        for (output) |v| if (v > max_val) { max_val = v; };
        var sum_exp: f32 = 0;
        for (output) |v| sum_exp += @exp(v - max_val);
        var loss: f32 = 0;
        for (grad_out, output, target) |*g, o, tgt| {
            const soft = @exp(o - max_val) / sum_exp;
            g.* = (soft - tgt) * inv;
            if (tgt > 0) loss -= tgt * @log(soft + 1e-9);
        }
        return loss;
    }
};

pub const CpuRMSNorm = struct {
    pub fn forward(
        input: []const f32,
        out: []f32,
        gamma: []const f32,
        eps: f32,
        x_norm_out: []f32,
        rstd_out: *f32,
    ) void {
        const dim = input.len;
        var mean_sq: f32 = 0;
        for (input) |x| mean_sq += x * x;
        mean_sq /= @as(f32, @floatFromInt(dim));
        const rms = @sqrt(mean_sq + eps);
        rstd_out.* = rms;
        for (input, out, x_norm_out, gamma) |x, *o, *xn, g| {
            xn.* = x / rms;
            o.* = g * xn.*;
        }
    }

    pub fn backward(
        grad_out: []const f32,
        grad_in: []f32,
        grad_gamma: []f32,
        x_norm: []const f32,
        gamma: []const f32,
        rstd: f32,
    ) void {
        const dim = grad_out.len;
        for (grad_out, x_norm, grad_gamma) |go, xn, *gg| gg.* += go * xn;
        var dot: f32 = 0;
        for (grad_out, x_norm, gamma) |go, xn, g| dot += go * xn * g;
        dot /= @as(f32, @floatFromInt(dim));
        const inv_rms = 1.0 / rstd;
        for (grad_in, grad_out, x_norm, gamma) |*gi, go, xn, g| {
            gi.* = inv_rms * (g * go - xn * dot);
        }
    }
};

pub const CpuPositionalEmbedding = struct {
    pub fn forward(input: []const f32, output: []f32, embed: []const f32) void {
        for (output, input, embed) |*o, x, pe| o.* = x + pe;
    }

    pub fn backward(grad_out: []const f32, grad_in: []f32, grad_embed: []f32) void {
        for (grad_in, grad_out) |*gi, go| gi.* = go;
        for (grad_embed, grad_out) |*ge, go| ge.* += go;
    }
};

pub const CpuEmbedding = struct {
    pub fn forward(d_model: usize, indices: []const u32, output: []f32, table: []const f32) void {
        for (indices, 0..) |idx, i| {
            @memcpy(output[i * d_model .. (i + 1) * d_model], table[idx * d_model .. (idx + 1) * d_model]);
        }
    }

    pub fn backward(d_model: usize, indices: []const u32, grad_out: []const f32, grad_table: []f32) void {
        for (indices, 0..) |idx, i| {
            const src = grad_out[i * d_model .. (i + 1) * d_model];
            const dst = grad_table[idx * d_model .. (idx + 1) * d_model];
            for (dst, src) |*gd, gr| gd.* += gr;
        }
    }
};

pub const CpuDropout = struct {
    pub fn forward(
        input: []const f32,
        output: []f32,
        mask: []bool,
        rate: f32,
        training: bool,
        rng: *std.Random.DefaultPrng,
    ) void {
        if (!training or rate == 0.0) {
            @memcpy(output[0..input.len], input);
            return;
        }
        const scale = 1.0 / (1.0 - rate);
        for (input, output[0..input.len], mask[0..input.len]) |x, *o, *keep| {
            keep.* = rng.random().float(f32) >= rate;
            o.* = if (keep.*) x * scale else 0;
        }
    }

    pub fn backward(grad_out: []const f32, grad_in: []f32, mask: []const bool, rate: f32) void {
        const scale = 1.0 / (1.0 - rate);
        for (grad_in, grad_out, mask[0..grad_out.len]) |*gi, go, keep| {
            gi.* = if (keep) go * scale else 0;
        }
    }
};

pub const CpuMatmul = struct {
    /// out[m*n] = a[m*k] @ b[k*n]  (overwrites out)
    pub fn forward(a: []const f32, b: []const f32, out: []f32, m: usize, k: usize, n: usize) void {
        for (0..m) |i| {
            for (0..n) |j| {
                var s: f32 = 0;
                for (0..k) |l| s += a[i * k + l] * b[l * n + j];
                out[i * n + j] = s;
            }
        }
    }

    /// grad_a += grad_out @ b^T:  grad_a[m*k] += grad_out[m*n] @ b[k*n]^T
    pub fn backwardA(grad_out: []const f32, b: []const f32, grad_a: []f32, m: usize, k: usize, n: usize) void {
        for (0..m) |i| {
            for (0..k) |l| {
                var s: f32 = 0;
                for (0..n) |j| s += grad_out[i * n + j] * b[l * n + j];
                grad_a[i * k + l] += s;
            }
        }
    }

    /// grad_b += a^T @ grad_out:  grad_b[k*n] += a[m*k]^T @ grad_out[m*n]
    pub fn backwardB(a: []const f32, grad_out: []const f32, grad_b: []f32, m: usize, k: usize, n: usize) void {
        for (0..k) |l| {
            for (0..n) |j| {
                var s: f32 = 0;
                for (0..m) |i| s += a[i * k + l] * grad_out[i * n + j];
                grad_b[l * n + j] += s;
            }
        }
    }
};

// ── unit tests ────────────────────────────────────────────────────────────────

const testing = std.testing;
const Optimizer = @import("../optimizer.zig").Optimizer;

fn approxEq(actual: f32, expected: f32, tol: f32) !void {
    if (@abs(actual - expected) > tol) {
        std.debug.print("expected ~{d}, got {d}\n", .{ expected, actual });
        return error.TestExpectedApproxEqual;
    }
}

test "CpuOptimizer: SGD param -= lr * grad, grads zeroed after" {
    const opt = Optimizer{ .kind = .sgd };
    var params = [_]f32{ 1.0, 2.0 };
    var grads  = [_]f32{ 0.1, 0.2 };
    var m      = [_]f32{ 0.0, 0.0 };
    var v      = [_]f32{ 0.0, 0.0 };
    CpuOptimizer.update(opt, 1, 0.1, &params, &grads, &m, &v);
    try approxEq(params[0], 0.99, 1e-6);
    try approxEq(params[1], 1.98, 1e-6);
    try approxEq(grads[0], 0.0, 1e-9);
    try approxEq(grads[1], 0.0, 1e-9);
}

test "CpuOptimizer: Adam first step close to -lr" {
    // After step 1 with grad=1: m_hat=1, v_hat=1 → update ≈ lr
    const opt = Optimizer{ .kind = .{ .adam = .{ .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8 } } };
    var params = [_]f32{ 0.0 };
    var grads  = [_]f32{ 1.0 };
    var m      = [_]f32{ 0.0 };
    var v      = [_]f32{ 0.0 };
    CpuOptimizer.update(opt, 1, 0.01, &params, &grads, &m, &v);
    try approxEq(params[0], -0.01, 1e-4);
}

test "CpuOptimizer: AdamW shrinks weight even with zero gradient" {
    // param *= (1 - lr*wd) = 2.0 * 0.999 = 1.998
    const opt = Optimizer{ .kind = .{ .adamw = .{
        .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8, .weight_decay = 0.1,
    } } };
    var params = [_]f32{ 2.0 };
    var grads  = [_]f32{ 0.0 };
    var m      = [_]f32{ 0.0 };
    var v      = [_]f32{ 0.0 };
    CpuOptimizer.update(opt, 1, 0.01, &params, &grads, &m, &v);
    try approxEq(params[0], 2.0 * (1.0 - 0.01 * 0.1), 1e-4);
}
