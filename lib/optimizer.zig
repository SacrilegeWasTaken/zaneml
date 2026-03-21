const std = @import("std");

/// Optimizer configuration passed to updateWeights on every module.
/// Adam moment buffers are stored inside each module; t (step) is passed externally.
pub const Optimizer = struct {
    kind: Kind,

    pub const Kind = union(enum) {
        sgd:   void,
        adam:  Adam,
        adamw: AdamW,
    };

    pub const Adam = struct {
        beta1: f32,
        beta2: f32,
        eps:   f32,
    };

    pub const AdamW = struct {
        beta1:        f32,
        beta2:        f32,
        eps:          f32,
        weight_decay: f32,
    };

    /// Apply in-place parameter update.
    /// m/v may be empty slices for SGD (ignored).
    /// Zeroes grads after update.
    pub fn update(
        self:   Optimizer,
        t:      usize,
        lr:     f32,
        params: []f32,
        grads:  []f32,
        m:      []f32,
        v:      []f32,
    ) void {
        switch (self.kind) {
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
                    // AdamW: weight decay applied directly to params, not grads
                    p.* = p.* * (1.0 - lr * wd) - lr * m_hat / (@sqrt(v_hat) + eps);
                }
            },
        }
        // Zero gradients after update
        @memset(grads, 0);
    }
};
