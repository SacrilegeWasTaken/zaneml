const std = @import("std");
const backend_mod   = @import("backend.zig");
pub const Backend   = backend_mod.Backend;
const optimizer_mod = @import("optimizer.zig");
pub const Optimizer = optimizer_mod.Optimizer;

/// Learning rate schedule.
pub const LRSchedule = union(enum) {
    /// Constant learning rate.
    constant: void,
    /// Linear warmup followed by cosine decay.
    warmup_cosine: struct { warmup: usize, total: usize },

    /// Return the effective learning rate at a given step.
    pub fn get(self: LRSchedule, base_lr: f32, step: usize) f32 {
        return switch (self) {
            .constant => base_lr,
            .warmup_cosine => |cfg| blk: {
                if (step < cfg.warmup) {
                    // Linear warmup: 0 -> base_lr
                    const t: f32 = @as(f32, @floatFromInt(step)) / @as(f32, @floatFromInt(cfg.warmup));
                    break :blk base_lr * t;
                } else {
                    // Cosine decay: base_lr -> 0
                    const progress = @as(f32, @floatFromInt(step - cfg.warmup)) /
                                     @as(f32, @floatFromInt(cfg.total - cfg.warmup));
                    break :blk base_lr * 0.5 * (1.0 + @cos(std.math.pi * progress));
                }
            },
        };
    }
};

/// Loss function selection.
pub const Loss = enum {
    mse,
    cross_entropy,
};

/// Training configuration. All fields must be set explicitly.
pub const TrainConfig = struct {
    lr:          f32,
    epochs:      usize,
    log_every:   usize,
    optimizer:   Optimizer,
    /// Global gradient norm clipping threshold (0 = disabled).
    grad_clip:   f32,
    lr_schedule: LRSchedule,
    loss:        Loss,
    batch_size:  usize,
};

/// Generic wrapper over any module.
///
/// `backend` selects which backend implementation to use for loss and optimizer compute.
/// `Model` must be a pointer to a type implementing:
///   pub fn forward      (self, input: []const f32, output: []f32) void
///   pub fn backward     (self, input: []const f32, grad_out: []const f32, grad_in: []f32) [!]void
///   pub fn updateWeights(self, opt: Optimizer, lr: f32, t: usize) void
///
/// Optionally the model may implement:
///   pub fn gradNormSq(self: *const Self) f32
///   pub fn scaleGrads(self: *Self, s: f32) void
/// These are used for gradient clipping if grad_clip > 0.
pub fn Network(comptime Model: type) type {
    comptime {
        if (@typeInfo(Model) != .pointer)
            @compileError("Network: Model must be a pointer, e.g. Network(*MyModel)");
    }
    const ModelBase_ = @typeInfo(Model).pointer.child;
    const backend_: Backend = if (@hasDecl(ModelBase_, "backend_tag"))
        ModelBase_.backend_tag
    else
        .cpu;
    const LossImpl = backend_mod.LossImpl(backend_);

    return struct {
        model:     Model,
        allocator: std.mem.Allocator,

        const Self = @This();
        const ModelBase = @typeInfo(Model).pointer.child;

        pub const Sample = struct {
            input:  []const f32,
            target: []const f32,
        };

        /// Initialize the Network wrapper (does not take ownership of model).
        pub fn init(allocator: std.mem.Allocator, model: Model) Self {
            return .{ .model = model, .allocator = allocator };
        }

        /// Run a single forward pass.
        pub fn forward(self: *Self, input: []const f32, output: []f32) void {
            self.model.forward(input, output);
        }

        /// Alias for forward -- used during inference.
        pub fn predict(self: *Self, input: []const f32, output: []f32) void {
            self.model.forward(input, output);
        }

        /// Train over all samples for the configured number of epochs.
        /// All samples must have the same input.len / target.len.
        pub fn train(self: *Self, samples: []const Sample, config: TrainConfig) !void {
            if (samples.len == 0) return;

            const out_size = samples[0].target.len;
            const in_size  = samples[0].input.len;

            // Metal batch path: stack a full batch into one forward/backward pass.
            // Requires the model to expose batchForward/batchBackward (TransformerStack on Metal).
            if (comptime backend_ == .metal and @hasDecl(ModelBase, "batchForward")) {
                try self.trainBatched(samples, config, in_size, out_size);
                return;
            }

            const output   = try self.allocator.alloc(f32, out_size);
            defer self.allocator.free(output);
            const grad_out = try self.allocator.alloc(f32, out_size);
            defer self.allocator.free(grad_out);
            const grad_in  = try self.allocator.alloc(f32, in_size);
            defer self.allocator.free(grad_in);

            const inv = 1.0 / @as(f32, @floatFromInt(out_size));

            // Global step counter for Adam bias correction (1-based)
            var t: usize = 0;

            for (0..config.epochs) |epoch| {
                var total_loss: f32 = 0;

                var si: usize = 0;
                while (si < samples.len) : (si += config.batch_size) {
                    const batch_end = @min(si + config.batch_size, samples.len);
                    const batch     = samples[si..batch_end];
                    const bs: f32   = @floatFromInt(batch_end - si);

                    t += 1;
                    const lr = config.lr_schedule.get(config.lr, t);

                    // Accumulate gradients over the batch
                    for (batch) |s| {
                        self.model.forward(s.input, output);
                        total_loss += switch (config.loss) {
                            .mse           => LossImpl.mse(output, s.target, grad_out, inv),
                            .cross_entropy => LossImpl.cross_entropy(output, s.target, grad_out, inv),
                        };
                        try callBackward(self.model, s.input, grad_out, grad_in);
                    }

                    // Average gradients over batch
                    if (batch.len > 1) {
                        if (comptime @hasDecl(ModelBase, "scaleGrads")) {
                            self.model.scaleGrads(1.0 / bs);
                        }
                    }

                    // Gradient clipping (global norm)
                    if (config.grad_clip > 0) {
                        if (comptime @hasDecl(ModelBase, "gradNormSq") and
                                    @hasDecl(ModelBase, "scaleGrads")) {
                            const norm_sq = self.model.gradNormSq();
                            const norm = @sqrt(norm_sq);
                            if (norm > config.grad_clip) {
                                self.model.scaleGrads(config.grad_clip / norm);
                            }
                        }
                    }

                    self.model.updateWeights(config.optimizer, lr, t);
                }

                if (config.log_every > 0 and (epoch + 1) % config.log_every == 0) {
                    const n: f32 = @floatFromInt(samples.len * out_size);
                    std.log.info("epoch {d:>6}  loss = {d:.6}", .{ epoch + 1, total_loss / n });
                }
            }
        }

        /// Metal batch training: stacks each batch into one forward+backward pass,
        /// reducing command buffer overhead from (2*batch_size + 1) to 3 per step.
        fn trainBatched(self: *Self, samples: []const Sample, config: TrainConfig, in_size: usize, out_size: usize) !void {
            // Global step counter for Adam bias correction (1-based)
            var t: usize = 0;

            var si: usize = 0;
            // Allocate stacked buffers for the largest batch
            const max_bs = @min(config.batch_size, samples.len);
            const stacked_input  = try self.allocator.alloc(f32, max_bs * in_size);
            defer self.allocator.free(stacked_input);
            const stacked_output = try self.allocator.alloc(f32, max_bs * out_size);
            defer self.allocator.free(stacked_output);
            const stacked_target = try self.allocator.alloc(f32, max_bs * out_size);
            defer self.allocator.free(stacked_target);
            const stacked_grad   = try self.allocator.alloc(f32, max_bs * out_size);
            defer self.allocator.free(stacked_grad);
            const stacked_gin    = try self.allocator.alloc(f32, max_bs * in_size);
            defer self.allocator.free(stacked_gin);

            for (0..config.epochs) |epoch| {
                var total_loss: f32 = 0;

                si = 0;
                while (si < samples.len) : (si += config.batch_size) {
                    const batch_end = @min(si + config.batch_size, samples.len);
                    const batch     = samples[si..batch_end];
                    const bs        = batch_end - si;
                    const bs_f: f32 = @floatFromInt(bs);

                    t += 1;
                    const lr = config.lr_schedule.get(config.lr, t);

                    // Stack inputs and targets
                    for (batch, 0..) |s, j| {
                        @memcpy(stacked_input [j * in_size  ..][0..in_size ], s.input);
                        @memcpy(stacked_target[j * out_size ..][0..out_size], s.target);
                    }

                    // Single forward pass over the stacked batch
                    self.model.batchForward(
                        stacked_input[0..bs * in_size],
                        stacked_output[0..bs * out_size],
                        bs,
                    );

                    // Compute loss and grad on CPU — data is already there after batchForward.
                    // Use inv_batch so each element's gradient is normalised over the full batch.
                    // This avoids bs separate GPU command buffer submissions for the loss kernel.
                    const CpuLoss = @import("backend/cpu.zig").CpuLoss;
                    const inv_batch = 1.0 / @as(f32, @floatFromInt(bs * out_size));
                    total_loss += CpuLoss.mse(
                        stacked_output[0..bs * out_size],
                        stacked_target[0..bs * out_size],
                        stacked_grad[0..bs * out_size],
                        inv_batch,
                    );

                    // Single backward pass
                    self.model.batchBackward(
                        stacked_grad[0..bs * out_size],
                        stacked_gin[0..bs * in_size],
                        bs,
                    );

                    // Gradient clipping (global norm) — grads are on CPU after downloadGrads
                    if (config.grad_clip > 0) {
                        if (comptime @hasDecl(ModelBase, "gradNormSq") and
                                    @hasDecl(ModelBase, "scaleGrads")) {
                            const norm_sq = self.model.gradNormSq();
                            const norm    = @sqrt(norm_sq);
                            if (norm > config.grad_clip) {
                                self.model.scaleGrads(config.grad_clip / norm);
                            }
                        }
                    }

                    self.model.updateWeights(config.optimizer, lr, t);
                    _ = bs_f;
                }

                if (config.log_every > 0 and (epoch + 1) % config.log_every == 0) {
                    const n: f32 = @floatFromInt(samples.len * out_size);
                    std.log.info("epoch {d:>6}  loss = {d:.6}", .{ epoch + 1, total_loss / n });
                }
            }
        }
    };
}

//  unit tests 

const testing = std.testing;

test "LRSchedule: constant always returns base_lr" {
    const sched: LRSchedule = .constant;
    try testing.expectApproxEqAbs(@as(f32, 0.01), sched.get(0.01,    0), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, 0.01), sched.get(0.01, 1000), 1e-7);
}

test "LRSchedule: warmup_cosine linearly ramps during warmup" {
    const sched: LRSchedule = .{ .warmup_cosine = .{ .warmup = 10, .total = 100 } };
    try testing.expectApproxEqAbs(@as(f32, 0.0), sched.get(1.0, 0), 1e-6); // step 0 → 0
    try testing.expectApproxEqAbs(@as(f32, 0.5), sched.get(1.0, 5), 1e-6); // halfway → 0.5
    try testing.expectApproxEqAbs(@as(f32, 1.0), sched.get(1.0, 10), 1e-6); // end of warmup → 1
}

/// Comptime-safe backward call: handles both void and !void return types.
fn callBackward(model: anytype, input: []const f32, grad_out: []const f32, grad_in: []f32) !void {
    const T = @TypeOf(model);
    const Base = @typeInfo(T).pointer.child;
    const ret = @typeInfo(@TypeOf(Base.backward)).@"fn".return_type.?;
    if (comptime @typeInfo(ret) == .error_union) {
        try model.backward(input, grad_out, grad_in);
    } else {
        model.backward(input, grad_out, grad_in);
    }
}
