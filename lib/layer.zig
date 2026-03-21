pub const std = @import("std");
pub const Activation = @import("activation.zig").Activation;
const optimizer_mod = @import("optimizer.zig");
pub const Optimizer = optimizer_mod.Optimizer;

pub const __b = @import("backend.zig");

pub fn Layer(comptime backend: __b.Backend) type {
    return struct {
        weights: []f16,
        biases: []f32,
        weights_compute_buffer: []f32,
        pre_activation_buffer: []f32,
        grad_w: []f32,
        grad_b: []f32,
        /// Adam moment buffers for weights
        m_w: []f32,
        v_w: []f32,
        /// Adam moment buffers for biases
        m_b: []f32,
        v_b: []f32,
        n_in: usize,
        n_out: usize,
        activation: Activation,
        allocator: std.mem.Allocator,

        const Self    = @This();
        const Impl    = __b.BackendImpl(backend);
        const OptImpl = __b.OptimizerImpl(backend);

        pub const Config = struct {
            n_in: usize,
            n_out: usize,
            activation: Activation,
        };

        /// Allocate and initialize a Layer. Weights use He initialization.
        pub fn init(allocator: std.mem.Allocator, config: Config) !Self {
            const n = config.n_in * config.n_out;

            const weights                = try allocator.alloc(f16,  n);
            errdefer allocator.free(weights);
            const biases                 = try allocator.alloc(f32,  config.n_out);
            errdefer allocator.free(biases);
            const weights_compute_buffer = try allocator.alloc(f32,  n);
            errdefer allocator.free(weights_compute_buffer);
            const pre_activation_buffer  = try allocator.alloc(f32,  config.n_out);
            errdefer allocator.free(pre_activation_buffer);
            const grad_w                 = try allocator.alloc(f32,  n);
            errdefer allocator.free(grad_w);
            const grad_b                 = try allocator.alloc(f32,  config.n_out);
            errdefer allocator.free(grad_b);
            const m_w                    = try allocator.alloc(f32,  n);
            errdefer allocator.free(m_w);
            const v_w                    = try allocator.alloc(f32,  n);
            errdefer allocator.free(v_w);
            const m_b                    = try allocator.alloc(f32,  config.n_out);
            errdefer allocator.free(m_b);
            const v_b                    = try allocator.alloc(f32,  config.n_out);
            errdefer allocator.free(v_b);

            var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
            const scale = @sqrt(2.0 / @as(f32, @floatFromInt(config.n_in)));
            for (weights) |*w| w.* = @floatCast(rng.random().floatNorm(f32) * scale);

            @memset(biases, 0.0);
            @memset(grad_w, 0.0);
            @memset(grad_b, 0.0);
            @memset(m_w, 0.0);
            @memset(v_w, 0.0);
            @memset(m_b, 0.0);
            @memset(v_b, 0.0);

            return .{
                .weights                = weights,
                .biases                 = biases,
                .weights_compute_buffer = weights_compute_buffer,
                .pre_activation_buffer  = pre_activation_buffer,
                .grad_w                 = grad_w,
                .grad_b                 = grad_b,
                .m_w                    = m_w,
                .v_w                    = v_w,
                .m_b                    = m_b,
                .v_b                    = v_b,
                .n_in                   = config.n_in,
                .n_out                  = config.n_out,
                .activation             = config.activation,
                .allocator              = allocator,
            };
        }

        /// Free all allocated memory.
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.weights);
            self.allocator.free(self.biases);
            self.allocator.free(self.weights_compute_buffer);
            self.allocator.free(self.pre_activation_buffer);
            self.allocator.free(self.grad_w);
            self.allocator.free(self.grad_b);
            self.allocator.free(self.m_w);
            self.allocator.free(self.v_w);
            self.allocator.free(self.m_b);
            self.allocator.free(self.v_b);
        }

        fn syncWeights(self: *Self) void {
            for (self.weights, self.weights_compute_buffer) |w, *wf| wf.* = @floatCast(w);
        }

        /// Forward pass: output = activation(W * input + b).
        pub fn forward(self: *Self, input: []const f32, out: []f32) void {
            self.syncWeights();
            Impl.forward(
                self.weights_compute_buffer,
                input,
                self.biases,
                self.pre_activation_buffer,
                out,
                self.activation,
            );
        }

        /// Backward pass: accumulates grad_w and grad_b, writes grad_in.
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) !void {
            try Impl.backward(
                self.allocator,
                self.weights_compute_buffer,
                input,
                self.biases,
                self.pre_activation_buffer,
                grad_out,
                grad_in,
                self.grad_w,
                self.grad_b,
                self.activation,
            );
        }

        /// Update weights using the given optimizer. t is the 1-based step counter.
        /// Zeroes gradients after update.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            // Convert f16 weights to f32 for update
            for (self.weights, self.weights_compute_buffer) |w, *wf| wf.* = @floatCast(w);

            OptImpl.update(opt, t, lr, self.weights_compute_buffer, self.grad_w, self.m_w, self.v_w);
            OptImpl.update(opt, t, lr, self.biases, self.grad_b, self.m_b, self.v_b);

            // Sync f32 -> f16
            for (self.weights, self.weights_compute_buffer) |*w, wf| w.* = @floatCast(wf);
        }

        /// Returns the sum of squared gradients for weights and biases.
        pub fn gradNormSq(self: *const Self) f32 {
            var sum: f32 = 0;
            for (self.grad_w) |g| sum += g * g;
            for (self.grad_b) |g| sum += g * g;
            return sum;
        }

        /// Multiply all gradients by the scalar s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            for (self.grad_w) |*g| g.* *= s;
            for (self.grad_b) |*g| g.* *= s;
        }
    };
}
