const std = @import("std");

pub const Activation = @import("activation.zig").Activation;

const cpu_b    = @import("backend/cpu.zig");
const metal_b  = @import("backend/metal.zig");
const ane_b    = @import("backend/ane.zig");
const vulkan_b = @import("backend/vulkan.zig");
const cuda_b   = @import("backend/cuda.zig");

pub const Backend = enum {
    cpu,
    metal,
    ane,
    cuda,
    vulkan,

    pub fn suggested() Backend {
        const builtin = @import("builtin");
        return switch (builtin.os.tag) {
            .macos, .ios => if (builtin.cpu.arch == .aarch64) .ane else .metal,
            .linux, .windows => .vulkan,
            else => .cpu,
        };
    }
};

fn BackendImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuBackend,
        .metal  => metal_b.MetalBackend,
        .ane    => ane_b.AneBackend,
        .vulkan => vulkan_b.VulkanBackend,
        .cuda   => cuda_b.CudaBackend,
    };
}

pub fn Layer(comptime backend: Backend) type {
    return struct {
        weights: []f16,
        biases: []f32,
        weights_compute_buffer: []f32,
        pre_activation_buffer: []f32,
        grad_w: []f32,
        grad_b: []f32,
        n_in: usize,
        n_out: usize,
        activation: Activation,
        allocator: std.mem.Allocator,

        const Self = @This();
        const Impl = BackendImpl(backend);

        pub const Config = struct {
            n_in: usize,
            n_out: usize,
            activation: Activation,
        };

        pub fn init(allocator: std.mem.Allocator, config: Config) !Self {
            const n = config.n_in * config.n_out;

            const weights                = try allocator.alloc(f16,  n);
            const biases                 = try allocator.alloc(f32,  config.n_out);
            const weights_compute_buffer = try allocator.alloc(f32,  n);
            const pre_activation_buffer  = try allocator.alloc(f32,  config.n_out);
            const grad_w                 = try allocator.alloc(f32,  n);
            const grad_b                 = try allocator.alloc(f32,  config.n_out);

            var rng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
            const scale = @sqrt(2.0 / @as(f32, @floatFromInt(config.n_in)));
            for (weights) |*w| w.* = @floatCast(rng.random().floatNorm(f32) * scale);

            @memset(biases, 0.0);
            @memset(grad_w, 0.0);
            @memset(grad_b, 0.0);

            return .{
                .weights                = weights,
                .biases                 = biases,
                .weights_compute_buffer = weights_compute_buffer,
                .pre_activation_buffer  = pre_activation_buffer,
                .grad_w                 = grad_w,
                .grad_b                 = grad_b,
                .n_in                   = config.n_in,
                .n_out                  = config.n_out,
                .activation             = config.activation,
                .allocator              = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.weights);
            self.allocator.free(self.biases);
            self.allocator.free(self.weights_compute_buffer);
            self.allocator.free(self.pre_activation_buffer);
            self.allocator.free(self.grad_w);
            self.allocator.free(self.grad_b);
        }

        fn syncWeights(self: *Self) void {
            for (self.weights, self.weights_compute_buffer) |w, *wf| wf.* = @floatCast(w);
        }

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

        pub fn updateWeights(self: *Self, lr: f32) void {
            for (self.weights_compute_buffer, self.grad_w) |*wf, gw| wf.* -= lr * gw;
            for (self.biases, self.grad_b) |*b, gb| b.* -= lr * gb;
            for (self.weights, self.weights_compute_buffer) |*w, wf| w.* = @floatCast(wf);
            @memset(self.grad_w, 0.0);
            @memset(self.grad_b, 0.0);
        }
    };
}

pub fn Network(comptime backend: Backend) type {
    return struct {
        layers: []Layer(backend),
        allocator: std.mem.Allocator,

        const Self = @This();
        const L = Layer(backend);

        pub const Sample = struct {
            input: []const f32,
            target: []const f32,
        };

        pub const TrainConfig = struct {
            lr: f32 = 0.01,
            epochs: usize = 1000,
            log_every: usize = 100,
        };

        pub fn init(allocator: std.mem.Allocator, configs: []const L.Config) !Self {
            const layers = try allocator.alloc(L, configs.len);
            errdefer allocator.free(layers);
            var n: usize = 0;
            errdefer for (layers[0..n]) |*l| l.deinit();
            for (layers, configs) |*layer, cfg| {
                layer.* = try L.init(allocator, cfg);
                n += 1;
            }
            return .{ .layers = layers, .allocator = allocator };
        }

        pub fn deinit(self: *Self) void {
            for (self.layers) |*l| l.deinit();
            self.allocator.free(self.layers);
        }

        /// Прогнать вход через сеть. `out` — размером n_out последнего слоя.
        pub fn predict(self: *Self, input: []const f32, out: []f32) !void {
            const n = self.layers.len;
            std.debug.assert(n > 0);

            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();
            const tmp = arena.allocator();

            var cur: []const f32 = input;
            for (self.layers[0 .. n - 1]) |*layer| {
                const buf = try tmp.alloc(f32, layer.n_out);
                layer.forward(cur, buf);
                cur = buf;
            }
            self.layers[n - 1].forward(cur, out);
        }

        /// SGD по всем сэмплам за каждую эпоху. Лосс — MSE.
        pub fn train(self: *Self, samples: []const Sample, config: TrainConfig) !void {
            const n = self.layers.len;
            std.debug.assert(n > 0);

            var arena = std.heap.ArenaAllocator.init(self.allocator);
            defer arena.deinit();
            const tmp = arena.allocator();

            // acts[0] = копия входа, acts[i+1] = выход слоя i
            const acts = try tmp.alloc([]f32, n + 1);
            acts[0] = try tmp.alloc(f32, self.layers[0].n_in);
            for (self.layers, 0..) |layer, i| acts[i + 1] = try tmp.alloc(f32, layer.n_out);

            // grads[i] = dL/d(вход слоя i)
            const grads = try tmp.alloc([]f32, n + 1);
            grads[0] = try tmp.alloc(f32, self.layers[0].n_in);
            for (self.layers, 0..) |layer, i| grads[i + 1] = try tmp.alloc(f32, layer.n_out);

            for (0..config.epochs) |epoch| {
                var total_loss: f32 = 0.0;

                for (samples) |s| {
                    @memcpy(acts[0], s.input);
                    for (self.layers, 0..) |*layer, i| layer.forward(acts[i], acts[i + 1]);

                    // MSE: dL/dy = 2*(pred - target) / n_out
                    const pred = acts[n];
                    const inv = 1.0 / @as(f32, @floatFromInt(pred.len));
                    for (grads[n], pred, s.target) |*g, p, t| {
                        const d = p - t;
                        total_loss += d * d;
                        g.* = 2.0 * d * inv;
                    }

                    var i: usize = n;
                    while (i > 0) : (i -= 1) {
                        try self.layers[i - 1].backward(acts[i - 1], grads[i], grads[i - 1]);
                        self.layers[i - 1].updateWeights(config.lr);
                    }
                }

                if (config.log_every > 0 and (epoch + 1) % config.log_every == 0) {
                    const avg = total_loss / @as(f32, @floatFromInt(samples.len));
                    std.log.info("epoch {d:>6}  loss = {d:.6}", .{ epoch + 1, avg });
                }
            }
        }
    };
}
