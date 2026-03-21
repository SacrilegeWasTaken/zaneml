const std = @import("std");
const optimizer_mod = @import("optimizer.zig");
const Optimizer = optimizer_mod.Optimizer;

/// Linear stack of modules: out = Mn(...M1(M0(input)...))
///
/// `Modules` is a struct type (or tuple) where each field is a module instance.
/// Example:
///   const FFN = Sequential(struct {
///       a: Layer(.cpu),
///       b: Layer(.cpu),
///   });
///   var ffn = try FFN.init(allocator, .{
///       .a = try Layer(.cpu).init(allocator, .{ .n_in = 256, .n_out = 512, .activation = .relu }),
///       .b = try Layer(.cpu).init(allocator, .{ .n_in = 512, .n_out = 256, .activation = .linear }),
///   }, &.{512});   // intermediate sizes
///
/// `intermediate_sizes[i]` = output size of module i (= input size of module i+1),
/// for i in 0..n-2.  If there is only one module, pass `&.{}`.
pub fn Sequential(comptime Modules: type) type {
    const fields = @typeInfo(Modules).@"struct".fields;
    const n = fields.len;
    comptime std.debug.assert(n >= 1);

    return struct {
        modules:   Modules,
        /// Intermediate activation and gradient buffers (n-1 of each).
        /// When n==1 both arrays have zero length -- no allocations.
        act_bufs:  [n - 1][]f32,
        grad_bufs: [n - 1][]f32,
        allocator: std.mem.Allocator,

        const Self = @This();

        /// Initialize Sequential and allocate intermediate buffers.
        pub fn init(
            allocator: std.mem.Allocator,
            modules: Modules,
            intermediate_sizes: []const usize,
        ) !Self {
            std.debug.assert(intermediate_sizes.len == n - 1);

            var self: Self = undefined;
            self.modules   = modules;
            self.allocator = allocator;

            var init_count: usize = 0;
            errdefer for (0..init_count) |i| {
                allocator.free(self.act_bufs[i]);
                allocator.free(self.grad_bufs[i]);
            };

            inline for (0..n - 1) |i| {
                const sz = intermediate_sizes[i];
                self.act_bufs[i]  = try allocator.alloc(f32, sz);
                self.grad_bufs[i] = try allocator.alloc(f32, sz);
                init_count += 1;
            }

            return self;
        }

        /// Free intermediate buffers and call deinit on owned modules.
        pub fn deinit(self: *Self) void {
            // Free modules that have a no-argument deinit.
            // Pointer-type modules (*MHA etc.) are owned by the caller.
            inline for (fields) |f| {
                if (comptime @hasDecl(f.type, "deinit")) {
                    @field(self.modules, f.name).deinit();
                }
            }
            inline for (0..n - 1) |i| {
                self.allocator.free(self.act_bufs[i]);
                self.allocator.free(self.grad_bufs[i]);
            }
        }

        /// Forward pass: input -> M0 -> buf[0] -> M1 -> ... -> out
        pub fn forward(self: *Self, input: []const f32, out: []f32) void {
            inline for (0..n) |i| {
                const cur_in:  []const f32 = if (i == 0)     input             else self.act_bufs[i - 1];
                const cur_out: []f32       = if (i == n - 1) out               else self.act_bufs[i];
                @field(self.modules, fields[i].name).forward(cur_in, cur_out);
            }
        }

        /// Backward pass in reverse order.
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) !void {
            inline for (0..n) |rev| {
                const i = n - 1 - rev;
                const layer_input:    []const f32 = if (i == 0)     input             else self.act_bufs[i - 1];
                const layer_grad_out: []const f32 = if (i == n - 1) grad_out          else self.grad_bufs[i];
                const layer_grad_in:  []f32       = if (i == 0)     grad_in           else self.grad_bufs[i - 1];
                try callBackward(&@field(self.modules, fields[i].name), layer_input, layer_grad_out, layer_grad_in);
            }
        }

        /// Update all module weights.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            inline for (fields) |f| @field(self.modules, f.name).updateWeights(opt, lr, t);
        }

        /// Sum of squared gradients over all modules.
        pub fn gradNormSq(self: *const Self) f32 {
            var sum: f32 = 0;
            inline for (fields) |f| {
                const m = @field(self.modules, f.name);
                if (comptime @hasDecl(@TypeOf(m), "gradNormSq")) {
                    sum += m.gradNormSq();
                }
            }
            return sum;
        }

        /// Scale gradients of all modules by s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            inline for (fields) |f| {
                if (comptime @hasDecl(f.type, "scaleGrads")) {
                    @field(self.modules, f.name).scaleGrads(s);
                }
            }
        }
    };
}

/// Comptime-safe backward call: handles both void and !void return types.
fn callBackward(module: anytype, input: []const f32, grad_out: []const f32, grad_in: []f32) !void {
    const T = @TypeOf(module.*);
    const ret = @typeInfo(@TypeOf(T.backward)).@"fn".return_type.?;
    if (comptime @typeInfo(ret) == .error_union) {
        try module.backward(input, grad_out, grad_in);
    } else {
        module.backward(input, grad_out, grad_in);
    }
}
