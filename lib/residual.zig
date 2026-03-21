const std = @import("std");
const optimizer_mod = @import("optimizer.zig");
const Optimizer = optimizer_mod.Optimizer;

/// Residual (skip) connection: out = Inner(x) + x
///
/// Requirements for Inner:
///   - pub fn forward(self: *Inner, input: []const f32, out: []f32) void
///   - pub fn backward(self: *Inner, input: []const f32, grad_out: []const f32, grad_in: []f32) [!]void
///   - pub fn updateWeights(self: *Inner, opt: Optimizer, lr: f32, t: usize) void
///
/// `dim` -- input and output size (must match -- residual invariant).
///
/// Residual does not accept a backend: it is a combinator over Inner, not a
/// compute primitive. Inner already dispatches to the appropriate backend.
pub fn Residual(comptime dim: usize, comptime Inner: type) type {
    return struct {
        inner: Inner,
        /// Temporary buffer: grad_in from inner before adding the identity gradient.
        grad_scratch: [dim]f32,

        const Self = @This();

        /// Initialize with a pre-constructed inner module.
        pub fn init(inner: Inner) Self {
            return .{
                .inner = inner,
                .grad_scratch = [_]f32{0} ** dim,
            };
        }

        /// Forward pass: out = inner(input) + input
        pub fn forward(self: *Self, input: []const f32, out: []f32) void {
            std.debug.assert(input.len == dim and out.len == dim);
            self.inner.forward(input, out);
            for (out, input) |*o, x| o.* += x;
        }

        /// Backward pass through the residual connection.
        /// dL/dx = dL/d(inner_out) * inner'(x)  +  dL/d(out)  (identity path)
        pub fn backward(self: *Self, input: []const f32, grad_out: []const f32, grad_in: []f32) !void {
            std.debug.assert(input.len == dim and grad_out.len == dim and grad_in.len == dim);
            try callBackward(&self.inner, input, grad_out, &self.grad_scratch);
            for (grad_in, &self.grad_scratch, grad_out) |*gi, gs, go| gi.* = gs + go;
        }

        /// Update inner module weights.
        pub fn updateWeights(self: *Self, opt: Optimizer, lr: f32, t: usize) void {
            self.inner.updateWeights(opt, lr, t);
        }

        /// Sum of squared gradients from inner.
        pub fn gradNormSq(self: *const Self) f32 {
            if (comptime @hasDecl(Inner, "gradNormSq")) {
                return self.inner.gradNormSq();
            }
            return 0;
        }

        /// Scale inner module gradients by s.
        pub fn scaleGrads(self: *Self, s: f32) void {
            if (comptime @hasDecl(Inner, "scaleGrads")) {
                self.inner.scaleGrads(s);
            }
        }

        /// Comptime-safe call to inner.backward: handles void and !void.
        fn callBackward(inner: *Inner, input: []const f32, grad_out: []const f32, grad_in: []f32) !void {
            const ret_type = @typeInfo(@TypeOf(Inner.backward)).@"fn".return_type.?;
            if (comptime @typeInfo(ret_type) == .error_union) {
                try inner.backward(input, grad_out, grad_in);
            } else {
                inner.backward(input, grad_out, grad_in);
            }
        }
    };
}
