/// Metal compute smoke test — matrix multiply on GPU.
/// Run: zig build metal-test
const std = @import("std");
const MetalEngine = @import("zaneml").MetalEngine;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const alloc = gpa.allocator();

    var engine = try MetalEngine.init(alloc);
    defer engine.deinit();

    // Print device name
    const name = engine.deviceName();
    std.debug.print("Metal device: {s}\n", .{std.mem.sliceTo(&name, 0)});

    // ── Test 1: matmul_f32 ───────────────────────────────────────────────────
    // A = [1, 2]    B = [5, 6]    C = A×B = [19, 22]
    //     [3, 4]        [7, 8]              [43, 50]
    const M: u32 = 2;
    const K: u32 = 2;
    const N: u32 = 2;

    const a_data = [_]f32{ 1, 2, 3, 4 };
    const b_data = [_]f32{ 5, 6, 7, 8 };

    const buf_a = try engine.createBufferFromSlice(f32, &a_data);
    defer engine.releaseBuffer(buf_a);
    const buf_b = try engine.createBufferFromSlice(f32, &b_data);
    defer engine.releaseBuffer(buf_b);
    const buf_c = try engine.createBuffer(f32, M * N);
    defer engine.releaseBuffer(buf_c);

    const matmul_pipe = try engine.getPipeline("matmul_f32");

    const MatmulParams = extern struct { M: u32, K: u32, N: u32 };
    try engine.dispatch(
        matmul_pipe,
        &.{ buf_a, buf_b, buf_c },
        MatmulParams{ .M = M, .K = K, .N = N },
        .{ .x = N, .y = M },
        null,
    );

    const result = buf_c.asSlice(f32);
    std.debug.print("\nMatmul test: A[2×2] × B[2×2]\n", .{});
    std.debug.print("  C = [{d:.0}, {d:.0}]  (expected [19, 22])\n", .{ result[0], result[1] });
    std.debug.print("      [{d:.0}, {d:.0}]  (expected [43, 50])\n", .{ result[2], result[3] });

    const expected = [_]f32{ 19, 22, 43, 50 };
    for (result, expected) |got, exp| {
        if (@abs(got - exp) > 0.001) {
            std.debug.print("FAIL: got {d}, expected {d}\n", .{ got, exp });
            return error.TestFailed;
        }
    }
    std.debug.print("  ✓ PASS\n", .{});

    // ── Test 2: relu_forward ─────────────────────────────────────────────────
    const relu_data = [_]f32{ -2, -1, 0, 1, 2, 3 };
    const buf_relu_in = try engine.createBufferFromSlice(f32, &relu_data);
    defer engine.releaseBuffer(buf_relu_in);
    const buf_relu_out = try engine.createBuffer(f32, relu_data.len);
    defer engine.releaseBuffer(buf_relu_out);

    const relu_pipe = try engine.getPipeline("relu_forward");

    const ElementwiseParams = extern struct { length: u32 };
    try engine.dispatch(
        relu_pipe,
        &.{ buf_relu_in, buf_relu_out },
        ElementwiseParams{ .length = relu_data.len },
        .{ .x = relu_data.len },
        null,
    );

    const relu_result = buf_relu_out.asSlice(f32);
    std.debug.print("\nReLU test: [-2, -1, 0, 1, 2, 3]\n  → [", .{});
    for (relu_result, 0..) |v, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.0}", .{v});
    }
    std.debug.print("]  (expected [0, 0, 0, 1, 2, 3])\n", .{});

    const relu_expected = [_]f32{ 0, 0, 0, 1, 2, 3 };
    for (relu_result, relu_expected) |got, exp| {
        if (@abs(got - exp) > 0.001) return error.TestFailed;
    }
    std.debug.print("  ✓ PASS\n", .{});

    // ── Test 3: SGD update ───────────────────────────────────────────────────
    const params_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const grads_data = [_]f32{ 0.1, 0.2, 0.3, 0.4 };

    const buf_params = try engine.createBufferFromSlice(f32, &params_data);
    defer engine.releaseBuffer(buf_params);
    const buf_grads = try engine.createBufferFromSlice(f32, &grads_data);
    defer engine.releaseBuffer(buf_grads);

    const sgd_pipe = try engine.getPipeline("sgd_update");

    const SGDParams = extern struct { length: u32, lr: f32 };
    try engine.dispatch(
        sgd_pipe,
        &.{ buf_params, buf_grads },
        SGDParams{ .length = 4, .lr = 0.1 },
        .{ .x = 4 },
        null,
    );

    const sgd_result = buf_params.asSlice(f32);
    std.debug.print("\nSGD test: params=[1,2,3,4], grads=[0.1,0.2,0.3,0.4], lr=0.1\n  → [", .{});
    for (sgd_result, 0..) |v, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{d:.4}", .{v});
    }
    std.debug.print("]\n  (expected [0.99, 1.98, 2.97, 3.96])\n", .{});

    const sgd_expected = [_]f32{ 0.99, 1.98, 2.97, 3.96 };
    for (sgd_result, sgd_expected) |got, exp| {
        if (@abs(got - exp) > 0.001) return error.TestFailed;
    }
    std.debug.print("  ✓ PASS\n", .{});

    std.debug.print("\n═══ All Metal smoke tests passed! ═══\n", .{});
}
