/// XOR via a two-layer perceptron.
/// Run: zig build perceptron
const std = @import("std");
const zaneml = @import("zaneml");

const L   = zaneml.Layer(.cpu);
const Seq = zaneml.Sequential(struct { fc1: L, fc2: L });
const Net = zaneml.Network(.cpu, *Seq);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var seq = try Seq.init(allocator, .{
        .fc1 = try L.init(allocator, .{ .n_in = 2, .n_out = 8, .activation = .relu }),
        .fc2 = try L.init(allocator, .{ .n_in = 8, .n_out = 1, .activation = .sigmoid }),
    }, &.{8});
    defer seq.deinit();

    var net = Net.init(allocator, &seq);

    const samples = [_]Net.Sample{
        .{ .input = &.{ 0, 0 }, .target = &.{0} },
        .{ .input = &.{ 0, 1 }, .target = &.{1} },
        .{ .input = &.{ 1, 0 }, .target = &.{1} },
        .{ .input = &.{ 1, 1 }, .target = &.{0} },
    };

    try net.train(&samples, .{
        .lr          = 0.1,
        .epochs      = 5000,
        .log_every   = 500,
        .optimizer   = .{ .kind = .sgd },
        .grad_clip   = 0,
        .lr_schedule = .constant,
        .loss        = .mse,
    });

    std.debug.print("\n--- predictions ---\n", .{});
    var out = [1]f32{0};
    for (samples) |s| {
        net.predict(s.input, &out);
        std.debug.print("  {d} XOR {d}  =>  {d:.4}  (expected {d:.0})\n", .{
            @as(u1, @intFromFloat(s.input[0])),
            @as(u1, @intFromFloat(s.input[1])),
            out[0],
            s.target[0],
        });
    }
}
