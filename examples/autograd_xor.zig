/// XOR via a tape-based MLP — gradients computed automatically.
/// Run: zig build autograd
const std    = @import("std");
const zaneml = @import("zaneml");

const Model = zaneml.TapeMLP(.cpu, 2, &.{
    .{ .n_out = 8, .activation = .relu },
    .{ .n_out = 1, .activation = .sigmoid },
});
const Net = zaneml.Network(.cpu, *Model);

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var model = try Model.init(allocator);
    defer model.deinit();

    var net = Net.init(allocator, &model);

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
        .batch_size  = 4,
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
