const std = @import("std");

pub fn Tensor(comptime rank: usize) type {
    return struct {
        shape: [rank]usize,
        data:  []f32,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, shape: [rank]usize) !Self {
            return .{
                .shape = shape,
                .data  = try allocator.alloc(f32, numelOf(shape)),
            };
        }

        pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
        }

        pub fn from(data: []f32, shape: [rank]usize) Self {
            std.debug.assert(data.len == numelOf(shape));
            return .{ .shape = shape, .data = data };
        }

        pub fn numel(self: Self) usize {
            return numelOf(self.shape);
        }

        pub fn stride(self: Self, comptime axis: usize) usize {
            comptime std.debug.assert(axis < rank);
            var s: usize = 1;
            inline for (axis + 1..rank) |j| s *= self.shape[j];
            return s;
        }

        pub fn flatIndex(self: Self, indices: [rank]usize) usize {
            var idx: usize = 0;
            comptime var i = rank;
            inline while (i > 0) {
                i -= 1;
                idx += indices[i] * self.stride(i);
            }
            return idx;
        }

        pub fn at(self: Self, indices: [rank]usize) f32 {
            return self.data[self.flatIndex(indices)];
        }

        pub fn atPtr(self: Self, indices: [rank]usize) *f32 {
            return &self.data[self.flatIndex(indices)];
        }

        pub fn row(self: Self, i: usize) Tensor(rank - 1) {
            if (comptime rank <= 1) @compileError("row() requires rank >= 2");
            const s = self.stride(0);
            var sub: [rank - 1]usize = undefined;
            inline for (0..rank - 1) |j| sub[j] = self.shape[j + 1];
            return Tensor(rank - 1).from(self.data[i * s .. (i + 1) * s], sub);
        }


        pub fn fill(self: Self, val: f32) void {
            @memset(self.data, val);
        }

        pub fn zeros(self: Self) void {
            self.fill(0.0);
        }

        pub fn addInPlace(self: Self, other: Self) void {
            for (self.data, other.data) |*a, b| a.* += b;
        }

        pub fn addTo(out: Self, a: Self, b: Self) void {
            for (out.data, a.data, b.data) |*o, x, y| o.* = x + y;
        }

        pub fn scale(self: Self, s: f32) void {
            for (self.data) |*x| x.* *= s;
        }

        pub fn copyFrom(self: Self, src: Self) void {
            @memcpy(self.data, src.data);
        }


        pub fn print(self: Self, label: []const u8) void {
            std.debug.print("{s} shape=[", .{label});
            for (self.shape, 0..) |d, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{d}", .{d});
            }
            std.debug.print("]  data={d:.4}\n", .{self.data});
        }
    };
}

fn numelOf(shape: anytype) usize {
    var n: usize = 1;
    for (shape) |d| n *= d;
    return n;
}


