/// A fixed-size block of Adam moment vectors for N parameters.
/// Used as a field inside modules.
pub fn MomentBuf(comptime n: usize) type {
    return struct {
        m: [n]f32,
        v: [n]f32,

        pub fn init() @This() {
            return .{
                .m = [_]f32{0} ** n,
                .v = [_]f32{0} ** n,
            };
        }
    };
}
