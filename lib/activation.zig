const std = @import("std");

pub const Activation = union(enum) {
    relu,
    leaky_relu: f32,
    elu: f32,
    sigmoid,
    tanh,
    linear,
};
