const std = @import("std");

pub const Activation = union(enum) {
    relu,
    leaky_relu: f32,
    elu: f32,
    sigmoid,
    tanh,
    linear,
    /// Gaussian Error Linear Unit (tanh approximation)
    gelu,
    /// Sigmoid Linear Unit: x * sigmoid(x)
    silu,
};
