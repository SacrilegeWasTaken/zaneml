/// Optimizer configuration passed to updateWeights on every module.
/// Adam moment buffers are stored inside each module; t (step) is passed externally.
pub const Optimizer = struct {
    kind: Kind,

    pub const Kind = union(enum) {
        sgd:   void,
        adam:  Adam,
        adamw: AdamW,
    };

    pub const Adam = struct {
        beta1: f32,
        beta2: f32,
        eps:   f32,
    };

    pub const AdamW = struct {
        beta1:        f32,
        beta2:        f32,
        eps:          f32,
        weight_decay: f32,
    };

};
