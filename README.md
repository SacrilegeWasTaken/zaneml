<p align="center">
  <img src="resources/zaneml-512.png" alt="zaneml" width="512" />
</p>

<h1 align="center">zaneml</h1>

<p align="center">
  A composable machine learning library for Zig 0.15 —<br/>
  backends, autograd, and a unified training interface.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/zig-0.15.2-orange" />
  <img src="https://img.shields.io/badge/license-MIT-blue" />
</p>

---

<p align="center">
  NOTE! This library will evolve slowly until zig 1.0
</p>

## Features

- **Unified training interface** — `Network.train` / `Network.predict` for every model type
- **Composable modules** — `Layer`, `Sequential`, `LayerNorm`, `RMSNorm`, `MultiHeadAttention`, `TransformerBlock`, `TransformerStack`, `Embedding`, `PositionalEmbedding`, `Dropout`, `Residual`
- **Tape-based autograd** — `Tape` records forward ops and replays them in reverse; `TapeMLP` wraps an arbitrary-depth MLP in the same Network interface
- **Optimizers** — SGD, Adam, AdamW with optional gradient clipping and LR schedules (constant, warmup + cosine)
- **Backend dispatch** — every compute kernel dispatches through a backend tag; CPU is fully implemented, Metal / ANE / Vulkan / CUDA are stub-ready

---

## Installation

Add zaneml as a dependency in your `build.zig.zon`:

```zig
.dependencies = .{
    .zaneml = .{
        .url  = "https://github.com/your-org/zaneml/archive/<commit>.tar.gz",
        .hash = "<hash>",
    },
},
```

Then wire it into your `build.zig`:

```zig
const zaneml = b.dependency("zaneml", .{
    .target   = target,
    .optimize = optimize,
});
exe.root_module.addImport("zaneml", zaneml.module("zaneml"));
```

---

## Quick start

### Perceptron — `Layer` + `Sequential`

```zig
const L   = zaneml.Layer(.cpu);
const Seq = zaneml.Sequential(struct { fc1: L, fc2: L });
const Net = zaneml.Network(*Seq);

var seq = try Seq.init(allocator, .{
    .fc1 = try L.init(allocator, .{ .n_in = 2, .n_out = 8, .activation = .relu }),
    .fc2 = try L.init(allocator, .{ .n_in = 8, .n_out = 1, .activation = .sigmoid }),
}, &.{8});
defer seq.deinit();

var net = Net.init(allocator, &seq);
try net.train(&samples, .{
    .lr = 0.1, .epochs = 5000, .log_every = 500,
    .optimizer = .{ .kind = .sgd }, .grad_clip = 0,
    .lr_schedule = .constant, .loss = .mse, .batch_size = 4,
});
```

### Transformer — `TransformerStack`

```zig
const Stack = zaneml.TransformerStack(.cpu, 16, 2, 32, 4, 3, .{
    .norm = .layer_norm, .ffn_activation = .silu, .causal = false,
});
const Net = zaneml.Network(*Stack);

var stack = try Stack.init(allocator);
defer stack.deinit();

var net = Net.init(allocator, &stack);
try net.train(&samples, .{
    .lr = 3e-4, .epochs = 2000, .log_every = 200,
    .optimizer = .{ .kind = .{ .adam = .{ .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-8 } } },
    .grad_clip = 1.0, .lr_schedule = .constant, .loss = .mse, .batch_size = 1,
});
```

### Autograd MLP — `TapeMLP`

```zig
const Model = zaneml.TapeMLP(.cpu, 2, &.{
    .{ .n_out = 8, .activation = .relu },
    .{ .n_out = 1, .activation = .sigmoid },
});
const Net = zaneml.Network(*Model);

var model = try Model.init(allocator);
defer model.deinit();

var net = Net.init(allocator, &model);
try net.train(&samples, .{
    .lr = 0.1, .epochs = 5000, .log_every = 500,
    .optimizer = .{ .kind = .sgd }, .grad_clip = 0,
    .lr_schedule = .constant, .loss = .mse, .batch_size = 4,
});
```

---

## Build & run

```sh
zig build perceptron     # XOR via Layer + Sequential
zig build transformer    # sequence inversion via TransformerStack
zig build autograd       # XOR via TapeMLP (tape-based autograd)

zig build test           # unit tests (inline) + integration tests (tests/)
zig build test-unit      # unit tests only
zig build test-integration  # integration tests only
```

---

## Backends

| Backend | Status  | Target                |
|---------|---------|-----------------------|
| `cpu`   | Full    | any                   |
| `metal` | Stub    | macOS / iOS           |
| `ane`   | Stub    | Apple Silicon (ANE)   |
| `vulkan`| Stub    | Linux / Windows       |
| `cuda`  | Stub    | NVIDIA                |

Select with the first comptime argument: `Layer(.metal)`, `TransformerStack(.ane, ...)`, etc.

---

## License

MIT
