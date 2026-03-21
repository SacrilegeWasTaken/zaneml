const cpu_b    = @import("backend/cpu.zig");
const metal_b  = @import("backend/metal.zig");
const ane_b    = @import("backend/ane.zig");
const vulkan_b = @import("backend/vulkan.zig");
const cuda_b   = @import("backend/cuda.zig");

pub const Backend = enum {
    cpu,
    metal,
    ane,
    cuda,
    vulkan,

    pub fn suggested() Backend {
        const builtin = @import("builtin");
        return switch (builtin.os.tag) {
            .macos, .ios => if (builtin.cpu.arch == .aarch64) .ane else .metal,
            .linux, .windows => .vulkan,
            else => .cpu,
        };
    }
};

pub fn BackendImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuBackend,
        .metal  => metal_b.MetalBackend,
        .ane    => ane_b.AneBackend,
        .vulkan => vulkan_b.VulkanBackend,
        .cuda   => cuda_b.CudaBackend,
    };
}

pub fn LayerNormImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuLayerNorm,
        .metal  => metal_b.MetalLayerNorm,
        .ane    => ane_b.AneLayerNorm,
        .vulkan => vulkan_b.VulkanLayerNorm,
        .cuda   => cuda_b.CudaLayerNorm,
    };
}

pub fn AttentionImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuAttention,
        .metal  => metal_b.MetalAttention,
        .ane    => ane_b.AneAttention,
        .vulkan => vulkan_b.VulkanAttention,
        .cuda   => cuda_b.CudaAttention,
    };
}
