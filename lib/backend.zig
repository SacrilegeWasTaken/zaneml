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

pub fn OptimizerImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuOptimizer,
        .metal  => metal_b.MetalOptimizer,
        .ane    => ane_b.AneOptimizer,
        .vulkan => vulkan_b.VulkanOptimizer,
        .cuda   => cuda_b.CudaOptimizer,
    };
}

pub fn LossImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuLoss,
        .metal  => metal_b.MetalLoss,
        .ane    => ane_b.AneLoss,
        .vulkan => vulkan_b.VulkanLoss,
        .cuda   => cuda_b.CudaLoss,
    };
}

pub fn RMSNormImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuRMSNorm,
        .metal  => metal_b.MetalRMSNorm,
        .ane    => ane_b.AneRMSNorm,
        .vulkan => vulkan_b.VulkanRMSNorm,
        .cuda   => cuda_b.CudaRMSNorm,
    };
}

pub fn PositionalEmbeddingImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuPositionalEmbedding,
        .metal  => metal_b.MetalPositionalEmbedding,
        .ane    => ane_b.AnePositionalEmbedding,
        .vulkan => vulkan_b.VulkanPositionalEmbedding,
        .cuda   => cuda_b.CudaPositionalEmbedding,
    };
}

pub fn EmbeddingImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuEmbedding,
        .metal  => metal_b.MetalEmbedding,
        .ane    => ane_b.AneEmbedding,
        .vulkan => vulkan_b.VulkanEmbedding,
        .cuda   => cuda_b.CudaEmbedding,
    };
}

pub fn DropoutImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuDropout,
        .metal  => metal_b.MetalDropout,
        .ane    => ane_b.AneDropout,
        .vulkan => vulkan_b.VulkanDropout,
        .cuda   => cuda_b.CudaDropout,
    };
}

pub fn MatmulImpl(comptime backend: Backend) type {
    return switch (backend) {
        .cpu    => cpu_b.CpuMatmul,
        .metal  => metal_b.MetalMatmul,
        .ane    => ane_b.AneMatmul,
        .vulkan => vulkan_b.VulkanMatmul,
        .cuda   => cuda_b.CudaMatmul,
    };
}
