#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "bridge.h"

//  Device & queue 

MTLDeviceRef zml_metal_create_device(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (__bridge_retained void*)device;
}

MTLCommandQueueRef zml_metal_create_command_queue(MTLDeviceRef device) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    return (__bridge_retained void*)queue;
}

usize zml_metal_device_name(MTLDeviceRef device, char* buf, usize buf_len) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    NSString* name = dev.name;
    const char* utf8 = [name UTF8String];
    usize len = strlen(utf8);
    if (buf && buf_len > 0) {
        usize copy_len = len < buf_len - 1 ? len : buf_len - 1;
        memcpy(buf, utf8, copy_len);
        buf[copy_len] = '\0';
    }
    return len;
}

// Shader library 

MTLLibraryRef zml_metal_compile_library(
    MTLDeviceRef device, const char* source, usize source_len,
    char* error_buf, usize error_buf_len
) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    NSString* src = [[NSString alloc] initWithBytes:source length:source_len encoding:NSUTF8StringEncoding];
    NSError* error = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;

    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&error];
    if (error && error_buf && error_buf_len > 0) {
        const char* desc = [[error localizedDescription] UTF8String];
        usize len = strlen(desc);
        usize copy_len = len < error_buf_len - 1 ? len : error_buf_len - 1;
        memcpy(error_buf, desc, copy_len);
        error_buf[copy_len] = '\0';
    }
    if (!lib) return NULL;
    return (__bridge_retained void*)lib;
}

MTLPipelineRef zml_metal_create_pipeline(
    MTLDeviceRef device, MTLLibraryRef library, const char* function_name,
    char* error_buf, usize error_buf_len
) {
    id<MTLDevice>  dev = (__bridge id<MTLDevice>)device;
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library;

    NSString* name = [NSString stringWithUTF8String:function_name];
    id<MTLFunction> func = [lib newFunctionWithName:name];
    if (!func) {
        if (error_buf && error_buf_len > 0)
            snprintf(error_buf, error_buf_len, "Function '%s' not found in library", function_name);
        return NULL;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:func error:&error];
    if (error && error_buf && error_buf_len > 0) {
        const char* desc = [[error localizedDescription] UTF8String];
        usize len = strlen(desc);
        usize copy_len = len < error_buf_len - 1 ? len : error_buf_len - 1;
        memcpy(error_buf, desc, copy_len);
        error_buf[copy_len] = '\0';
    }
    if (!pso) return NULL;
    return (__bridge_retained void*)pso;
}

u32 zml_metal_pipeline_max_threads(MTLPipelineRef pipeline) {
    id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline;
    return (u32)pso.maxTotalThreadsPerThreadgroup;
}

void zml_metal_release_library(MTLLibraryRef library) {
    if (library) { (void)(__bridge_transfer id<MTLLibrary>)library; }
}
void zml_metal_release_pipeline(MTLPipelineRef pipeline) {
    if (pipeline) { (void)(__bridge_transfer id<MTLComputePipelineState>)pipeline; }
}

//  Buffers 

MTLBufferRef zml_metal_create_buffer(MTLDeviceRef device, usize size) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    id<MTLBuffer> buf = [dev newBufferWithLength:size options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buf;
}

MTLBufferRef zml_metal_create_buffer_with_data(MTLDeviceRef device, const void* data, usize size) {
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device;
    id<MTLBuffer> buf = [dev newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    return (__bridge_retained void*)buf;
}

void* zml_metal_buffer_contents(MTLBufferRef buffer) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    return [buf contents];
}

usize zml_metal_buffer_length(MTLBufferRef buffer) {
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
    return (usize)buf.length;
}

void zml_metal_release_buffer(MTLBufferRef buffer) {
    if (buffer) { (void)(__bridge_transfer id<MTLBuffer>)buffer; }
}

//  Batched compute encoding 

MTLCommandBufferRef zml_metal_begin_command_buffer(MTLCommandQueueRef queue) {
    id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    if (!cmd) return NULL;
    return (__bridge_retained void*)cmd;
}

MTLComputeEncoderRef zml_metal_begin_compute_encoder(MTLCommandBufferRef cmd_buf) {
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_buf;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) return NULL;
    return (__bridge_retained void*)enc;
}

void zml_metal_encode_dispatch(
    MTLComputeEncoderRef encoder,
    MTLPipelineRef       pipeline,
    MTLBufferRef*        buffers,
    u32             buffer_count,
    const void*          params,
    usize               params_size,
    u32 grid_x, u32 grid_y, u32 grid_z,
    u32 tg_x,   u32 tg_y,   u32 tg_z
) {
    id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)encoder;
    id<MTLComputePipelineState>  pso = (__bridge id<MTLComputePipelineState>)pipeline;

    [enc setComputePipelineState:pso];
    for (u32 i = 0; i < buffer_count; i++) {
        [enc setBuffer:(__bridge id<MTLBuffer>)buffers[i] offset:0 atIndex:i];
    }
    if (params && params_size > 0) {
        [enc setBytes:params length:params_size atIndex:buffer_count];
    }
    [enc dispatchThreads:MTLSizeMake(grid_x, grid_y, grid_z)
       threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, tg_z)];
}

void zml_metal_end_compute_encoder(MTLComputeEncoderRef encoder) {
    id<MTLComputeCommandEncoder> enc = (__bridge_transfer id<MTLComputeCommandEncoder>)encoder;
    [enc endEncoding];
}

int zml_metal_commit_and_wait(MTLCommandBufferRef cmd_buf) {
    id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)cmd_buf;
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.error == nil;
}

void zml_metal_commit_async(MTLCommandBufferRef cmd_buf) {
    // Retains ownership — caller must later call wait_command_buffer to release.
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)cmd_buf;
    [cmd commit];
}

int zml_metal_wait_command_buffer(MTLCommandBufferRef cmd_buf) {
    id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)cmd_buf;
    [cmd waitUntilCompleted];
    return cmd.error == nil;
}

//  Legacy single-shot dispatch 

int zml_metal_dispatch_sync(
    MTLPipelineRef pipeline, MTLCommandQueueRef queue,
    MTLBufferRef* buffers, u32 buffer_count,
    const void* params, usize params_size,
    u32 grid_x, u32 grid_y, u32 grid_z,
    u32 tg_x, u32 tg_y, u32 tg_z
) {
    MTLCommandBufferRef cmd = zml_metal_begin_command_buffer(queue);
    if (!cmd) return 0;
    MTLComputeEncoderRef enc = zml_metal_begin_compute_encoder(cmd);
    if (!enc) return 0;
    zml_metal_encode_dispatch(enc, pipeline, buffers, buffer_count, params, params_size,
                              grid_x, grid_y, grid_z, tg_x, tg_y, tg_z);
    zml_metal_end_compute_encoder(enc);
    return zml_metal_commit_and_wait(cmd);
}

//  Cleanup 

void zml_metal_release_device(MTLDeviceRef device) {
    if (device) { (void)(__bridge_transfer id<MTLDevice>)device; }
}

void zml_metal_release_queue(MTLCommandQueueRef queue) {
    if (queue) { (void)(__bridge_transfer id<MTLCommandQueue>)queue; }
}
