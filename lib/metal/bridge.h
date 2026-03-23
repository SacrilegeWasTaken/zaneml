#ifndef ZANEML_METAL_BRIDGE_H
#define ZANEML_METAL_BRIDGE_H

// export types

typedef void* MTLDeviceRef;
typedef void* MTLCommandQueueRef;
typedef void* MTLLibraryRef;
typedef void* MTLPipelineRef;
typedef void* MTLBufferRef;
typedef void* MTLCommandBufferRef;
typedef void* MTLComputeEncoderRef;

typedef unsigned int  u32;
typedef unsigned long usize;

//  Device & queue 

MTLDeviceRef zml_metal_create_device(void);
MTLCommandQueueRef zml_metal_create_command_queue(MTLDeviceRef device);
usize zml_metal_device_name(MTLDeviceRef device, char* buf, usize buf_len);

//  Shader library 

MTLLibraryRef zml_metal_compile_library(
    MTLDeviceRef device,
    const char*  source,
    usize        source_len,
    char*        error_buf,
    usize        error_buf_len
);

MTLPipelineRef zml_metal_create_pipeline(
    MTLDeviceRef   device,
    MTLLibraryRef  library,
    const char*    function_name,
    char*          error_buf,
    usize          error_buf_len
);

u32  zml_metal_pipeline_max_threads(MTLPipelineRef pipeline);
void zml_metal_release_library(MTLLibraryRef library);
void zml_metal_release_pipeline(MTLPipelineRef pipeline);

//  Buffers 

MTLBufferRef zml_metal_create_buffer(MTLDeviceRef device, usize size);
MTLBufferRef zml_metal_create_buffer_with_data(MTLDeviceRef device, const void* data, usize size);
void* zml_metal_buffer_contents(MTLBufferRef buffer);
usize zml_metal_buffer_length(MTLBufferRef buffer);
void  zml_metal_release_buffer(MTLBufferRef buffer);

//  Batched compute encoding 

MTLCommandBufferRef  zml_metal_begin_command_buffer(MTLCommandQueueRef queue);
MTLComputeEncoderRef zml_metal_begin_compute_encoder(MTLCommandBufferRef cmd_buf);

void zml_metal_encode_dispatch(
    MTLComputeEncoderRef encoder,
    MTLPipelineRef       pipeline,
    MTLBufferRef*        buffers,
    u32                  buffer_count,
    const void*          params,
    usize                params_size,
    u32 grid_x, u32 grid_y, u32 grid_z,
    u32 tg_x,   u32 tg_y,   u32 tg_z
);

// Like zml_metal_encode_dispatch but also sets threadgroup memory (index 0) of tg_mem_bytes bytes.
void zml_metal_encode_dispatch_tgmem(
    MTLComputeEncoderRef encoder,
    MTLPipelineRef       pipeline,
    MTLBufferRef*        buffers,
    u32                  buffer_count,
    const void*          params,
    usize                params_size,
    u32 grid_x, u32 grid_y, u32 grid_z,
    u32 tg_x,   u32 tg_y,   u32 tg_z,
    usize                tg_mem_bytes
);

void zml_metal_end_compute_encoder(MTLComputeEncoderRef encoder);

int  zml_metal_commit_and_wait(MTLCommandBufferRef cmd_buf);
void zml_metal_commit_async(MTLCommandBufferRef cmd_buf);
int  zml_metal_wait_command_buffer(MTLCommandBufferRef cmd_buf);

//  Legacy single-shot dispatch 

int zml_metal_dispatch_sync(
    MTLPipelineRef      pipeline,
    MTLCommandQueueRef  queue,
    MTLBufferRef*       buffers,
    u32                 buffer_count,
    const void*         params,
    usize               params_size,
    u32 grid_x, u32 grid_y, u32 grid_z,
    u32 tg_x,   u32 tg_y,   u32 tg_z
);

//  Cleanup 

void zml_metal_release_device(MTLDeviceRef device);
void zml_metal_release_queue(MTLCommandQueueRef queue);

#endif // ZANEML_METAL_BRIDGE_H
