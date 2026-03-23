const std = @import("std");

const c = @cImport({
    @cInclude("bridge.h");
});

//  GPU buffer handle 

pub const GpuBuffer = struct {
    ref: c.MTLBufferRef,

    pub fn contents(self: GpuBuffer, comptime T: type) [*]T {
        return @ptrCast(@alignCast(c.zml_metal_buffer_contents(self.ref)));
    }

    pub fn asSlice(self: GpuBuffer, comptime T: type) []T {
        const len = c.zml_metal_buffer_length(self.ref) / @sizeOf(T);
        return self.contents(T)[0..len];
    }
};

pub const Pipeline = struct {
    ref: c.MTLPipelineRef,

    pub fn maxThreadsPerThreadgroup(self: Pipeline) u32 {
        return c.zml_metal_pipeline_max_threads(self.ref);
    }
};

//  Cached buffer entry 

const CachedBuffer = struct {
    gpu_buf: GpuBuffer,
    byte_len: usize,
};

//  Metal compute engine 

pub const MetalEngine = struct {
    device: c.MTLDeviceRef,
    queue: c.MTLCommandQueueRef,
    library: c.MTLLibraryRef,

    pipelines: std.StringHashMap(Pipeline),

    /// Persistent buffer cache: keyed by CPU pointer address.
    buffer_cache: std.AutoHashMap(usize, CachedBuffer),

    /// Active command buffer recording state.
    active_cmd_buf: ?c.MTLCommandBufferRef,
    active_encoder: ?c.MTLComputeEncoderRef,

    /// Pending async command buffer (needs wait before CPU reads GPU results).
    pending_cmd_buf: ?c.MTLCommandBufferRef,

    alloc: std.mem.Allocator,

    const Self = @This();

    pub fn init(alloc: std.mem.Allocator) !Self {
        const device = c.zml_metal_create_device();
        if (device == null) return error.MetalUnavailable;
        errdefer c.zml_metal_release_device(device);

        const queue = c.zml_metal_create_command_queue(device);
        if (queue == null) return error.MetalQueueFailed;
        errdefer c.zml_metal_release_queue(queue);

        var err_buf: [1024]u8 = undefined;
        const library = c.zml_metal_compile_library(
            device, shader_source.ptr, shader_source.len, &err_buf, err_buf.len,
        );
        if (library == null) {
            std.log.err("Metal shader compile error: {s}", .{std.mem.sliceTo(&err_buf, 0)});
            return error.ShaderCompileFailed;
        }
        errdefer c.zml_metal_release_library(library);

        return .{
            .device = device,
            .queue = queue,
            .library = library,
            .pipelines = std.StringHashMap(Pipeline).init(alloc),
            .buffer_cache = std.AutoHashMap(usize, CachedBuffer).init(alloc),
            .active_cmd_buf = null,
            .active_encoder = null,
            .pending_cmd_buf = null,
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Self) void {
        self.waitIfPending();

        // Release cached buffers
        var cache_it = self.buffer_cache.valueIterator();
        while (cache_it.next()) |entry| c.zml_metal_release_buffer(entry.gpu_buf.ref);
        self.buffer_cache.deinit();

        var it = self.pipelines.valueIterator();
        while (it.next()) |pipe| c.zml_metal_release_pipeline(pipe.ref);
        self.pipelines.deinit();

        c.zml_metal_release_library(self.library);
        c.zml_metal_release_queue(self.queue);
        c.zml_metal_release_device(self.device);
    }

    pub fn deviceName(self: *const Self) [128]u8 {
        var buf: [128]u8 = [_]u8{0} ** 128;
        _ = c.zml_metal_device_name(self.device, &buf, buf.len);
        return buf;
    }

    //  Buffer cache 

    /// Get or create a GPU buffer mirroring a CPU slice.
    /// Cached by pointer address — stable pointers (weights, biases, grads) hit cache.
    /// On cache hit, syncs CPU data → GPU shared memory (cheap memcpy, no alloc).
    pub fn getOrUpload(self: *Self, data: []const f32) GpuBuffer {
        const key = @intFromPtr(data.ptr);
        const byte_len = data.len * @sizeOf(f32);

        if (self.buffer_cache.get(key)) |entry| {
            if (entry.byte_len == byte_len) {
                // Cache hit — sync CPU → GPU shared memory
                const dst = entry.gpu_buf.asSlice(f32);
                @memcpy(dst[0..data.len], data);
                return entry.gpu_buf;
            }
            // Size mismatch — evict
            c.zml_metal_release_buffer(entry.gpu_buf.ref);
            _ = self.buffer_cache.remove(key);
        }

        // Cache miss — create new buffer
        const ref = c.zml_metal_create_buffer_with_data(self.device, @ptrCast(data.ptr), byte_len);
        const gpu_buf = GpuBuffer{ .ref = ref.? };
        self.buffer_cache.put(key, .{ .gpu_buf = gpu_buf, .byte_len = byte_len }) catch {};
        return gpu_buf;
    }

    /// Get cached buffer for a mutable slice (upload current contents).
    pub fn getOrUploadMut(self: *Self, data: []f32) GpuBuffer {
        return self.getOrUpload(@as([]const f32, data));
    }

    /// Download GPU buffer back to a mutable CPU slice (after GPU writes).
    pub fn downloadTo(_: *Self, buf: GpuBuffer, dst: []f32) void {
        const src = buf.asSlice(f32);
        @memcpy(dst, src[0..dst.len]);
    }

    /// Download with accumulate (+=).
    pub fn downloadAccumTo(_: *Self, buf: GpuBuffer, dst: []f32) void {
        const src = buf.asSlice(f32);
        for (dst, src[0..dst.len]) |*d, s| d.* += s;
    }

    //  Raw buffer alloc (not cached, for transient intermediates) 

    pub fn createBuffer(self: *Self, comptime T: type, count: usize) !GpuBuffer {
        const ref = c.zml_metal_create_buffer(self.device, count * @sizeOf(T));
        if (ref == null) return error.MetalBufferAllocFailed;
        return .{ .ref = ref };
    }

    pub fn createBufferFromSlice(self: *Self, comptime T: type, data: []const T) !GpuBuffer {
        const ref = c.zml_metal_create_buffer_with_data(self.device, @ptrCast(data.ptr), data.len * @sizeOf(T));
        if (ref == null) return error.MetalBufferAllocFailed;
        return .{ .ref = ref };
    }

    pub fn releaseBuffer(_: *Self, buf: GpuBuffer) void {
        c.zml_metal_release_buffer(buf.ref);
    }

    //  Pipeline cache 

    pub fn getPipeline(self: *Self, function_name: [:0]const u8) !Pipeline {
        if (self.pipelines.get(function_name)) |cached| return cached;

        var err_buf: [1024]u8 = undefined;
        const ref = c.zml_metal_create_pipeline(
            self.device, self.library, function_name.ptr, &err_buf, err_buf.len,
        );
        if (ref == null) {
            std.log.err("Metal pipeline error for '{s}': {s}", .{
                function_name, std.mem.sliceTo(&err_buf, 0),
            });
            return error.PipelineCreateFailed;
        }
        const pipe = Pipeline{ .ref = ref };
        try self.pipelines.put(function_name, pipe);
        return pipe;
    }

    //  Command buffer recording 

    /// Start recording a batch of kernel dispatches.
    pub fn beginRecording(self: *Self) void {
        std.debug.assert(self.active_cmd_buf == null);
        self.active_cmd_buf = c.zml_metal_begin_command_buffer(self.queue);
        self.active_encoder = c.zml_metal_begin_compute_encoder(self.active_cmd_buf.?);
    }

    /// Encode a kernel dispatch into the active recording.
    pub fn encode(
        self: *Self,
        pipeline: Pipeline,
        buffers: []const GpuBuffer,
        params: ?*const anyopaque,
        params_size: usize,
        grid: Grid,
        tg: ?Grid,
    ) void {
        var buf_refs: [16]c.MTLBufferRef = undefined;
        for (buffers, 0..) |b, i| buf_refs[i] = b.ref;

        const threadgroup = tg orelse blk: {
            const max = pipeline.maxThreadsPerThreadgroup();
            const total = grid.x * grid.y;
            const threads = if (total < max) total else max;
            break :blk Grid{ .x = threads, .y = 1, .z = 1 };
        };

        c.zml_metal_encode_dispatch(
            self.active_encoder.?,
            pipeline.ref,
            &buf_refs,
            @intCast(buffers.len),
            params,
            params_size,
            grid.x, grid.y, grid.z,
            threadgroup.x, threadgroup.y, threadgroup.z,
        );
    }

    /// Typed params convenience for encode.
    pub fn encodeTyped(
        self: *Self,
        pipeline: Pipeline,
        buffers: []const GpuBuffer,
        params: anytype,
        grid: Grid,
        tg: ?Grid,
    ) void {
        const P = @TypeOf(params);
        self.encode(pipeline, buffers, @ptrCast(&params), @sizeOf(P), grid, tg);
    }

    /// End recording, commit, and wait for GPU to finish.
    pub fn commitAndWait(self: *Self) void {
        c.zml_metal_end_compute_encoder(self.active_encoder.?);
        _ = c.zml_metal_commit_and_wait(self.active_cmd_buf.?);
        self.active_encoder = null;
        self.active_cmd_buf = null;
    }

    /// End recording and commit asynchronously.  Call waitIfPending() before reading results.
    pub fn commitAsync(self: *Self) void {
        c.zml_metal_end_compute_encoder(self.active_encoder.?);
        c.zml_metal_commit_async(self.active_cmd_buf.?);
        self.pending_cmd_buf = self.active_cmd_buf;
        self.active_encoder = null;
        self.active_cmd_buf = null;
    }

    /// Block until any pending async command buffer has finished.
    pub fn waitIfPending(self: *Self) void {
        if (self.pending_cmd_buf) |cmd| {
            _ = c.zml_metal_wait_command_buffer(cmd);
            self.pending_cmd_buf = null;
        }
    }

    //  Legacy single-shot dispatch (used by tests/metal_test.zig) 

    pub const Grid = struct { x: u32, y: u32 = 1, z: u32 = 1 };

    pub fn dispatchSync(
        self: *Self, pipeline: Pipeline, buffers: []const GpuBuffer,
        params: ?*const anyopaque, params_size: usize, grid: Grid, tg: ?Grid,
    ) !void {
        var buf_refs: [16]c.MTLBufferRef = undefined;
        if (buffers.len > 16) return error.TooManyBuffers;
        for (buffers, 0..) |b, i| buf_refs[i] = b.ref;

        const threadgroup = tg orelse blk: {
            const max = pipeline.maxThreadsPerThreadgroup();
            const threads = if (grid.x < max) grid.x else max;
            break :blk Grid{ .x = threads, .y = 1, .z = 1 };
        };

        const ok = c.zml_metal_dispatch_sync(
            pipeline.ref, self.queue, &buf_refs, @intCast(buffers.len),
            params, params_size,
            grid.x, grid.y, grid.z, threadgroup.x, threadgroup.y, threadgroup.z,
        );
        if (ok == 0) return error.MetalDispatchFailed;
    }

    pub fn dispatch(
        self: *Self, pipeline: Pipeline, buffers: []const GpuBuffer,
        params: anytype, grid: Grid, tg: ?Grid,
    ) !void {
        const P = @TypeOf(params);
        try self.dispatchSync(pipeline, buffers, @ptrCast(&params), @sizeOf(P), grid, tg);
    }
};

const shader_source = @embedFile("shaders/kernels.metal");
