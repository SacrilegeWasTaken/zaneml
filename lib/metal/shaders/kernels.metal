#include <metal_stdlib>
using namespace metal;

//Params structs (must match Zig-side layout) 

struct MatmulParams {
    uint M;  // rows of A / rows of C
    uint K;  // cols of A / rows of B
    uint N;  // cols of B / cols of C
};

struct ElementwiseParams {
    uint length;
};

struct SGDParams {
    uint   length;
    float  lr;
};

struct AdamParams {
    uint   length;
    float  lr;
    float  beta1;
    float  beta2;
    float  eps;
    float  weight_decay;  // 0 for plain Adam
    uint   t;             // timestep (1-based)
};

// MATMUL: C[M×N] = A[M×K] × B[K×N]

// Naive per-element kernel — each thread computes one C[row, col].
// Good enough for small matrices; will upgrade to tiled later.
kernel void matmul_f32(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device       float* C      [[buffer(2)]],
    constant MatmulParams& p   [[buffer(3)]],
    uint2 gid                  [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= p.M || col >= p.N) return;

    float sum = 0.0f;
    for (uint k = 0; k < p.K; k++) {
        sum += A[row * p.K + k] * B[k * p.N + col];
    }
    C[row * p.N + col] = sum;
}

// Backward: dA[M×K] = dC[M×N] × B^T[N×K]
kernel void matmul_backward_a(
    device const float* grad_c  [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device       float* grad_a  [[buffer(2)]],
    constant MatmulParams& p    [[buffer(3)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint row = gid.y;  // M
    uint col = gid.x;  // K
    if (row >= p.M || col >= p.K) return;

    float sum = 0.0f;
    for (uint n = 0; n < p.N; n++) {
        sum += grad_c[row * p.N + n] * B[col * p.N + n];  // B^T[k,n] = B[n,k]... wait
        // B is K×N, B^T is N×K.  B^T[n, col] = B[col * N + n]?  No.
        // B[k, n] = B[k * N + n].  We need B^T[n, k] = B[k * N + n].
        // dA[row, col] = sum_n dC[row, n] * B^T[n, col] = sum_n dC[row, n] * B[col, n]
        // B[col, n] = B[col * N + n]  ✓
    }
    grad_a[row * p.K + col] += sum;
}

// Backward: dB[K×N] = A^T[K×M] × dC[M×N]
kernel void matmul_backward_b(
    device const float* A       [[buffer(0)]],
    device const float* grad_c  [[buffer(1)]],
    device       float* grad_b  [[buffer(2)]],
    constant MatmulParams& p    [[buffer(3)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint row = gid.y;  // K
    uint col = gid.x;  // N
    if (row >= p.K || col >= p.N) return;

    float sum = 0.0f;
    for (uint m = 0; m < p.M; m++) {
        // A^T[row, m] = A[m, row] = A[m * K + row]
        sum += A[m * p.K + row] * grad_c[m * p.N + col];
    }
    grad_b[row * p.N + col] += sum;
}

// ─── TILED MATMUL (16×16 tiles with threadgroup memory) 
// ~16x fewer global memory reads than naive kernels via data reuse in shared memory.

constant uint TILE = 16;

// C[M×N] = A[M×K] × B[K×N]  (overwrite)
kernel void matmul_tiled_f32(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device       float* C      [[buffer(2)]],
    constant MatmulParams& p   [[buffer(3)]],
    uint2 gid                  [[thread_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    threadgroup float* sh      [[threadgroup(0)]]
) {
    threadgroup float* tA = sh;
    threadgroup float* tB = sh + TILE * TILE;
    uint row = tgid.y * TILE + lid.y;
    uint col = tgid.x * TILE + lid.x;
    float sum = 0.0f;
    uint num_tiles = (p.K + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE + lid.x;
        uint b_row = t * TILE + lid.y;
        tA[lid.y * TILE + lid.x] = (row < p.M && a_col < p.K) ? A[row * p.K + a_col] : 0.0f;
        tB[lid.y * TILE + lid.x] = (b_row < p.K && col < p.N) ? B[b_row * p.N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++)
            sum += tA[lid.y * TILE + k] * tB[k * TILE + lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < p.M && col < p.N)
        C[row * p.N + col] = sum;
}

// C[M×N] = A[M×K] × B^T[N×K]  (B stored transposed as [N×K])  (overwrite)
kernel void matmul_bT_tiled_f32(
    device const float* A      [[buffer(0)]],
    device const float* BT     [[buffer(1)]],
    device       float* C      [[buffer(2)]],
    constant MatmulParams& p   [[buffer(3)]],
    uint2 gid                  [[thread_position_in_grid]],
    uint2 lid                  [[thread_position_in_threadgroup]],
    uint2 tgid                 [[threadgroup_position_in_grid]],
    threadgroup float* sh      [[threadgroup(0)]]
) {
    threadgroup float* tA = sh;
    threadgroup float* tB = sh + TILE * TILE;
    uint row = tgid.y * TILE + lid.y;
    uint col = tgid.x * TILE + lid.x;
    float sum = 0.0f;
    uint num_tiles = (p.K + TILE - 1) / TILE;
    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE + lid.x;
        uint b_k   = t * TILE + lid.y;  // k index into BT
        tA[lid.y * TILE + lid.x] = (row < p.M && a_col < p.K) ? A[row * p.K + a_col] : 0.0f;
        // BT is [N×K], logical B[k, col] = BT[col * K + k]
        tB[lid.y * TILE + lid.x] = (b_k < p.K && col < p.N) ? BT[col * p.K + b_k] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++)
            sum += tA[lid.y * TILE + k] * tB[k * TILE + lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < p.M && col < p.N)
        C[row * p.N + col] = sum;
}

// dA[M×K] += dC[M×N] × B^T  (accumulate)
// dA[row,col] = sum_n dC[row,n] * B[col,n]  (B is K×N, accessing row col of B)
kernel void matmul_backward_a_tiled(
    device const float* grad_c  [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device       float* grad_a  [[buffer(2)]],
    constant MatmulParams& p    [[buffer(3)]],
    uint2 gid                   [[thread_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    threadgroup float* sh       [[threadgroup(0)]]
) {
    // Treating this as C[M×K] = dC[M×N] × B_logical^T[N×K]
    // where B_logical^T[n, k] = B[k, n] = B[k * N + n]
    // So: A_mat = grad_c[M×N], B_mat stored as B[K×N] but we need B^T[N×K]
    threadgroup float* tA = sh;
    threadgroup float* tB = sh + TILE * TILE;
    uint row = tgid.y * TILE + lid.y;  // M dim
    uint col = tgid.x * TILE + lid.x;  // K dim
    float sum = 0.0f;
    uint num_tiles = (p.N + TILE - 1) / TILE;  // reduce over N
    for (uint t = 0; t < num_tiles; t++) {
        uint a_col = t * TILE + lid.x;  // N index
        uint b_row = t * TILE + lid.y;  // N index
        tA[lid.y * TILE + lid.x] = (row < p.M && a_col < p.N) ? grad_c[row * p.N + a_col] : 0.0f;
        // B^T[n, k] = B[k * N + n], k=col, n=b_row
        tB[lid.y * TILE + lid.x] = (b_row < p.N && col < p.K) ? B[col * p.N + b_row] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++)
            sum += tA[lid.y * TILE + k] * tB[k * TILE + lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < p.M && col < p.K)
        grad_a[row * p.K + col] += sum;
}

// dB[K×N] += A^T[K×M] × dC[M×N]  (accumulate)
kernel void matmul_backward_b_tiled(
    device const float* A       [[buffer(0)]],
    device const float* grad_c  [[buffer(1)]],
    device       float* grad_b  [[buffer(2)]],
    constant MatmulParams& p    [[buffer(3)]],
    uint2 gid                   [[thread_position_in_grid]],
    uint2 lid                   [[thread_position_in_threadgroup]],
    uint2 tgid                  [[threadgroup_position_in_grid]],
    threadgroup float* sh       [[threadgroup(0)]]
) {
    // Output: dB[row=k, col=n], reduce over M
    // A^T[k, m] = A[m * K + k]
    threadgroup float* tA = sh;
    threadgroup float* tB = sh + TILE * TILE;
    uint row = tgid.y * TILE + lid.y;  // K dim
    uint col = tgid.x * TILE + lid.x;  // N dim
    float sum = 0.0f;
    uint num_tiles = (p.M + TILE - 1) / TILE;  // reduce over M
    for (uint t = 0; t < num_tiles; t++) {
        uint m_a = t * TILE + lid.x;  // M index for A^T column
        uint m_b = t * TILE + lid.y;  // M index for dC row
        // A^T[row, m_a] = A[m_a * K + row]
        tA[lid.y * TILE + lid.x] = (m_a < p.M && row < p.K) ? A[m_a * p.K + row] : 0.0f;
        tB[lid.y * TILE + lid.x] = (m_b < p.M && col < p.N) ? grad_c[m_b * p.N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; k++)
            sum += tA[lid.y * TILE + k] * tB[k * TILE + lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < p.K && col < p.N)
        grad_b[row * p.N + col] += sum;
}

// ─── UTILITY KERNELS 

// Element-wise sum of 3 vectors: out[i] = a[i] + b[i] + c[i]
kernel void add3_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device const float* c       [[buffer(2)]],
    device       float* out     [[buffer(3)]],
    constant ElementwiseParams& p [[buffer(4)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    out[gid] = a[gid] + b[gid] + c[gid];
}

// Fused activation backward: out[i] = grad_out[i] * act'(pre_act[i])
struct ActBwdParams { uint length; uint act_type; };
// act_type: 0=linear, 1=relu, 2=sigmoid, 3=tanh, 4=silu, 5=gelu

kernel void fused_act_backward(
    device const float* pre_act  [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device       float* delta    [[buffer(2)]],
    constant ActBwdParams& p     [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    float go = grad_out[gid];
    float x  = pre_act[gid];
    float d;
    switch (p.act_type) {
        case 0: d = go; break;  // linear
        case 1: d = (x > 0.0f) ? go : 0.0f; break;  // relu
        case 2: { float s = 1.0f / (1.0f + exp(-x)); d = go * s * (1.0f - s); break; }  // sigmoid
        case 3: { float t = tanh(x); d = go * (1.0f - t * t); break; }  // tanh
        case 4: { float s = 1.0f / (1.0f + exp(-x)); d = go * s * (1.0f + x * (1.0f - s)); break; }  // silu
        case 5: {  // gelu
            float cc = 0.7978845608f;
            float x3 = x * x * x;
            float inner = cc * (x + 0.044715f * x3);
            float th = tanh(inner);
            float dtanh = 1.0f - th * th;
            float dinner = cc * (1.0f + 3.0f * 0.044715f * x * x);
            d = go * (0.5f * (1.0f + th) + 0.5f * x * dtanh * dinner);
            break;
        }
        default: d = go; break;
    }
    delta[gid] = d;
}

// Reduce bias gradient: grad_b[i] += sum over seq of delta[t * n_out + i]
// Grid: (n_out), each thread sums over seq positions for one neuron
struct ReduceBiasParams { uint seq; uint n_out; };

kernel void reduce_bias_grad(
    device const float* delta    [[buffer(0)]],  // [seq × n_out]
    device       float* grad_b   [[buffer(1)]],  // [n_out]  (accumulated)
    constant ReduceBiasParams& p [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= p.n_out) return;
    float sum = 0.0f;
    for (uint t = 0; t < p.seq; t++)
        sum += delta[t * p.n_out + gid];
    grad_b[gid] += sum;
}

// ELEMENTWISE OPERATIONS

kernel void relu_forward(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    output[gid] = max(input[gid], 0.0f);
}

kernel void relu_backward(
    device const float* input    [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device       float* grad_in  [[buffer(2)]],
    constant ElementwiseParams& p [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    grad_in[gid] += (input[gid] > 0.0f) ? grad_out[gid] : 0.0f;
}

kernel void sigmoid_forward(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    output[gid] = 1.0f / (1.0f + exp(-input[gid]));
}

kernel void sigmoid_backward(
    device const float* output   [[buffer(0)]],  // sigmoid output (not input)
    device const float* grad_out [[buffer(1)]],
    device       float* grad_in  [[buffer(2)]],
    constant ElementwiseParams& p [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    float s = output[gid];
    grad_in[gid] += grad_out[gid] * s * (1.0f - s);
}

kernel void tanh_forward(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    output[gid] = tanh(input[gid]);
}

kernel void tanh_backward(
    device const float* output   [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device       float* grad_in  [[buffer(2)]],
    constant ElementwiseParams& p [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    float t = output[gid];
    grad_in[gid] += grad_out[gid] * (1.0f - t * t);
}

kernel void silu_forward(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    float x = input[gid];
    float s = 1.0f / (1.0f + exp(-x));
    output[gid] = x * s;
}

kernel void silu_backward(
    device const float* input    [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device       float* grad_in  [[buffer(2)]],
    constant ElementwiseParams& p [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    float x = input[gid];
    float s = 1.0f / (1.0f + exp(-x));
    // d/dx [x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //                        = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    grad_in[gid] += grad_out[gid] * s * (1.0f + x * (1.0f - s));
}

kernel void gelu_forward(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant ElementwiseParams& p [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    float x = input[gid];
    // tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    float c = 0.7978845608f;  // sqrt(2/π)
    float inner = c * (x + 0.044715f * x * x * x);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}

kernel void gelu_backward(
    device const float* input    [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device       float* grad_in  [[buffer(2)]],
    constant ElementwiseParams& p [[buffer(3)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    float x = input[gid];
    float c = 0.7978845608f;
    float x3 = x * x * x;
    float inner = c * (x + 0.044715f * x3);
    float th = tanh(inner);
    float dtanh = 1.0f - th * th;
    float dinner = c * (1.0f + 3.0f * 0.044715f * x * x);
    grad_in[gid] += grad_out[gid] * (0.5f * (1.0f + th) + 0.5f * x * dtanh * dinner);
}

// BATCH MATMUL: C[M×N] = A[M×K] × B^T[N×K]
// B is stored transposed as [N×K] (e.g. weight matrix W[n_out×n_in]).
// Used for FFN batch-forward: input_seq[seq×n_in] × W^T → out[seq×n_out].
kernel void matmul_bT_f32(
    device const float* A       [[buffer(0)]],  // [M × K] row-major
    device const float* BT      [[buffer(1)]],  // [N × K] row-major (B stored as B^T)
    device       float* C       [[buffer(2)]],  // [M × N] row-major
    constant MatmulParams& p    [[buffer(3)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint row = gid.y;  // 0 .. M-1
    uint col = gid.x;  // 0 .. N-1
    if (row >= p.M || col >= p.N) return;
    float sum = 0.0f;
    for (uint k = 0; k < p.K; k++) {
        sum += A[row * p.K + k] * BT[col * p.K + k];
    }
    C[row * p.N + col] = sum;
}

// BATCH BIAS ADD: out[t, i] = x[t, i] + bias[i]
// 2-D dispatch: gid.y = sequence position, gid.x = neuron index.
struct AddBiasBatchParams {
    uint seq;
    uint n_out;
};

kernel void add_bias_batch_f32(
    device const float* x       [[buffer(0)]],  // [seq × n_out]
    device const float* bias    [[buffer(1)]],  // [n_out]
    device       float* out     [[buffer(2)]],  // [seq × n_out]
    constant AddBiasBatchParams& p [[buffer(3)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint t = gid.y;
    uint i = gid.x;
    if (t >= p.seq || i >= p.n_out) return;
    uint idx = t * p.n_out + i;
    out[idx] = x[idx] + bias[i];
}

// vector add: out = a + b
kernel void add_f32(
    device const float* a       [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    device       float* out     [[buffer(2)]],
    constant ElementwiseParams& p [[buffer(3)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    out[gid] = a[gid] + b[gid];
}

// vector scale: out = a * scalar
struct ScaleParams {
    uint  length;
    float scalar;
};

kernel void scale_f32(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant ScaleParams& p     [[buffer(2)]],
    uint gid                    [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    output[gid] = input[gid] * p.scalar;
}

// OPTIMIZERS

kernel void sgd_update(
    device float* params         [[buffer(0)]],
    device const float* grads    [[buffer(1)]],
    constant SGDParams& p        [[buffer(2)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    params[gid] -= p.lr * grads[gid];
}

kernel void adam_update(
    device float* params         [[buffer(0)]],
    device const float* grads    [[buffer(1)]],
    device float* m              [[buffer(2)]],
    device float* v              [[buffer(3)]],
    constant AdamParams& p       [[buffer(4)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;

    float g = grads[gid];

    // AdamW: decouple weight decay (applied before moment update)
    if (p.weight_decay > 0.0f) {
        params[gid] -= p.lr * p.weight_decay * params[gid];
    }

    // Moment updates
    m[gid] = p.beta1 * m[gid] + (1.0f - p.beta1) * g;
    v[gid] = p.beta2 * v[gid] + (1.0f - p.beta2) * g * g;

    // Bias correction
    float m_hat = m[gid] / (1.0f - pow(p.beta1, (float)p.t));
    float v_hat = v[gid] / (1.0f - pow(p.beta2, (float)p.t));

    params[gid] -= p.lr * m_hat / (sqrt(v_hat) + p.eps);
}

// LOSS

// MSE: per-element (output - target)² and gradient 2*(output - target) / n
struct LossParams {
    uint  length;
    float inv_n;    // 1.0 / n  for averaging
};

kernel void mse_grad(
    device const float* output   [[buffer(0)]],
    device const float* target   [[buffer(1)]],
    device       float* grad     [[buffer(2)]],
    device       float* loss_buf [[buffer(3)]],  // per-element loss for reduction
    constant LossParams& p       [[buffer(4)]],
    uint gid                     [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    float diff = output[gid] - target[gid];
    grad[gid] = 2.0f * diff * p.inv_n;
    loss_buf[gid] = diff * diff * p.inv_n;
}

// Simple parallel reduction: sum an array.  Call with grid = length/2 (round up).
// Iteratively halves in-place.  Caller must loop with shrinking grid.
kernel void reduce_sum(
    device float* data           [[buffer(0)]],
    constant uint& length        [[buffer(1)]],
    uint gid                     [[thread_position_in_grid]]
) {
    uint stride = length / 2;
    if (gid < stride) {
        data[gid] += data[gid + stride];
    }
    // Handle odd length: thread 0 picks up the last element
    if (gid == 0 && (length & 1) != 0) {
        data[0] += data[length - 1];
    }
}

// ─── LAYER NORM (batch, power-of-2 d_model) 
// forward: one threadgroup per token position, d threads per group (d = d_model)
// Single params struct at buffer(6): { uint d; float eps; }

struct LayerNormFwdParams { uint d; float eps; };
struct NormBwdParams      { uint d; };
struct RMSNormFwdParams   { uint d; float eps; };

kernel void layernorm_fwd_seq(
    device const float* x        [[buffer(0)]],   // [seq × d]
    device const float* gamma    [[buffer(1)]],   // [d]
    device const float* beta     [[buffer(2)]],   // [d]
    device       float* out      [[buffer(3)]],   // [seq × d]
    device       float* x_norm   [[buffer(4)]],   // [seq × d]
    device       float* rstd_buf [[buffer(5)]],   // [seq]
    constant LayerNormFwdParams& p [[buffer(6)]],
    uint2 tgpos [[threadgroup_position_in_grid]],  // tgpos.y = position
    uint  lid   [[thread_index_in_threadgroup]],
    threadgroup float* sh        [[threadgroup(0)]]
) {
    uint pos  = tgpos.y;
    uint d    = p.d;
    uint base = pos * d;

    float xi = (lid < d) ? x[base + lid] : 0.0f;
    sh[lid] = xi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // parallel sum → mean
    for (uint s = d >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sh[0] / float(d);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // variance
    float diff = xi - mean;
    sh[lid] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = d >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rs = rsqrt(sh[0] / float(d) + p.eps);

    if (lid < d) {
        float xn    = diff * rs;
        x_norm[base + lid] = xn;
        out[base + lid]    = gamma[lid] * xn + beta[lid];
    }
    if (lid == 0) rstd_buf[pos] = rs;
}

// backward: accumulates grad_gamma, grad_beta; writes grad_in
// 7 data buffers (0-6), params struct { uint d } at buffer(7)
kernel void layernorm_bwd_seq(
    device const float* grad_out  [[buffer(0)]],  // [seq × d]
    device const float* x_norm    [[buffer(1)]],  // [seq × d]
    device const float* gamma     [[buffer(2)]],  // [d]
    device const float* rstd_buf  [[buffer(3)]],  // [seq]
    device atomic_float* grad_gamma [[buffer(4)]],// [d]  (atomic accumulate)
    device atomic_float* grad_beta  [[buffer(5)]],// [d]  (atomic accumulate)
    device       float* grad_in   [[buffer(6)]],  // [seq × d]
    constant NormBwdParams& p     [[buffer(7)]],
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    threadgroup float* sh         [[threadgroup(0)]]  // [d]
) {
    uint pos  = tgpos.y;
    uint d    = p.d;
    uint base = pos * d;
    float rs  = rstd_buf[pos];

    float go  = (lid < d) ? grad_out[base + lid]   : 0.0f;
    float xn  = (lid < d) ? x_norm  [base + lid]   : 0.0f;
    float g   = (lid < d) ? gamma[lid]              : 0.0f;

    // accumulate grad_gamma and grad_beta (atomic because multiple positions contribute)
    if (lid < d) {
        atomic_fetch_add_explicit(&grad_gamma[lid], go * xn, memory_order_relaxed);
        atomic_fetch_add_explicit(&grad_beta [lid], go,      memory_order_relaxed);
    }

    // sum1 = sum(go * g),  sum2 = sum(go * g * xn)
    sh[lid] = go * g;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = d >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum1 = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sh[lid] = go * g * xn;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = d >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum2 = sh[0];

    if (lid < d) {
        float inv_d = rs / float(d);
        grad_in[base + lid] = inv_d * (float(d) * go * g - sum1 - xn * sum2);
    }
}

// ─── RMS NORM (batch) 
// 5 data buffers (0-4), params struct { uint d; float eps } at buffer(5)
kernel void rmsnorm_fwd_seq(
    device const float* x        [[buffer(0)]],
    device const float* gamma    [[buffer(1)]],
    device       float* out      [[buffer(2)]],
    device       float* x_norm   [[buffer(3)]],
    device       float* rms_buf  [[buffer(4)]],  // stores rms (not rstd!)
    constant RMSNormFwdParams& p [[buffer(5)]],
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    threadgroup float* sh        [[threadgroup(0)]]
) {
    uint pos  = tgpos.y;
    uint d    = p.d;
    uint base = pos * d;

    float xi  = (lid < d) ? x[base + lid] : 0.0f;
    sh[lid]   = xi * xi;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = d >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = sqrt(sh[0] / float(d) + p.eps);

    if (lid < d) {
        float xn         = xi / rms;
        x_norm[base + lid] = xn;
        out   [base + lid] = gamma[lid] * xn;
    }
    if (lid == 0) rms_buf[pos] = rms;
}

// 6 data buffers (0-5), params struct { uint d } at buffer(6)
kernel void rmsnorm_bwd_seq(
    device const float* grad_out  [[buffer(0)]],
    device const float* x_norm    [[buffer(1)]],
    device const float* gamma     [[buffer(2)]],
    device const float* rms_buf   [[buffer(3)]],
    device atomic_float* grad_gamma [[buffer(4)]],
    device       float* grad_in   [[buffer(5)]],
    constant NormBwdParams& p     [[buffer(6)]],
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    threadgroup float* sh         [[threadgroup(0)]]
) {
    uint pos  = tgpos.y;
    uint d    = p.d;
    uint base = pos * d;
    float rms = rms_buf[pos];

    float go  = (lid < d) ? grad_out[base + lid] : 0.0f;
    float xn  = (lid < d) ? x_norm  [base + lid] : 0.0f;
    float g   = (lid < d) ? gamma[lid]            : 0.0f;

    if (lid < d) atomic_fetch_add_explicit(&grad_gamma[lid], go * xn, memory_order_relaxed);

    sh[lid] = go * g * xn;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = d >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = sh[0] / float(d);

    if (lid < d) {
        float inv_rms   = 1.0f / rms;
        grad_in[base + lid] = inv_rms * (g * go - xn * dot);
    }
}

// ─── POSITIONAL EMBEDDING 
kernel void pe_fwd(
    device const float* input  [[buffer(0)]],  // [len]
    device const float* embed  [[buffer(1)]],  // [len]
    device       float* out    [[buffer(2)]],  // [len]
    constant uint& length      [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    out[gid] = input[gid] + embed[gid];
}

// grad_embed += grad_out  (atomic since multiple calls may overlap, but here sequential)
kernel void pe_bwd_embed(
    device const float* grad_out      [[buffer(0)]],  // [len]
    device atomic_float* grad_embed   [[buffer(1)]],  // [len]
    constant uint& length             [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= length) return;
    atomic_fetch_add_explicit(&grad_embed[gid], grad_out[gid], memory_order_relaxed);
}

// ─── ATTENTION 

struct AttnScoreParams {
    uint  seq;
    uint  n_heads;
    uint  d_head;
    float scale;
    uint  causal;
};

// scores[h, i, j] = sum_dk(Q[i, h*d_head+dk] * K[j, h*d_head+dk]) * scale
// (+ causal mask)
// Grid: (seq, seq, n_heads)
kernel void attn_scores_fwd(
    device const float* Q       [[buffer(0)]],  // [seq × (n_heads*d_head)]
    device const float* K       [[buffer(1)]],  // [seq × (n_heads*d_head)]
    device       float* scores  [[buffer(2)]],  // [n_heads × seq × seq]
    constant AttnScoreParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]       // x=j, y=i, z=h
) {
    uint j = gid.x, i = gid.y, h = gid.z;
    if (j >= p.seq || i >= p.seq || h >= p.n_heads) return;

    uint ho  = h * p.d_head;
    uint dm  = p.n_heads * p.d_head;
    float s  = 0.0f;
    for (uint dk = 0; dk < p.d_head; dk++)
        s += Q[i * dm + ho + dk] * K[j * dm + ho + dk];
    s *= p.scale;
    if (p.causal && j > i) s = -1e9f;
    scores[h * p.seq * p.seq + i * p.seq + j] = s;
}

struct SoftmaxParams { uint n_rows; uint row_len; };

// In-place softmax per row.  Threadgroup = (row_len, 1), grid = (row_len, n_rows)
kernel void softmax_rows_inplace(
    device float* x             [[buffer(0)]],
    constant SoftmaxParams& p   [[buffer(1)]],
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    threadgroup float* sh       [[threadgroup(0)]]
) {
    uint row  = tgpos.y;
    uint rlen = p.row_len;
    uint base = row * rlen;

    float xi  = (lid < rlen) ? x[base + lid] : -INFINITY;
    sh[lid]   = xi;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // max reduction
    for (uint s = rlen >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] = max(sh[lid], sh[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mx  = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ex  = (lid < rlen) ? exp(xi - mx) : 0.0f;
    sh[lid]   = ex;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // sum reduction
    for (uint s = rlen >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sm  = sh[0];

    if (lid < rlen) x[base + lid] = ex / sm;
}

struct AttnContextParams { uint seq; uint n_heads; uint d_head; };

// context[i, h*d_head+dk] = sum_j(attn[h,i,j] * V[j, h*d_head+dk])
// Grid: (d_head, seq, n_heads)
kernel void attn_context_fwd(
    device const float* attn    [[buffer(0)]],  // [n_heads × seq × seq]
    device const float* V       [[buffer(1)]],  // [seq × (n_heads*d_head)]
    device       float* ctx     [[buffer(2)]],  // [seq × (n_heads*d_head)]
    constant AttnContextParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]       // x=dk, y=i, z=h
) {
    uint dk = gid.x, i = gid.y, h = gid.z;
    if (dk >= p.d_head || i >= p.seq || h >= p.n_heads) return;

    uint dm  = p.n_heads * p.d_head;
    uint ho  = h * p.d_head;
    float s  = 0.0f;
    for (uint j = 0; j < p.seq; j++)
        s += attn[h * p.seq * p.seq + i * p.seq + j] * V[j * dm + ho + dk];
    ctx[i * dm + ho + dk] = s;
}

// ─── ATTENTION BACKWARD 

// grad_v[j, h*d_head+dk] += sum_i(attn[h,i,j] * grad_ctx[i, h*d_head+dk])
// Grid: (d_head, seq, n_heads)
kernel void attn_grad_v(
    device const float* attn      [[buffer(0)]],
    device const float* grad_ctx  [[buffer(1)]],
    device atomic_float* grad_v   [[buffer(2)]],
    constant AttnContextParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]       // x=dk, y=j, z=h
) {
    uint dk = gid.x, j = gid.y, h = gid.z;
    if (dk >= p.d_head || j >= p.seq || h >= p.n_heads) return;

    uint dm = p.n_heads * p.d_head;
    uint ho = h * p.d_head;
    float s = 0.0f;
    for (uint i = 0; i < p.seq; i++)
        s += attn[h * p.seq * p.seq + i * p.seq + j] * grad_ctx[i * dm + ho + dk];
    atomic_fetch_add_explicit(&grad_v[j * dm + ho + dk], s, memory_order_relaxed);
}

// grad_attn_pre[h,i,j] = sum_dk(grad_ctx[i,h*d_head+dk] * V[j,h*d_head+dk])
// (before softmax backward)
// Grid: (seq, seq, n_heads)
kernel void attn_grad_attn_pre(
    device const float* grad_ctx  [[buffer(0)]],
    device const float* V         [[buffer(1)]],
    device       float* grad_pre  [[buffer(2)]],  // [n_heads × seq × seq]
    constant AttnContextParams& p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]          // x=j, y=i, z=h
) {
    uint j = gid.x, i = gid.y, h = gid.z;
    if (j >= p.seq || i >= p.seq || h >= p.n_heads) return;

    uint dm = p.n_heads * p.d_head;
    uint ho = h * p.d_head;
    float s = 0.0f;
    for (uint dk = 0; dk < p.d_head; dk++)
        s += grad_ctx[i * dm + ho + dk] * V[j * dm + ho + dk];
    grad_pre[h * p.seq * p.seq + i * p.seq + j] = s;
}

// Softmax backward (in-place on grad_pre):
//   grad_pre[i, j] = attn[i,j] * (grad_pre[i,j] - sum_k(grad_pre[i,k]*attn[i,k]))
// Processes all rows for all heads.  Threadgroup=(seq,1), grid=(seq, n_heads*seq)
struct SoftmaxBwdParams { uint n_rows; uint row_len; };

kernel void softmax_bwd_rows(
    device const float* attn      [[buffer(0)]],  // [n_heads × seq × seq] softmax output
    device       float* grad_pre  [[buffer(1)]],  // in-place [n_heads × seq × seq]
    constant SoftmaxBwdParams& p  [[buffer(2)]],
    uint2 tgpos [[threadgroup_position_in_grid]],
    uint  lid   [[thread_index_in_threadgroup]],
    threadgroup float* sh         [[threadgroup(0)]]
) {
    uint row  = tgpos.y;
    uint rlen = p.row_len;
    uint base = row * rlen;

    float gp  = (lid < rlen) ? grad_pre[base + lid] : 0.0f;
    float a   = (lid < rlen) ? attn    [base + lid] : 0.0f;

    // dot = sum_j(grad_pre[j] * attn[j])
    sh[lid] = gp * a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = rlen >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = sh[0];

    if (lid < rlen) grad_pre[base + lid] = a * (gp - dot);
}

// Scale elementwise: out[i] *= scale  (used to apply attn_scale to grad_q / grad_k)
struct ScaleInplaceParams { uint length; float scale; };
kernel void scale_inplace_f32(
    device float* x                   [[buffer(0)]],
    constant ScaleInplaceParams& p    [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    x[gid] *= p.scale;
}

// grad_q[i, h*d_head+dk] += sum_j(grad_attn[h,i,j] * K[j, h*d_head+dk]) * scale
// Grid: (d_head, seq, n_heads)
struct AttnGradQKParams { uint seq; uint n_heads; uint d_head; float scale; };

kernel void attn_grad_q(
    device const float* grad_attn [[buffer(0)]],  // [n_heads × seq × seq] (after softmax bwd)
    device const float* K         [[buffer(1)]],
    device atomic_float* grad_q   [[buffer(2)]],
    constant AttnGradQKParams& p  [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]          // x=dk, y=i, z=h
) {
    uint dk = gid.x, i = gid.y, h = gid.z;
    if (dk >= p.d_head || i >= p.seq || h >= p.n_heads) return;

    uint dm = p.n_heads * p.d_head;
    uint ho = h * p.d_head;
    float s = 0.0f;
    for (uint j = 0; j < p.seq; j++)
        s += grad_attn[h * p.seq * p.seq + i * p.seq + j] * K[j * dm + ho + dk];
    atomic_fetch_add_explicit(&grad_q[i * dm + ho + dk], s * p.scale, memory_order_relaxed);
}

kernel void attn_grad_k(
    device const float* grad_attn [[buffer(0)]],  // [n_heads × seq × seq]
    device const float* Q         [[buffer(1)]],
    device atomic_float* grad_k   [[buffer(2)]],
    constant AttnGradQKParams& p  [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]          // x=dk, y=j, z=h
) {
    uint dk = gid.x, j = gid.y, h = gid.z;
    if (dk >= p.d_head || j >= p.seq || h >= p.n_heads) return;

    uint dm = p.n_heads * p.d_head;
    uint ho = h * p.d_head;
    float s = 0.0f;
    for (uint i = 0; i < p.seq; i++)
        s += grad_attn[h * p.seq * p.seq + i * p.seq + j] * Q[i * dm + ho + dk];
    atomic_fetch_add_explicit(&grad_k[j * dm + ho + dk], s * p.scale, memory_order_relaxed);
}

// ── Embedding 

struct EmbedParams { uint d_model; };

// Forward: output[row, col] = table[indices[row], col]
// Grid: (d_model, num_tokens)
kernel void embedding_fwd(
    device const uint*  indices  [[buffer(0)]],
    device const float* table    [[buffer(1)]],
    device       float* output   [[buffer(2)]],
    constant EmbedParams& p      [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x, row = gid.y;
    if (col >= p.d_model) return;
    output[row * p.d_model + col] = table[indices[row] * p.d_model + col];
}

// Backward: grad_table[indices[row], col] += grad_out[row, col]
// Grid: (d_model, num_tokens)
kernel void embedding_bwd(
    device const uint*   indices    [[buffer(0)]],
    device const float*  grad_out   [[buffer(1)]],
    device atomic_float* grad_table [[buffer(2)]],
    constant EmbedParams& p         [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x, row = gid.y;
    if (col >= p.d_model) return;
    atomic_fetch_add_explicit(
        &grad_table[indices[row] * p.d_model + col],
        grad_out[row * p.d_model + col],
        memory_order_relaxed);
}

// ── Dropout 

struct DropoutFwdParams { uint length; float scale; uint training; };

// mask is uchar (bool, 1 byte): nonzero = keep
kernel void dropout_fwd(
    device const float* input    [[buffer(0)]],
    device const uchar* mask     [[buffer(1)]],
    device       float* output   [[buffer(2)]],
    constant DropoutFwdParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    output[gid] = (p.training && !mask[gid]) ? 0.0f : input[gid] * (p.training ? p.scale : 1.0f);
}

struct DropoutBwdParams { uint length; float scale; };

kernel void dropout_bwd(
    device const float* grad_out [[buffer(0)]],
    device const uchar* mask     [[buffer(1)]],
    device       float* grad_in  [[buffer(2)]],
    constant DropoutBwdParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= p.length) return;
    grad_in[gid] = mask[gid] ? grad_out[gid] * p.scale : 0.0f;
}

// ── Loss 
//
// Both loss kernels run as a SINGLE threadgroup whose size is the next
// power-of-2 >= length.  Threads with lid >= length contribute 0 / identity.
// Threadgroup memory = tg_size * sizeof(float).

struct LossFusedParams { uint length; float inv; };

// MSE: grad[i] = 2*(out[i]-tgt[i])*inv,  loss = sum((out[i]-tgt[i])^2)
kernel void mse_loss(
    device const float* output  [[buffer(0)]],
    device const float* target  [[buffer(1)]],
    device       float* grad    [[buffer(2)]],
    device       float* loss    [[buffer(3)]],   // 1-element output
    constant LossFusedParams& p      [[buffer(4)]],
    uint lid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* sh [[threadgroup(0)]]
) {
    float d = (lid < p.length) ? output[lid] - target[lid] : 0.0f;
    if (lid < p.length) grad[lid] = 2.0f * d * p.inv;
    sh[lid] = d * d;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) loss[0] = sh[0];
}

// Cross-entropy with fused softmax:
//   soft[i] = exp(out[i]-max) / sum_exp
//   grad[i]  = (soft[i] - tgt[i]) * inv
//   loss     = -sum(tgt[i] * log(soft[i] + 1e-9))
kernel void cross_entropy_loss(
    device const float* output  [[buffer(0)]],
    device const float* target  [[buffer(1)]],
    device       float* grad    [[buffer(2)]],
    device       float* loss    [[buffer(3)]],
    constant LossFusedParams& p      [[buffer(4)]],
    uint lid     [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* sh [[threadgroup(0)]]
) {
    uint len = p.length;

    // 1. max reduction
    sh[lid] = (lid < len) ? output[lid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] = max(sh[lid], sh[lid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. sum_exp reduction
    sh[lid] = (lid < len) ? exp(output[lid] - max_val) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. grad + loss reduction
    float tgt  = (lid < len) ? target[lid] : 0.0f;
    float soft = (lid < len) ? exp(output[lid] - max_val) / sum_exp : 0.0f;
    if (lid < len) grad[lid] = (soft - tgt) * p.inv;
    sh[lid] = (tgt > 0.0f) ? -tgt * log(soft + 1e-9f) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size >> 1; s > 0; s >>= 1) {
        if (lid < s) sh[lid] += sh[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) loss[0] = sh[0];
}
