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
