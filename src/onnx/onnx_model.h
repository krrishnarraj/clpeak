#ifndef CLPEAK_ONNX_MODEL_H
#define CLPEAK_ONNX_MODEL_H

// In-memory ONNX model construction.  A single-op model is a few hundred
// bytes of protobuf plus the embedded weights, so clpeak emits the wire
// format directly -- no protobuf library, no .onnx files on disk, and the
// exact same model bytes on every platform.

#include <cstdint>
#include <string>

// ONNX TensorProto.DataType values used by the emitters (the ONNXTensorElement
// DataType enum in the ORT C API uses the same numbering for these).
enum OnnxDtype : int
{
  ONNX_DT_FLOAT   = 1,
  ONNX_DT_UINT8   = 2,
  ONNX_DT_INT8    = 3,
  ONNX_DT_FLOAT16 = 10,
  ONNX_DT_BFLOAT16 = 16,
};

// Model: C[M,N] = MatMul(A[M,K], B[K,N]).  A is a graph input; B is an
// embedded initializer (`weightRaw`, M*K elements of `dtype`), so an EP sees
// it as constant weights and can pre-pack them -- the weight-stationary GEMM
// every inference workload is built from.  Opset 17, IR version 8.
std::string onnxMatMulModel(int64_t M, int64_t K, int64_t N, int dtype,
                            const std::string &weightRaw);

// Scalar float -> IEEE fp16 / bfloat16 bit conversion for weight buffers.
uint16_t floatToHalf(float f);
uint16_t floatToBf16(float f);

#endif // CLPEAK_ONNX_MODEL_H
