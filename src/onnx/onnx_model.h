#ifndef CLPEAK_ONNX_MODEL_H
#define CLPEAK_ONNX_MODEL_H

// In-memory ONNX model construction.  The graphs clpeak benchmarks are a
// handful of nodes, so it emits the protobuf wire format directly -- no
// protobuf library, no .onnx files on disk, and byte-identical models on
// every platform, which is what makes the cross-vendor comparison mean
// anything.

#include <cstdint>
#include <string>
#include <vector>

// ONNX TensorProto.DataType values.  The ONNXTensorElementDataType enum in
// the ORT C API uses the same numbering, so these cross over directly.
enum OnnxDtype : int
{
  ONNX_DT_FLOAT    = 1,
  ONNX_DT_UINT8    = 2,
  ONNX_DT_INT8     = 3,
  ONNX_DT_INT32    = 6,
  ONNX_DT_FLOAT16  = 10,
  ONNX_DT_BFLOAT16 = 16,
};

using OnnxDims = std::vector<int64_t>;   // empty = scalar

// Builds one GraphProto and wraps it in a ModelProto.  Nodes must be added
// in topological order (ONNX requires it, and no sort is done here).
class OnnxGraph
{
public:
  void input(const std::string &name, int dtype, const OnnxDims &dims);
  void output(const std::string &name, int dtype, const OnnxDims &dims);

  // Constant tensor embedded in the model.  `raw` is little-endian element
  // data, exactly as ONNX raw_data expects.
  void initializer(const std::string &name, int dtype, const OnnxDims &dims,
                   const std::string &raw);

  void node(const std::string &opType,
            const std::vector<std::string> &inputs,
            const std::vector<std::string> &outputs);

  std::string build() const;   // ModelProto bytes

private:
  std::string m_nodes, m_inits, m_inputs, m_outputs;
  int         m_nodeCount = 0;
};

// C[M,N] = MatMul(A[M,K], B[K,N]) in `dtype`.  A is a graph input; B is an
// embedded initializer, so the EP sees it as constant weights it may
// pre-pack -- the weight-stationary GEMM inference is built from.
std::string onnxMatMulModel(int64_t M, int64_t K, int64_t N, int dtype,
                            const std::string &weightRaw);

// The same GEMM in QDQ form, the shape every NPU actually wants:
//
//   A_q(int8) -> DequantizeLinear -\
//                                   MatMul -> QuantizeLinear -> C_q(int8)
//   B_q(int8) -> DequantizeLinear -/
//
// Input and output are int8, so no float conversion sits in the timed path.
// ORT's QDQ handling fuses this into the EP's own quantized matmul.
std::string onnxQdqMatMulModel(int64_t M, int64_t K, int64_t N,
                               const std::string &weightRawInt8,
                               float aScale, float bScale, float cScale);

// Scalar float -> IEEE fp16 / bfloat16 bit conversion for weight buffers.
uint16_t floatToHalf(float f);
uint16_t floatToBf16(float f);
float    halfToFloat(uint16_t h);

#endif // CLPEAK_ONNX_MODEL_H
