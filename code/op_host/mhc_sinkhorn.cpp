// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

#include "../op_kernel/mhc_sinkhorn_tiling.h"
#include "../op_kernel/tiling_key_mhc_sinkhorn.h"

namespace optiling {
namespace {
constexpr uint32_t kMinOrder = 4;
constexpr uint32_t kMaxOrder = 8;
constexpr uint32_t kMaxIterations = 100;
constexpr uint32_t kSingleMatrixDims = 3;
constexpr uint32_t kBatchMatrixDims = 4;
constexpr uint32_t kTileWidth = 8;    // padded tile row width in fp32 lanes
constexpr uint32_t kSentinelSlot = 60;  // compact slot the padding lanes read (-inf)
constexpr uint32_t kHeadBlocks = 8;  // block count with the lowest measured head cost

bool SupportedOrder(int64_t order) {
    return order == 4 || order == 6 || order == 8;
}

bool SupportedDtype(ge::DataType dtype) {
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16;
}

// Number of independent n x n matrices in the leading dimensions.
bool MatrixCount(const gert::Shape &shape, uint32_t &matrices) {
    const size_t dims = shape.GetDimNum();
    if (dims != kSingleMatrixDims && dims != kBatchMatrixDims) {
        return false;
    }
    const int64_t order = shape.GetDim(static_cast<int32_t>(dims) - 1);
    if (!SupportedOrder(order) || shape.GetDim(static_cast<int32_t>(dims) - 2) != order) {
        return false;
    }
    uint64_t total = 1;
    for (size_t axis = 0; axis + 2 < dims; ++axis) {
        const int64_t dim = shape.GetDim(static_cast<int32_t>(axis));
        if (dim <= 0) {
            return false;
        }
        total *= static_cast<uint64_t>(dim);
        if (total > 0x3FFFFFFFULL) {
            return false;
        }
    }
    matrices = static_cast<uint32_t>(total);
    return true;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (tensor_x == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (!SupportedDtype(tensor_x->GetDataType())) {
        return ge::GRAPH_FAILED;
    }
    uint32_t matrices = 0;
    if (!MatrixCount(tensor_x->GetOriginShape(), matrices)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *attr_iterations = attrs->GetInt(0);
    const float *attr_eps = attrs->GetFloat(1);
    const int64_t iterations = attr_iterations == nullptr ? 20 : *attr_iterations;
    const float eps = attr_eps == nullptr ? 1.0e-6F : *attr_eps;
    if (iterations < 1 || iterations > static_cast<int64_t>(kMaxIterations)) {
        return ge::GRAPH_FAILED;
    }
    if (!(eps >= -1.0e38F && eps <= 1.0e38F)) {
        return ge::GRAPH_FAILED;
    }

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t order = static_cast<uint32_t>(tensor_x->GetOriginShape().GetDim(
        static_cast<int32_t>(tensor_x->GetOriginShape().GetDimNum()) - 1));
    const int32_t aivCores = platform.GetCoreNumAiv();
    const uint32_t cores = static_cast<uint32_t>(aivCores > 0 ? aivCores : 1);
    // A microsecond scale kernel pays for every block it starts, but the head
    // cost is not monotone in the block count: fewer than an octet of blocks is
    // both slower on average and bimodal, while the octet is the stable minimum
    // before the cost climbs again towards the full core count. Small batches
    // therefore ask for an octet of blocks and let the idle ones return
    // immediately; larger batches keep one matrix per block up to the core count.
    const uint32_t headBlocks = std::min<uint32_t>(kHeadBlocks, cores);
    const uint32_t groupMatrices = order == 4U ? 2U : 4U;
    const uint32_t tasks = (matrices + groupMatrices - 1U) / groupMatrices;
    const uint32_t blockCount = std::min(std::max(tasks, headBlocks), cores);

    const uint32_t DT_LOGITS = static_cast<uint32_t>(tensor_x->GetDataType());
    const uint32_t matricesPerCore = tasks <= cores ? groupMatrices : (matrices + blockCount - 1U) / blockCount;
    const uint32_t GROUP_ORDER = matricesPerCore == groupMatrices ? order : 0U;
    ASCENDC_TPL_SEL_PARAM(context, DT_LOGITS, GROUP_ORDER);

    MhcSinkhornTilingData *tiling = context->GetTilingData<MhcSinkhornTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->matrices = matrices;
    tiling->matricesPerCore = matricesPerCore;
    tiling->n = order;
    tiling->iterations = static_cast<uint32_t>(iterations);
    tiling->eps = eps;
    tiling->dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(tensor_x->GetDataType()));
    tiling->reserved0 = 0U;
    tiling->reserved1 = 0U;
    // Byte offsets of the two indexed copies that move a narrow matrix between
    // its compact n*n form and the padded 8x8 tile. Padding lanes and padding
    // rows all point at one sentinel slot past the matrix, which the kernel
    // primes with -inf so the softmax turns them into zeros.
    for (uint32_t slot = 0; slot < 64U; ++slot) {
        const uint32_t row = slot / kTileWidth;
        const uint32_t column = slot % kTileWidth;
        const bool inside = row < order && column < order;
        tiling->gather[slot] = (inside ? (row * order + column) : kSentinelSlot) *
                               static_cast<uint32_t>(sizeof(float));
    }
    for (uint32_t slot = 0; slot < 64U; ++slot) {
        uint32_t offset = 0U;
        if (slot < order * order) {
            offset = (slot / order) * kTileWidth + (slot % order);
        }
        tiling->gather[64U + slot] = offset * static_cast<uint32_t>(sizeof(float));
    }

    for (uint32_t slot = 0; slot < 128U; ++slot) {
        const uint32_t batch = slot / 64U;
        const uint32_t row = (slot % 64U) / 8U;
        const uint32_t column = slot % 8U;
        tiling->pairGather[slot] = (row < order && column < order ?
            batch * order * order + row * order + column : 252U) * sizeof(float);
    }
    for (uint32_t slot = 0; slot < 80U; ++slot) {
        const uint32_t batch = slot / (order * order);
        const uint32_t row = (slot % (order * order)) / order;
        const uint32_t column = slot % order;
        tiling->pairGather[128U + slot] = (slot < 2U * order * order ?
            batch * 64U + row * 8U + column : 0U) * sizeof(float);
    }

    context->SetBlockDim(blockCount);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *outShape = context->GetOutputShape(0);
    if (xShape == nullptr || outShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class MhcSinkhorn : public OpDef {
public:
    explicit MhcSinkhorn(const char *name) : OpDef(name) {
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("mask")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("iterations").AttrType(OPTIONAL).Int(20);
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-06);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(MhcSinkhorn);
}  // namespace ops
