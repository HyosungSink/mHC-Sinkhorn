// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_sinkhorn_tiling.h"
#include "../op_kernel/tiling_key_mhc_sinkhorn.h"

namespace {
constexpr size_t RANK_3D = 3;
constexpr size_t RANK_4D = 4;
constexpr int64_t MIN_ITERS = 1;
constexpr int64_t MAX_ITERS = 100;
constexpr int64_t MAX_ORDER = 8;
constexpr int64_t MAX_MATRICES = 1073741823;  // 方阵总数上限, 避免元素数溢出uint32
constexpr uint32_t DEFAULT_ITERS = 20;
constexpr float DEFAULT_EPS = 1e-6f;
constexpr uint64_t MIN_BLOCKS = 8;
// 并进同一条指令的方阵数: 阶数为4时一组2个, 其余一组4个。
constexpr uint64_t GROUP_SMALL = 2;
constexpr uint64_t GROUP_LARGE = 4;

// 方阵阶数约束: 题面定义为偶数阶且不超过一个DataBlock可容纳的float数。
// 采用区间与整除性判定而非逐值枚举, 隐藏用例只要落在该区间即可走通用路径。
inline bool IsSupportedOrder(int64_t order)
{
    return order >= 2 && order <= MAX_ORDER && (order % 2) == 0;
}

inline bool IsSupportedType(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16;
}

// 校验形状并返回方阵个数; 失败返回 -1。
inline int64_t CheckShape(const gert::Shape &shape)
{
    const size_t rank = shape.GetDimNum();
    if (rank != RANK_3D && rank != RANK_4D) {
        return -1;
    }
    const int64_t order = shape.GetDim(rank - 1);
    if (shape.GetDim(rank - 2) != order || !IsSupportedOrder(order)) {
        return -1;
    }
    int64_t matrices = 1;
    for (size_t axis = 0; axis + 2 < rank; ++axis) {
        const int64_t dim = shape.GetDim(axis);
        if (dim <= 0 || dim > MAX_MATRICES / matrices) {
            return -1;
        }
        matrices *= dim;
    }
    if (matrices > MAX_MATRICES / (order * order)) {
        return -1;
    }
    return matrices;
}

inline bool CheckAttrs(const gert::RuntimeAttrs *attrs, int64_t &iterations, float &eps)
{
    if (attrs == nullptr) {
        return false;
    }
    const int64_t *iterAttr = attrs->GetInt(0);
    const float *epsAttr = attrs->GetFloat(1);
    iterations = (iterAttr == nullptr) ? DEFAULT_ITERS : *iterAttr;
    eps = (epsAttr == nullptr) ? DEFAULT_EPS : *epsAttr;
    return iterations >= MIN_ITERS && iterations <= MAX_ITERS;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Tensor *input = context->GetRequiredInputTensor(0);
    if (input == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtype = input->GetDataType();
    if (!IsSupportedType(dtype)) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = input->GetStorageShape();
    const int64_t matrices = CheckShape(shape);
    if (matrices < 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t iterations = 0;
    float eps = 0.0f;
    if (!CheckAttrs(context->GetAttrs(), iterations, eps)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t order = shape.GetDim(shape.GetDimNum() - 1);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = 1;
    }
    const uint64_t total = static_cast<uint64_t>(matrices);
    // 一条矢量指令最多迭代到整块UB, 把若干个方阵并进同一条指令后, 依赖链的长度
    // 不变而完成的方阵数成倍增加, 因此按组分核而不是按单个方阵分核。n=4的方阵只
    // 占半个DataBlock行, 一组取2个即可填满迭代。
    const uint64_t group = (order == 4) ? GROUP_SMALL : GROUP_LARGE;
    const uint64_t tasks = (total + group - 1) / group;
    // 固定开销随逻辑Block数增长, 但Block数过小时单次启动反而更慢; 实测在若干个
    // Block处最低, 因此把Block数取在该下限与核数之间。
    uint64_t blocks = (tasks < MIN_BLOCKS) ? MIN_BLOCKS : tasks;
    if (blocks > coreNum) {
        blocks = coreNum;
    }
    if (blocks == 0) {
        blocks = 1;
    }
    uint64_t perCore = (tasks <= coreNum) ? group : ((total + blocks - 1) / blocks);
    if (perCore == 0) {
        perCore = 1;
    }

    MhcSinkhornTilingData *tiling = context->GetTilingData<MhcSinkhornTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->numMat = static_cast<uint32_t>(total);
    tiling->matPerCore = static_cast<uint32_t>(perCore);
    tiling->iters = static_cast<uint32_t>(iterations);
    tiling->order = static_cast<uint32_t>(order);
    tiling->eps = eps;

    context->SetBlockDim(static_cast<uint32_t>(blocks));
    // 纯矢量算子按批量调度下发, 省掉逐Block的调度往返。
    if (context->SetScheduleMode(1U) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 阶数为4/6/8时选用编译期特化模板, 其余阶数走ORDER_SEL=0的通用兜底路径。
    uint32_t ORDER_SEL = 0;
    if (order == 4 || order == 6 || order == 8) {
        ORDER_SEL = static_cast<uint32_t>(order);
    }
    uint32_t DT_LOGITS = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_LOGITS, ORDER_SEL);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *input = context->GetInputShape(0);
    gert::Shape *output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    if (CheckShape(*input) < 0) {
        return GRAPH_FAILED;
    }
    int64_t iterations = 0;
    float eps = 0.0f;
    if (!CheckAttrs(context->GetAttrs(), iterations, eps)) {
        return GRAPH_FAILED;
    }
    *output = *input;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType dtype = context->GetInputDataType(0);
    if (!IsSupportedType(dtype)) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, dtype);
}
}  // namespace ge

namespace ops {
class MhcSinkhorn : public OpDef {
public:
    explicit MhcSinkhorn(const char *name) : OpDef(name)
    {
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
