#include "base/context_builder/op_tiling_context_builder.h"
#include "base/context_builder/op_infer_shape_context_builder.h"
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "platform/platform_info.h"
#include "../../code/op_host/mhc_sinkhorn.cpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// OpDef's Tiling registration uses a 2048-byte payload capacity by default.
constexpr size_t kTilingCapacity = 2048;
constexpr size_t kWorkspaceCapacity = 16;
// CANN's SIMD launch interface accepts logical Block counts in [1, 65535].
constexpr uint32_t kMaxLogicalBlocks = 65535;
// OpDef registration supplies an empty compile-info object to Tiling.
struct CompileInfo {};

void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

std::string Describe(const gert::Shape &shape, ge::DataType type, int64_t iterations, float eps) {
    std::ostringstream text;
    text << "shape=(";
    for (size_t axis = 0; axis < shape.GetDimNum(); ++axis) {
        if (axis) text << ',';
        text << shape.GetDim(axis);
    }
    text << "), dtype=" << static_cast<int>(type)
         << ", iterations=" << iterations << ", eps=" << eps;
    return text.str();
}

int CheckHost() {
    fe::PlatFormInfos platform;
    fe::OptionalInfos optional;
    auto &manager = fe::PlatformInfoManager::Instance();
    Require(manager.InitializePlatformInfo() == 0 &&
            manager.GetPlatformInfos("Ascend910B3", platform, optional) == 0, "platform metadata unavailable");

    ops::MhcSinkhorn definition("MhcSinkhorn");
    const auto inferShape = definition.GetInferShape();
    const auto inferType = definition.GetInferDataType();
    const auto tiling = definition.AICore().GetTiling();
    Require(inferShape && inferType && tiling, "Host inference and Tiling callbacks must be registered");

    size_t checked = 0;
    std::vector<std::string> failures;
    auto check = [&](const gert::Shape &shape, ge::DataType type, int64_t iterations,
                     float eps, bool valid) {
        ++checked;
        try {
            gert::StorageShape inputStorage;
            inputStorage.MutableOriginShape() = shape;
            inputStorage.MutableStorageShape() = shape;
            gert::Tensor input(inputStorage, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {}),
                               gert::kOnHost, type, nullptr);
            std::vector<gert::Tensor *> inputs{&input};

            gert::OpInferShapeContextBuilder shapeBuilder;
            auto shapeHolder = shapeBuilder.OpType("MhcSinkhorn").OpName("host_shape").IONum(1, 1)
                .AppendAttr(iterations).AppendAttr(eps).InputTensors(inputs)
                .OutputTensorDesc(0, type, ge::FORMAT_ND, ge::FORMAT_ND).Build();
            auto *shapeContext = shapeHolder.GetContext();
            Require(shapeContext && shapeContext->GetOutputShape(0), "shape context unavailable");
            *shapeContext->GetOutputShape(0) = gert::Shape{};
            if (inferShape(shapeContext) != ge::GRAPH_SUCCESS) {
                Require(!valid, "shape inference rejected a valid input");
                return;
            }
            if (valid) Require(*shapeContext->GetOutputShape(0) == shape,
                               "output shape must match input shape; inference did not populate it correctly");

            gert::OpInferDataTypeContextBuilder typeBuilder;
            auto typeHolder = typeBuilder.OpType("MhcSinkhorn").OpName("host_dtype").IONum(1, 1)
                .AppendAttr(iterations).AppendAttr(eps)
                .InputTensorDesc(0, type, ge::FORMAT_ND, ge::FORMAT_ND)
                .OutputTensorDesc(0, ge::FORMAT_ND, ge::FORMAT_ND).Build();
            auto *typeContext = typeHolder.GetContext();
            Require(typeContext != nullptr, "dtype context unavailable");
            Require(typeContext->SetOutputDataType(0, ge::DT_UNDEFINED) == ge::GRAPH_SUCCESS,
                    "cannot initialize output dtype");
            if (inferType(typeContext) != ge::GRAPH_SUCCESS) {
                Require(!valid, "dtype inference rejected a valid input");
                return;
            }
            if (valid) Require(typeContext->GetOutputDataType(0) == type,
                               "output dtype must match input dtype; inference did not populate it correctly");

            gert::StorageShape outputStorage;
            outputStorage.MutableOriginShape() = *shapeContext->GetOutputShape(0);
            outputStorage.MutableStorageShape() = *shapeContext->GetOutputShape(0);
            gert::Tensor output(outputStorage, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, {}),
                                gert::kOnHost, typeContext->GetOutputDataType(0), nullptr);
            auto workspaceStorage = gert::ContinuousVector::Create<size_t>(kWorkspaceCapacity);
            auto *workspace = reinterpret_cast<gert::ContinuousVector *>(workspaceStorage.get());
            Require(workspace != nullptr, "workspace metadata allocation failed");
            std::fill_n(static_cast<size_t *>(workspace->MutableData()), kWorkspaceCapacity,
                        std::numeric_limits<size_t>::max());
            CompileInfo compileInfo;
            gert::OpTilingContextBuilder tilingBuilder;
            auto tilingHolder = tilingBuilder.OpType("MhcSinkhorn").OpName("host_tiling").IONum(1, 1)
                .AppendAttr(iterations).AppendAttr(eps).InputTensors(inputs).OutputTensors({&output})
                .PlatformInfo(&platform).CompileInfo(&compileInfo)
                .Workspace(workspace).TilingDataSize(kTilingCapacity).Build();
            auto *context = tilingHolder.GetContext();
            Require(context != nullptr, "tiling context unavailable");
            Require(context->SetBlockDim(0) == ge::GRAPH_SUCCESS, "cannot initialize Block count");
            if (tiling(context) != ge::GRAPH_SUCCESS) {
                Require(!valid, "Tiling rejected a valid input");
                return;
            }
            Require(valid, "invalid input accepted by every Host callback");

            const auto blocks = context->GetBlockDim();
            Require(blocks > 0 && blocks <= kMaxLogicalBlocks, "invalid logical Block count");
            const auto *payload = context->GetRawTilingData();
            Require(payload && payload->GetDataSize() <= payload->GetCapacity(),
                    "Tiling payload exceeds the supplied buffer");
            const auto workspaceCount = context->GetWorkspaceNum();
            Require(workspaceCount <= kWorkspaceCapacity, "workspace metadata exceeds the supplied capacity");
            if (workspaceCount) {
                const auto *sizes = context->GetWorkspaceSizes(workspaceCount);
                Require(sizes != nullptr, "workspace sizes unavailable");
                size_t total = 0;
                for (size_t i = 0; i < workspaceCount; ++i) {
                    Require(sizes[i] != std::numeric_limits<size_t>::max(), "workspace size was not initialized");
                    Require(sizes[i] <= std::numeric_limits<size_t>::max() - total,
                            "workspace size sum overflows");
                    total += sizes[i];
                }
            }
        } catch (const std::exception &error) {
            failures.push_back(Describe(shape, type, iterations, eps) + ": " + error.what());
        }
    };

    for (auto type : {ge::DT_FLOAT, ge::DT_FLOAT16})
        for (int64_t n : {4, 6, 8})
            for (int64_t it : {1, 2, 20, 100})
                for (const auto &shape : std::vector<gert::Shape>{{1,n,n},{41,n,n},{4097,n,n},{3,5,n,n}})
                    check(shape, type, it, 1e-6F, true);
    for (const auto &shape : std::vector<gert::Shape>{{4,4},{1,1,1,4,4},{2,4,6},{1,5,5},{1,7,7},
            {0,4,4},{-1,4,4},{INT64_MAX,INT64_MAX,8,8},{INT64_MAX,8,8}})
        check(shape, ge::DT_FLOAT, 20, 1e-6F, false);
    for (auto type : {ge::DT_DOUBLE, ge::DT_INT32, ge::DT_BF16})
        check({1,4,4}, type, 20, 1e-6F, false);
    for (int64_t it : {0,101}) check({1,4,4}, ge::DT_FLOAT, it, 1e-6F, false);
    for (float eps : {0.F,1e-35F,1e-30F,0.25F,1.F,2.F})
        check({81,6,6}, ge::DT_FLOAT, 3, eps, true);

    for (size_t i = 0; i < std::min<size_t>(failures.size(), 12); ++i)
        std::cerr << "FAIL " << failures[i] << '\n';
    if (failures.size() > 12) std::cerr << "... " << failures.size() - 12 << " more failing scenarios\n";
    std::cout << (failures.empty() ? "PASS" : "FAIL") << " Host contract: "
              << checked - failures.size() << '/' << checked << " scenarios\n";
    return failures.empty() ? 0 : 1;
}

}  // namespace

int main() {
    try {
        return CheckHost();
    } catch (const std::exception &error) {
        std::cerr << "Host contract: " << error.what() << '\n';
        return 1;
    }
}
