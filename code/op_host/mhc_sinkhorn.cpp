// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_sinkhorn_tiling.h"
#include "../op_kernel/tiling_key_mhc_sinkhorn.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_logits = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_mask = context->GetOptionalInputTensor(1);
        if(tensor_mask) {
            // 可选输入数组存在
        }
        ge::DataType dtype_logits = tensor_logits->GetDataType(); // 获取数据类型
        int dtype_size_logits = ge::GetSizeByDataType(dtype_logits); // 获取数据类型的字长
        uint32_t length_logits = tensor_logits->GetShapeSize(); // 获取元素个数
        uint32_t size_logits = tensor_logits->GetSize(); // 获取内存大小
        // 示例: 获取算子输入属性
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_iterations = attrs->GetInt(0);
        const float *attr_eps = attrs->GetFloat(1);
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_LOGITS = static_cast<uint32_t>(dtype_logits);
        ASCENDC_TPL_SEL_PARAM(context, DT_LOGITS);
        // 示例: 计算tiling方案并填充tiling结构体
        MhcSinkhornTilingData *tiling = context->GetTilingData<MhcSinkhornTilingData>();
        tiling->length = length_logits;
        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
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
