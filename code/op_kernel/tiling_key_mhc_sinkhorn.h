// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

// ORDER_SEL: 0 表示通用兜底路径（阶数在运行期读取），4/6/8 为特化路径。
// SPEC_SEL: 1 表示每个逻辑Block恰好负责一个方阵，源地址只由BlockIdx决定，
//           搬入可以先于tiling读取下发；0 为一般切分。
ASCENDC_TPL_ARGS_DECL(MhcSinkhorn,
    ASCENDC_TPL_DATATYPE_DECL(DT_LOGITS, C_DT_FLOAT16, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(ORDER_SEL, 4, ASCENDC_TPL_UI_LIST, 0, 4, 6, 8),
    ASCENDC_TPL_UINT_DECL(SPEC_SEL, 1, ASCENDC_TPL_UI_LIST, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_LOGITS, C_DT_FLOAT16, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(ORDER_SEL, ASCENDC_TPL_UI_LIST, 0, 4, 6, 8),
        ASCENDC_TPL_UINT_SEL(SPEC_SEL, ASCENDC_TPL_UI_LIST, 0, 1),
    ),
);
