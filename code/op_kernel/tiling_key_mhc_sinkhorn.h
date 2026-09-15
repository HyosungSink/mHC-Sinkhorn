// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(MhcSinkhorn,
    ASCENDC_TPL_DATATYPE_DECL(DT_LOGITS, C_DT_FLOAT16, C_DT_FLOAT),
    ASCENDC_TPL_UINT_DECL(GROUP_ORDER, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_LIST, 0, 4, 6, 8),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_LOGITS, C_DT_FLOAT16, C_DT_FLOAT),
        ASCENDC_TPL_UINT_SEL(GROUP_ORDER, ASCENDC_TPL_UI_LIST, 0, 4, 6, 8),
        // The operator is pure vector work: announcing it keeps the scheduler
        // from starting cube cores that have nothing to run, which is pure head
        // cost at this kernel size.
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
    ),
);