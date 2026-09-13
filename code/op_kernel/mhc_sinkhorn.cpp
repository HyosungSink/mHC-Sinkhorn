// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

template <class DT_LOGITS>
class KernelMhcSinkhorn {
public:
    __aicore__ inline KernelMhcSinkhorn() {}
    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights, uint32_t length) {

    }
    __aicore__ inline void Process() {

    }
private:

};

template <typename DT_LOGITS>
 __global__ __aicore__ void mhc_sinkhorn(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcSinkhornTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcSinkhornTilingData, tiling_data, tiling);
    KernelMhcSinkhorn<DT_LOGITS> op;
    op.Init(logits, mask, weights, tiling_data.length);
    op.Process();
}
