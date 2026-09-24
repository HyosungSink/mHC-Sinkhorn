// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

namespace mhcs {

constexpr uint32_t BLK = 8;            // 一个32B DataBlock容纳的float个数
constexpr uint32_t MAX_ORDER = 8;      // 支持的最大方阵阶数
constexpr uint32_t CHUNK = 64;         // 单轮驻留UB的方阵数上限
constexpr uint32_t MAX_REPEAT = 255;   // 单条矢量指令的最大迭代数

// UB分区（单位: float元素），按CHUNK个方阵的最坏情况预留。
constexpr uint32_t MAT_CAP = CHUNK * MAX_ORDER * BLK;   // 主矩阵缓冲
constexpr uint32_t BCT_CAP = CHUNK * MAX_ORDER * BLK;   // Brcb展开缓冲
constexpr uint32_t VEC_CAP = CHUNK * BLK;               // 行/列和缓冲
constexpr uint32_t T1_CAP = CHUNK * 4 * BLK;
constexpr uint32_t T2_CAP = CHUNK * 2 * BLK;

constexpr uint32_t OFF_MAT = 0;
constexpr uint32_t OFF_BCT = OFF_MAT + MAT_CAP;
constexpr uint32_t OFF_RS = OFF_BCT + BCT_CAP;
constexpr uint32_t OFF_CS = OFF_RS + VEC_CAP;
constexpr uint32_t OFF_T1 = OFF_CS + VEC_CAP;
constexpr uint32_t OFF_T2 = OFF_T1 + T1_CAP;
constexpr uint32_t OFF_STG = OFF_T2 + T2_CAP;           // 半精度落地区(以float计)
constexpr uint32_t STG_CAP = CHUNK * MAX_ORDER * BLK;
constexpr uint32_t UB_FLOATS = OFF_STG + STG_CAP;

// 每个DataBlock内前 n 个lane置位的逐bit掩码。
__aicore__ inline uint64_t LaneMask(uint32_t n, uint32_t blocks)
{
    uint64_t lane = (n >= BLK) ? 0xFFULL : ((1ULL << n) - 1ULL);
    uint64_t value = 0;
    for (uint32_t i = 0; i < blocks; ++i) {
        value |= lane << (i * BLK);
    }
    return value;
}

}  // namespace mhcs

template <typename DT_LOGITS, uint32_t ORDER_SEL>
class KernelMhcSinkhorn {
public:
    __aicore__ inline KernelMhcSinkhorn() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR weights,
                                const MhcSinkhornTilingData &tiling)
    {
        order_ = (ORDER_SEL != 0) ? ORDER_SEL : tiling.order;
        iters_ = tiling.iters;
        eps_ = tiling.eps;

        const uint32_t total = tiling.numMat;
        const uint32_t per = tiling.matPerCore;
        const uint32_t start = static_cast<uint32_t>(AscendC::GetBlockIdx()) * per;
        if (start >= total) {
            count_ = 0;
            return;
        }
        count_ = total - start;
        if (count_ > per) {
            count_ = per;
        }
        const uint64_t elems = static_cast<uint64_t>(order_) * order_;
        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(logits) + start * elems);
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(weights) + start * elems);

        pipe_.InitBuffer(buf_, mhcs::UB_FLOATS * sizeof(float));
        AscendC::LocalTensor<float> base = buf_.Get<float>();
        mat_ = base[mhcs::OFF_MAT];
        bct_ = base[mhcs::OFF_BCT];
        rs_ = base[mhcs::OFF_RS];
        cs_ = base[mhcs::OFF_CS];
        t1_ = base[mhcs::OFF_T1];
        t2_ = base[mhcs::OFF_T2];
        stage_ = base[mhcs::OFF_STG].template ReinterpretCast<DT_LOGITS>();
    }

    __aicore__ inline void Process()
    {
        if (count_ == 0) {
            return;
        }
        const uint32_t rowBlocks = order_;              // 每个方阵占用的DataBlock数
        const uint32_t matStride = rowBlocks * mhcs::BLK;
        const uint32_t laneMaskLow = 0;                 // 占位, 实际掩码在下方构造
        (void)laneMaskLow;

        for (uint32_t done = 0; done < count_; done += mhcs::CHUNK) {
            uint32_t cnt = count_ - done;
            if (cnt > mhcs::CHUNK) {
                cnt = mhcs::CHUNK;
            }
            LoadChunk(done, cnt);
            Compute(cnt, matStride);
            StoreChunk(done, cnt);
        }
    }

private:
    // ---------------------------------------------------------------- 数据搬入
    __aicore__ inline void LoadChunk(uint32_t done, uint32_t cnt)
    {
        const uint32_t n = order_;
        const uint64_t elems = static_cast<uint64_t>(n) * n;
        // 先清零整块工作区，避免残留lane影响归约结果与确定性。
        AscendC::Duplicate(mat_, 0.0f, static_cast<int32_t>(cnt * n * mhcs::BLK));

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(cnt * n);
        copyParams.blockLen = n * sizeof(DT_LOGITS);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPadExtParams<DT_LOGITS> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.rightPadding = static_cast<uint8_t>(mhcs::BLK - n);
        padParams.paddingValue = static_cast<DT_LOGITS>(0.0f);

        if constexpr (AscendC::IsSameType<DT_LOGITS, float>::value) {
            AscendC::DataCopyPad(mat_, srcGm_[done * elems], copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            // 半精度: 每行落到32B对齐位置，再按行升精度到float布局。
            padParams.rightPadding = static_cast<uint8_t>(2 * mhcs::BLK - n);
            AscendC::DataCopyPad(stage_, srcGm_[done * elems], copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            uint32_t rows = cnt * n;
            for (uint32_t base = 0; base < rows; base += mhcs::MAX_REPEAT) {
                uint32_t step = rows - base;
                if (step > mhcs::MAX_REPEAT) {
                    step = mhcs::MAX_REPEAT;
                }
                AscendC::Cast(mat_[base * mhcs::BLK], stage_[base * 2 * mhcs::BLK],
                              AscendC::RoundMode::CAST_NONE, mhcs::BLK,
                              static_cast<uint8_t>(step), {1, 1, 1, 1});
            }
        }
    }

    // ---------------------------------------------------------------- 数据搬出
    __aicore__ inline void StoreChunk(uint32_t done, uint32_t cnt)
    {
        const uint32_t n = order_;
        const uint64_t elems = static_cast<uint64_t>(n) * n;
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(cnt * n);
        copyParams.blockLen = n * sizeof(DT_LOGITS);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        if constexpr (AscendC::IsSameType<DT_LOGITS, float>::value) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(dstGm_[done * elems], mat_, copyParams);
        } else {
            uint32_t rows = cnt * n;
            for (uint32_t base = 0; base < rows; base += mhcs::MAX_REPEAT) {
                uint32_t step = rows - base;
                if (step > mhcs::MAX_REPEAT) {
                    step = mhcs::MAX_REPEAT;
                }
                AscendC::Cast(stage_[base * 2 * mhcs::BLK], mat_[base * mhcs::BLK],
                              AscendC::RoundMode::CAST_RINT, mhcs::BLK,
                              static_cast<uint8_t>(step), {1, 1, 1, 1});
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(dstGm_[done * elems], stage_, copyParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    // ---------------------------------------------------------------- 主计算
    __aicore__ inline void Compute(uint32_t cnt, uint32_t matStride)
    {
        const uint32_t n = order_;
        const uint8_t rep = static_cast<uint8_t>(cnt);
        const int32_t fullMask = static_cast<int32_t>(n * mhcs::BLK);
        uint64_t colMask[2] = {mhcs::LaneMask(n, n), 0};

        // --- 初始化: 行方向softmax + eps ---
        AscendC::BlockReduceMax<float>(rs_, mat_, rep, colMask, 1, 1, static_cast<int32_t>(n));
        AscendC::Brcb(bct_, rs_, rep, {1, mhcs::BLK});
        AscendC::Sub(mat_, mat_, bct_, colMask, rep,
                     {1, 1, 1, static_cast<uint8_t>(n), static_cast<uint8_t>(n), mhcs::BLK});
        AscendC::Exp(mat_, mat_, colMask, rep,
                     {1, 1, static_cast<uint8_t>(n), static_cast<uint8_t>(n)});
        RowSum(cnt, rep, fullMask, n);
        RowDivide(rep, fullMask, n);
        AscendC::Adds(mat_, mat_, eps_, colMask, rep,
                      {1, 1, static_cast<uint8_t>(n), static_cast<uint8_t>(n)});

        ColumnStep(cnt, rep, fullMask, n);
        for (uint32_t i = 1; i < iters_; ++i) {
            RowStep(cnt, rep, fullMask, n);
            ColumnStep(cnt, rep, fullMask, n);
        }
    }

    __aicore__ inline void RowSum(uint32_t cnt, uint8_t rep, int32_t fullMask, uint32_t n)
    {
        AscendC::BlockReduceSum<float>(rs_, mat_, rep, fullMask, 1, 1, static_cast<int32_t>(n));
    }

    __aicore__ inline void RowDivide(uint8_t rep, int32_t fullMask, uint32_t n)
    {
        AscendC::Brcb(bct_, rs_, rep, {1, mhcs::BLK});
        AscendC::Div(mat_, mat_, bct_, fullMask, rep,
                     {1, 1, 1, static_cast<uint8_t>(n), static_cast<uint8_t>(n), mhcs::BLK});
    }

    __aicore__ inline void RowStep(uint32_t cnt, uint8_t rep, int32_t fullMask, uint32_t n)
    {
        RowSum(cnt, rep, fullMask, n);
        AscendC::Adds(rs_, rs_, eps_, static_cast<int32_t>(cnt * mhcs::BLK));
        RowDivide(rep, fullMask, n);
    }

    __aicore__ inline void ColumnStep(uint32_t cnt, uint8_t rep, int32_t fullMask, uint32_t n)
    {
        ColumnSum(rep, n);
        AscendC::Adds(cs_, cs_, eps_, static_cast<int32_t>(cnt * mhcs::BLK));
        AscendC::Div(mat_, mat_, cs_, fullMask, rep,
                     {1, 1, 0, static_cast<uint8_t>(n), static_cast<uint8_t>(n), 1});
    }

    // 列方向求和: 折叠 n 个DataBlock。
    __aicore__ inline void ColumnSum(uint8_t rep, uint32_t n)
    {
        if constexpr (ORDER_SEL == 8) {
            AscendC::Add(t1_, mat_, mat_[4 * mhcs::BLK], 32, rep, {1, 1, 1, 4, 8, 8});
            AscendC::Add(t2_, t1_, t1_[2 * mhcs::BLK], 16, rep, {1, 1, 1, 2, 4, 4});
            AscendC::Add(cs_, t2_, t2_[mhcs::BLK], 8, rep, {1, 1, 1, 1, 2, 2});
        } else if constexpr (ORDER_SEL == 6) {
            AscendC::Add(t1_, mat_, mat_[3 * mhcs::BLK], 24, rep, {1, 1, 1, 3, 6, 6});
            AscendC::Add(t2_, t1_, t1_[mhcs::BLK], 8, rep, {1, 1, 1, 1, 3, 3});
            AscendC::Add(cs_, t2_, t1_[2 * mhcs::BLK], 8, rep, {1, 1, 1, 1, 1, 3});
        } else if constexpr (ORDER_SEL == 4) {
            AscendC::Add(t1_, mat_, mat_[2 * mhcs::BLK], 16, rep, {1, 1, 1, 2, 4, 4});
            AscendC::Add(cs_, t1_, t1_[mhcs::BLK], 8, rep, {1, 1, 1, 1, 2, 2});
        } else {
            const uint8_t stride = static_cast<uint8_t>(n);
            AscendC::Add(cs_, mat_, mat_[mhcs::BLK], 8, rep, {1, 1, 1, 1, stride, stride});
            for (uint32_t r = 2; r < n; ++r) {
                AscendC::Add(cs_, cs_, mat_[r * mhcs::BLK], 8, rep, {1, 1, 1, 1, 1, stride});
            }
        }
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_;
    AscendC::GlobalTensor<DT_LOGITS> srcGm_;
    AscendC::GlobalTensor<DT_LOGITS> dstGm_;
    AscendC::LocalTensor<float> mat_;
    AscendC::LocalTensor<float> bct_;
    AscendC::LocalTensor<float> rs_;
    AscendC::LocalTensor<float> cs_;
    AscendC::LocalTensor<float> t1_;
    AscendC::LocalTensor<float> t2_;
    AscendC::LocalTensor<DT_LOGITS> stage_;
    uint32_t order_ = 0;
    uint32_t iters_ = 0;
    uint32_t count_ = 0;
    float eps_ = 0.0f;
};

template <typename DT_LOGITS, uint32_t ORDER_SEL>
__global__ __aicore__ void mhc_sinkhorn(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights,
                                        GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MhcSinkhornTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcSinkhornTilingData, tiling_data, tiling);
    KernelMhcSinkhorn<DT_LOGITS, ORDER_SEL> op;
    op.Init(logits, weights, tiling_data);
    op.Process();
}
