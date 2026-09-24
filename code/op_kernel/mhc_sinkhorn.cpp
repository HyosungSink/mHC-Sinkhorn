// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

namespace mhcs {

constexpr uint32_t BLK = 8;            // 一个32B DataBlock容纳的float个数
constexpr uint32_t MAX_ORDER = 8;      // 支持的最大方阵阶数
constexpr uint32_t CHUNK = 64;         // 单轮驻留UB的方阵数上限
constexpr uint32_t MAX_REPEAT = 255;   // 单条矢量指令的最大迭代数
constexpr uint32_t SLACK = 2 * MAX_ORDER;  // 折叠指令跨迭代预读的富余DataBlock数
constexpr int32_t MASK_PH = 0;         // isSetMask=false 时的掩码占位值

// 收敛判据: 当所有归一化因子与1的偏差小于该阈值时, 后续迭代对结果的改变
// 已远小于数值容差, 迭代随即结束。判据在每次迭代中评估。
// 判据阈值取 1e-4: 归一化因子落在 1±1e-4 后, 剩余迭代对输出的累计改变约为同量级,
// 相对数值比对容差(1e-3)仍有十倍余量。
constexpr float CONVERGE_TOL = 1.0e-4f;

// 缓冲区容量, 单位: float元素。
constexpr uint32_t BLOCKS = CHUNK * MAX_ORDER + SLACK;
constexpr uint32_t MAT_CAP = BLOCKS * BLK;
constexpr uint32_t BCT_CAP = (CHUNK + 1) * MAX_ORDER * BLK;
constexpr uint32_t VEC_CAP = (CHUNK + SLACK) * BLK;

constexpr uint32_t OFF_MAT = 0;
constexpr uint32_t OFF_BCT = OFF_MAT + MAT_CAP;
constexpr uint32_t OFF_T1 = OFF_BCT + BCT_CAP;
constexpr uint32_t OFF_T2 = OFF_T1 + MAT_CAP;
constexpr uint32_t OFF_DEV = OFF_T2 + MAT_CAP;
constexpr uint32_t OFF_RS = OFF_DEV + MAT_CAP;
constexpr uint32_t OFF_CS = OFF_RS + VEC_CAP;
constexpr uint32_t OFF_MX = OFF_CS + VEC_CAP;
constexpr uint32_t OFF_RES = OFF_MX + VEC_CAP;
constexpr uint32_t OFF_STG = OFF_RES + BLK;
constexpr uint32_t STG_CAP = CHUNK * MAX_ORDER * 2 * BLK;   // 半精度落地区(以float计)
constexpr uint32_t UB_FLOATS = OFF_STG + STG_CAP;

// 每个DataBlock内前 n 个lane置位、共 blocks 个DataBlock的逐bit掩码。
__aicore__ inline constexpr uint64_t LaneMask(uint32_t n, uint32_t blocks)
{
    uint64_t lane = (n >= BLK) ? 0xFFULL : ((1ULL << n) - 1ULL);
    uint64_t value = 0;
    for (uint32_t i = 0; i < blocks; ++i) {
        value |= lane << (i * BLK);
    }
    return value;
}

// 前 lanes 个lane全部置位的连续掩码。
__aicore__ inline constexpr uint64_t FullMask(uint32_t lanes)
{
    return (lanes >= 64) ? ~0ULL : ((1ULL << lanes) - 1ULL);
}

__aicore__ inline uint8_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return static_cast<uint8_t>((value + divisor - 1) / divisor);
}

}  // namespace mhcs

template <typename DT_LOGITS, uint32_t ORDER_SEL>
class KernelMhcSinkhorn {
    static constexpr bool FAST = (ORDER_SEL != 0);
    static constexpr bool PACKED = (ORDER_SEL == mhcs::MAX_ORDER);
    static constexpr bool IS_FP32 = AscendC::IsSameType<DT_LOGITS, float>::value;
    static constexpr uint32_t N = FAST ? ORDER_SEL : mhcs::MAX_ORDER;
    static constexpr uint64_t MASK_ALL = mhcs::FullMask(N * mhcs::BLK);
    static constexpr uint64_t MASK_COL = mhcs::LaneMask(N, N);
    static constexpr uint8_t NB = static_cast<uint8_t>(N);
    static constexpr uint8_t NB2 = static_cast<uint8_t>(2 * N);

public:
    __aicore__ inline KernelMhcSinkhorn() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR weights,
                                const MhcSinkhornTilingData &tiling)
    {
        order_ = FAST ? ORDER_SEL : tiling.order;
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
        t1_ = base[mhcs::OFF_T1];
        t2_ = base[mhcs::OFF_T2];
        dev_ = base[mhcs::OFF_DEV];
        rs_ = base[mhcs::OFF_RS];
        cs_ = base[mhcs::OFF_CS];
        mx_ = base[mhcs::OFF_MX];
        res_ = base[mhcs::OFF_RES];
        stage_ = base[mhcs::OFF_STG].template ReinterpretCast<DT_LOGITS>();
    }

    __aicore__ inline void Process()
    {
        if (count_ == 0) {
            return;
        }
        for (uint32_t done = 0; done < count_; done += mhcs::CHUNK) {
            uint32_t cnt = count_ - done;
            if (cnt > mhcs::CHUNK) {
                cnt = mhcs::CHUNK;
            }
            LoadChunk(done, cnt);
            if constexpr (FAST) {
                ComputeFast(cnt);
            } else {
                ComputeGeneric(cnt);
            }
            StoreChunk(done, cnt);
        }
    }

private:
    // ---------------------------------------------------------------- 数据搬入
    __aicore__ inline void LoadChunk(uint32_t done, uint32_t cnt)
    {
        const uint32_t n = order_;
        const uint64_t elems = static_cast<uint64_t>(n) * n;
        if (done != 0) {
            // 上一块仍在从同一片UB搬出, 清零与搬入都必须等它读完。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        // 只清零折叠指令的预读富余区: 行内padding由DataCopyPad按0填充, 二者互不重叠。
        AscendC::Duplicate(mat_[cnt * n * mhcs::BLK], 0.0f,
                           static_cast<int32_t>(mhcs::SLACK * mhcs::BLK));

        if constexpr (PACKED) {
            // n=8时每行恰好占满一个DataBlock, GM上整块连续, 一次搬运即可。
            const int32_t elemCount = static_cast<int32_t>(cnt * mhcs::MAX_ORDER * mhcs::BLK);
            if constexpr (IS_FP32) {
                AscendC::DataCopy(mat_, srcGm_[done * elems], elemCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            } else {
                AscendC::DataCopy(stage_, srcGm_[done * elems], elemCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::Cast(mat_, stage_, AscendC::RoundMode::CAST_NONE, elemCount);
            }
            return;
        }

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(cnt * n);
        copyParams.blockLen = n * sizeof(DT_LOGITS);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        AscendC::DataCopyPadExtParams<DT_LOGITS> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.paddingValue = static_cast<DT_LOGITS>(0.0f);

        if constexpr (IS_FP32) {
            padParams.rightPadding = static_cast<uint8_t>(mhcs::BLK - n);
            AscendC::DataCopyPad(mat_, srcGm_[done * elems], copyParams, padParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        } else {
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

        if constexpr (PACKED) {
            const int32_t elemCount = static_cast<int32_t>(cnt * mhcs::MAX_ORDER * mhcs::BLK);
            if constexpr (IS_FP32) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(dstGm_[done * elems], mat_, elemCount);
            } else {
                AscendC::Cast(stage_, mat_, AscendC::RoundMode::CAST_RINT, elemCount);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(dstGm_[done * elems], stage_, elemCount);
            }
            return;
        }

        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(cnt * n);
        copyParams.blockLen = n * sizeof(DT_LOGITS);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        if constexpr (IS_FP32) {
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
    }

    // ------------------------------------------------ 特化路径: 掩码常驻矢量寄存器
    //
    // 主循环中每条矢量指令都覆盖 n 个DataBlock, 掩码只在Brcb(内部复位掩码)之后重设;
    // 列方向折叠改用DataBlock步长成对相加, 从而与行方向共享同一掩码。
    __aicore__ inline void ComputeFast(uint32_t cnt)
    {
        // 单方阵是最常见的分核结果; 为它单独实例化一份, 所有迭代次数都成为立即数,
        // 指令不再需要通过标量寄存器下发迭代配置。
        if (cnt == 1) {
            ComputeFastImpl<true>(1);
        } else {
            ComputeFastImpl<false>(cnt);
        }
    }

    template <bool SINGLE>
    __aicore__ inline void ComputeFastImpl(uint32_t cnt)
    {
        const uint8_t rep = SINGLE ? uint8_t{1} : static_cast<uint8_t>(cnt);
        const uint8_t repVec = SINGLE ? uint8_t{1} : mhcs::CeilDiv(cnt, N);
        const uint32_t devBlocks = SINGLE ? N : cnt * N;
        const bool canCheck = SINGLE ? true : (devBlocks <= 64);
        const uint8_t devRep = SINGLE ? uint8_t{(N + mhcs::BLK - 1) / mhcs::BLK}
                                      : mhcs::CeilDiv(devBlocks, mhcs::BLK);

        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<float>(0, MASK_COL);
        AscendC::BlockReduceMax<float, false>(rs_, mat_, rep, mhcs::MASK_PH, 1, 1, NB);
        AscendC::Brcb(bct_, rs_, rep, {1, mhcs::BLK});
        AscendC::SetVectorMask<float>(0, MASK_COL);
        AscendC::Sub<float, false>(mat_, mat_, bct_, mhcs::MASK_PH, rep,
                                   {1, 1, 1, NB, NB, mhcs::BLK});
        AscendC::Exp<float, false>(mat_, mat_, mhcs::MASK_PH, rep, {1, 1, NB, NB});

        AscendC::SetVectorMask<float>(0, MASK_ALL);
        AscendC::BlockReduceSum<float, false>(rs_, mat_, rep, mhcs::MASK_PH, 1, 1, NB);
        AscendC::Brcb(bct_, rs_, rep, {1, mhcs::BLK});
        AscendC::SetVectorMask<float>(0, MASK_ALL);
        AscendC::Div<float, false>(mat_, mat_, bct_, mhcs::MASK_PH, rep,
                                   {1, 1, 1, NB, NB, mhcs::BLK});

        AscendC::SetVectorMask<float>(0, MASK_COL);
        AscendC::Adds<float, false>(mat_, mat_, eps_, mhcs::MASK_PH, rep, {1, 1, NB, NB});

        AscendC::SetVectorMask<float>(0, MASK_ALL);
        if (canCheck) {
            AscendC::Duplicate(dev_, 0.0f, static_cast<int32_t>(devRep * mhcs::BLK * mhcs::BLK));
            AscendC::SetVectorMask<float>(0, MASK_ALL);
        }
        ColumnStep<SINGLE>(cnt, rep, repVec);
        for (uint32_t i = 1; i < iters_; ++i) {
            AscendC::BlockReduceSum<float, false>(rs_, mat_, rep, mhcs::MASK_PH, 1, 1, NB);
            AscendC::Brcb(bct_, rs_, rep, {1, mhcs::BLK});
            AscendC::SetVectorMask<float>(0, MASK_ALL);
            AscendC::Adds<float, false>(bct_, bct_, eps_, mhcs::MASK_PH, rep,
                                        {1, 1, mhcs::BLK, mhcs::BLK});
            if (canCheck && Settled<SINGLE>(rep, devRep)) {
                break;
            }
            AscendC::Div<float, false>(mat_, mat_, bct_, mhcs::MASK_PH, rep,
                                       {1, 1, 1, NB, NB, mhcs::BLK});
            ColumnStep<SINGLE>(cnt, rep, repVec);
        }
        AscendC::ResetMask();
    }

    // 归一化因子是否已停在1附近: 此时本次及其后的迭代都不再改变结果。
    template <bool SINGLE>
    __aicore__ inline bool Settled(uint8_t rep, uint8_t devRep)
    {
        AscendC::Adds<float, false>(dev_, bct_, -1.0f, mhcs::MASK_PH, rep,
                                    {1, 1, NB, mhcs::BLK});
        AscendC::Abs<float, false>(dev_, dev_, mhcs::MASK_PH, rep, {1, 1, NB, NB});
        if constexpr (SINGLE) {
            // 单方阵时偏差恰好占满一个迭代(n*8 <= 64 lane), 一条归约指令即可取到最大值。
            AscendC::WholeReduceMax<float, false>(res_, dev_, mhcs::MASK_PH, 1, 1, 1, mhcs::BLK);
        } else {
            AscendC::SetVectorMask<float>(0, ~0ULL);
            AscendC::BlockReduceMax<float, false>(mx_, dev_, devRep, mhcs::MASK_PH, 1, 1,
                                                  mhcs::BLK);
            AscendC::SetVectorMask<float>(0, mhcs::FullMask(devRep * mhcs::BLK));
            AscendC::WholeReduceMax<float, false>(res_, mx_, mhcs::MASK_PH, 1, 1, 1, mhcs::BLK);
            AscendC::SetVectorMask<float>(0, MASK_ALL);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        return res_.GetValue(0) < mhcs::CONVERGE_TOL;
    }

    template <bool SINGLE>
    __aicore__ inline void ColumnStep(uint32_t cnt, uint8_t rep, uint8_t repVec)
    {
        // 成对折叠: 每条指令用DataBlock步长2把相邻两块相加, 掩码保持 n*8 lane。
        if constexpr (ORDER_SEL == 8) {
            AscendC::Add<float, false>(t1_, mat_, mat_[mhcs::BLK], mhcs::MASK_PH,
                                       (SINGLE ? uint8_t{1} : mhcs::CeilDiv(cnt, 2)), {1, 2, 2, NB, NB2, NB2});
            AscendC::Add<float, false>(t2_, t1_, t1_[mhcs::BLK], mhcs::MASK_PH,
                                       (SINGLE ? uint8_t{1} : mhcs::CeilDiv(cnt, 4)), {1, 2, 2, NB, NB2, NB2});
            AscendC::Add<float, false>(cs_, t2_, t2_[mhcs::BLK], mhcs::MASK_PH,
                                       (SINGLE ? uint8_t{1} : mhcs::CeilDiv(cnt, 8)), {1, 2, 2, NB, NB2, NB2});
        } else if constexpr (ORDER_SEL == 6) {
            AscendC::Add<float, false>(t1_, mat_, mat_[mhcs::BLK], mhcs::MASK_PH,
                                       (SINGLE ? uint8_t{1} : mhcs::CeilDiv(cnt, 2)), {1, 2, 2, NB, NB2, NB2});
            AscendC::Add<float, false>(t2_, t1_, t1_[mhcs::BLK], mhcs::MASK_PH,
                                       (SINGLE ? uint8_t{1} : mhcs::CeilDiv(cnt, 6)), {1, 3, 3, NB, 18, 18});
            AscendC::Add<float, false>(cs_, t2_, t1_[2 * mhcs::BLK], mhcs::MASK_PH,
                                       (SINGLE ? uint8_t{1} : mhcs::CeilDiv(cnt, 6)), {1, 1, 3, NB, NB, 18});
        } else {
            AscendC::Add<float, false>(t1_, mat_, mat_[mhcs::BLK], mhcs::MASK_PH,
                                       (SINGLE ? uint8_t{1} : mhcs::CeilDiv(cnt, 2)), {1, 2, 2, NB, NB2, NB2});
            AscendC::Add<float, false>(cs_, t1_, t1_[mhcs::BLK], mhcs::MASK_PH,
                                       (SINGLE ? uint8_t{1} : mhcs::CeilDiv(cnt, 4)), {1, 2, 2, NB, NB2, NB2});
        }
        AscendC::Adds<float, false>(cs_, cs_, eps_, mhcs::MASK_PH, repVec, {1, 1, NB, NB});
        AscendC::Div<float, false>(mat_, mat_, cs_, mhcs::MASK_PH, rep,
                                   {1, 1, 0, NB, NB, 1});
    }

    // ------------------------------------------------------------ 通用兜底路径
    __aicore__ inline void ComputeGeneric(uint32_t cnt)
    {
        const uint32_t n = order_;
        const uint8_t rep = static_cast<uint8_t>(cnt);
        const int32_t span = static_cast<int32_t>(n * mhcs::BLK);
        const uint8_t stride = static_cast<uint8_t>(n);
        uint64_t colMask[2] = {mhcs::LaneMask(n, n), 0};

        AscendC::BlockReduceMax<float>(rs_, mat_, rep, colMask, 1, 1, stride);
        AscendC::Brcb(bct_, rs_, rep, {1, mhcs::BLK});
        AscendC::Sub(mat_, mat_, bct_, colMask, rep, {1, 1, 1, stride, stride, mhcs::BLK});
        AscendC::Exp(mat_, mat_, colMask, rep, {1, 1, stride, stride});
        AscendC::BlockReduceSum<float>(rs_, mat_, rep, span, 1, 1, stride);
        AscendC::Brcb(bct_, rs_, rep, {1, mhcs::BLK});
        AscendC::Div(mat_, mat_, bct_, span, rep, {1, 1, 1, stride, stride, mhcs::BLK});
        AscendC::Adds(mat_, mat_, eps_, colMask, rep, {1, 1, stride, stride});

        GenericColumnStep(cnt, rep, span, stride);
        for (uint32_t i = 1; i < iters_; ++i) {
            AscendC::BlockReduceSum<float>(rs_, mat_, rep, span, 1, 1, stride);
            AscendC::Brcb(bct_, rs_, rep, {1, mhcs::BLK});
            AscendC::Adds(bct_, bct_, eps_, span, rep, {1, 1, mhcs::BLK, mhcs::BLK});
            AscendC::Div(mat_, mat_, bct_, span, rep, {1, 1, 1, stride, stride, mhcs::BLK});
            GenericColumnStep(cnt, rep, span, stride);
        }
    }

    __aicore__ inline void GenericColumnStep(uint32_t cnt, uint8_t rep, int32_t span,
                                             uint8_t stride)
    {
        const uint32_t n = order_;
        AscendC::Add(cs_, mat_, mat_[mhcs::BLK], 8, rep, {1, 1, 1, 1, stride, stride});
        for (uint32_t r = 2; r < n; ++r) {
            AscendC::Add(cs_, cs_, mat_[r * mhcs::BLK], 8, rep, {1, 1, 1, 1, 1, stride});
        }
        AscendC::Adds(cs_, cs_, eps_, static_cast<int32_t>(cnt * mhcs::BLK));
        AscendC::Div(mat_, mat_, cs_, span, rep, {1, 1, 0, stride, stride, 1});
    }

    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_;
    AscendC::GlobalTensor<DT_LOGITS> srcGm_;
    AscendC::GlobalTensor<DT_LOGITS> dstGm_;
    AscendC::LocalTensor<float> mat_;
    AscendC::LocalTensor<float> bct_;
    AscendC::LocalTensor<float> t1_;
    AscendC::LocalTensor<float> t2_;
    AscendC::LocalTensor<float> dev_;
    AscendC::LocalTensor<float> rs_;
    AscendC::LocalTensor<float> cs_;
    AscendC::LocalTensor<float> mx_;
    AscendC::LocalTensor<float> res_;
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
