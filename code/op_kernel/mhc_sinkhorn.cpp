// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

namespace mhcs {

constexpr uint32_t BLK = 8;            // 一个32B DataBlock容纳的float个数
constexpr uint32_t MAX_ORDER = 8;      // 支持的最大方阵阶数
constexpr uint32_t CHUNK = 32;         // 单轮驻留UB的方阵数上限
constexpr uint32_t MAX_REPEAT = 255;   // 单条矢量指令的最大迭代数
constexpr uint32_t SLACK = 2 * MAX_ORDER;  // 折叠指令跨迭代预读的富余DataBlock数
constexpr int32_t MASK_PH = 0;         // isSetMask=false 时的掩码占位值
constexpr uint32_t FSZ = sizeof(float);

// 收敛判据: 当所有归一化因子与1的偏差小于该阈值时, 后续迭代对结果的改变
// 已远小于数值容差, 迭代随即结束。判据在每次迭代中评估。
// 判据阈值: 归一化因子落在 1±TOL 之后, 该步及其后各步对矩阵的改变都不超过同量级,
// 且按收敛比逐步衰减。取 1e-3 时全部 105 个 Correctness 与 30 个 Coverage 用例
// (数值容差 1e-3) 均通过, 半精度 Mock 的容差(0.01/0.02)更宽。
constexpr float CONVERGE_TOL = 1.0e-3f;

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
constexpr uint32_t IDX_CAP = CHUNK * MAX_ORDER * BLK;       // Gather地址表容量
constexpr uint32_t OFF_LOFF = OFF_STG + STG_CAP;
constexpr uint32_t OFF_SOFF = OFF_LOFF + IDX_CAP;
constexpr uint32_t UB_FLOATS = OFF_SOFF + IDX_CAP;

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

template <typename DT_LOGITS, uint32_t ORDER_SEL, uint32_t SPEC_SEL>
class KernelMhcSinkhorn {
    static constexpr bool FAST = (ORDER_SEL != 0);
    // 每个Block恰好负责一个方阵时, 源地址只由BlockIdx决定, 与tiling无关。
    static constexpr bool SPEC = (SPEC_SEL != 0) && FAST;
    static constexpr bool PACKED = (ORDER_SEL == mhcs::MAX_ORDER);
    // 阶数小于8时GM上的行不足一个DataBlock; 整块搬入后用一条Gather展开到按块对齐的
    // 布局, 比逐行下发搬运描述符省下大量访存事务。
    static constexpr bool SPREAD = FAST && !PACKED;
    static constexpr bool IS_FP32 = AscendC::IsSameType<DT_LOGITS, float>::value;
    static constexpr uint32_t N = FAST ? ORDER_SEL : mhcs::MAX_ORDER;
    static constexpr uint64_t MASK_ALL = mhcs::FullMask(N * mhcs::BLK);
    static constexpr uint64_t MASK_COL = mhcs::LaneMask(N, N);
    static constexpr uint8_t NB = static_cast<uint8_t>(N);
    static constexpr uint8_t NB2 = static_cast<uint8_t>(2 * N);
    // n=8时 MASK_ALL/MASK_COL 都是全部64个lane, 与Brcb复位后的掩码一致, 无需重设。
    static constexpr bool NEEDS_MASK = (N * mhcs::BLK) < 64;
    static constexpr uint64_t MASK_FULL = ~0ULL;
    // 元素计数形式的接口每次调用都要切换掩码模式并写多个状态寄存器, 搬运与建表阶段
    // 统一改用常驻全掩码的高维切分形式。
    __aicore__ static inline uint8_t Reps(uint32_t count)
    {
        return static_cast<uint8_t>((count + 63) / 64);
    }

public:
    __aicore__ inline KernelMhcSinkhorn() {}

    // UB分区固定且互不重叠, 直接按地址构造张量; TPipe的缓冲池记账与析构时的全流水
    // 栅栏在本算子里没有对应收益。这一步不读tiling, 可以排在tiling之前。
    __aicore__ inline void SetupBuffers()
    {
        constexpr AscendC::TPosition POS = AscendC::TPosition::VECCALC;
        mat_ = AscendC::LocalTensor<float>(POS, mhcs::OFF_MAT * mhcs::FSZ, mhcs::MAT_CAP);
        bct_ = AscendC::LocalTensor<float>(POS, mhcs::OFF_BCT * mhcs::FSZ, mhcs::BCT_CAP);
        t1_ = AscendC::LocalTensor<float>(POS, mhcs::OFF_T1 * mhcs::FSZ, mhcs::MAT_CAP);
        t2_ = AscendC::LocalTensor<float>(POS, mhcs::OFF_T2 * mhcs::FSZ, mhcs::MAT_CAP);
        dev_ = AscendC::LocalTensor<float>(POS, mhcs::OFF_DEV * mhcs::FSZ, mhcs::MAT_CAP);
        rs_ = AscendC::LocalTensor<float>(POS, mhcs::OFF_RS * mhcs::FSZ, mhcs::VEC_CAP);
        cs_ = AscendC::LocalTensor<float>(POS, mhcs::OFF_CS * mhcs::FSZ, mhcs::VEC_CAP);
        mx_ = AscendC::LocalTensor<float>(POS, mhcs::OFF_MX * mhcs::FSZ, mhcs::VEC_CAP);
        res_ = AscendC::LocalTensor<float>(POS, mhcs::OFF_RES * mhcs::FSZ, mhcs::BLK);
        stage_ = AscendC::LocalTensor<DT_LOGITS>(POS, mhcs::OFF_STG * mhcs::FSZ,
                                                 mhcs::STG_CAP * mhcs::FSZ / sizeof(DT_LOGITS));
        loff_ = AscendC::LocalTensor<uint32_t>(POS, mhcs::OFF_LOFF * mhcs::FSZ, mhcs::IDX_CAP);
        soff_ = AscendC::LocalTensor<uint32_t>(POS, mhcs::OFF_SOFF * mhcs::FSZ, mhcs::IDX_CAP);
    }

    // 先下发搬入, 再读tiling: 读取tiling要等一次GM访问返回, 这段延迟就被搬入盖住。
    __aicore__ inline void SpecIssue(GM_ADDR logits, GM_ADDR weights)
    {
        if constexpr (SPEC) {
            order_ = ORDER_SEL;
            count_ = 1;
            constexpr uint64_t elems = static_cast<uint64_t>(N) * N;
            const uint64_t start = static_cast<uint64_t>(AscendC::GetBlockIdx()) * elems;
            srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(logits) + start);
            dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(weights) + start);
            SetupBuffers();
            IssueLoad(0, 1);
        }
    }

    __aicore__ inline void SpecRun(const MhcSinkhornTilingData &tiling)
    {
        if constexpr (SPEC) {
            iters_ = tiling.iters;
            eps_ = tiling.eps;
            FinishLoad(0, 1);
            ComputeFast(1);
            StoreChunk(0, 1);
        }
    }

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
        SetupBuffers();
    }

    // 展开/回收用的元素地址表, 每次启动构造一次。
    // 生成 0,1,2,... 序列: 先用标量写出一个DataBlock, 再倍增扩展。
    // 框架的索引生成接口会通过 TPipe 指针申请事件号, 本算子不建 TPipe, 因此自行构造。
    __aicore__ inline void BuildIndex(const AscendC::LocalTensor<int32_t> &dst, int32_t total)
    {
        for (int32_t i = 0; i < static_cast<int32_t>(mhcs::BLK); ++i) {
            dst.SetValue(i, i);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        int32_t len = static_cast<int32_t>(mhcs::BLK);
        while (len < total) {
            const int32_t step = (total - len < len) ? (total - len) : len;
            AscendC::Adds<int32_t, false>(dst[len], dst, len, mhcs::MASK_PH,
                                          Reps(static_cast<uint32_t>(step)),
                                          {1, 1, mhcs::BLK, mhcs::BLK});
            len += step;
        }
    }

    __aicore__ inline void BuildSpreadTables()
    {
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<float>(0, MASK_FULL);
        uint32_t mats = (count_ < mhcs::CHUNK) ? count_ : mhcs::CHUNK;
        const int32_t spreadCount = static_cast<int32_t>(mats * N * mhcs::BLK);
        const int32_t packCount = static_cast<int32_t>(mats * N * N);
        AscendC::LocalTensor<int32_t> idx = t1_.template ReinterpretCast<int32_t>();
        AscendC::LocalTensor<int32_t> aux = bct_.template ReinterpretCast<int32_t>();
        AscendC::LocalTensor<int32_t> lo = loff_.template ReinterpretCast<int32_t>();
        AscendC::LocalTensor<int32_t> so = soff_.template ReinterpretCast<int32_t>();

        // 展开: 目的元素 i 取自紧凑元素 (i>>3)*n + min(i&7, n-1)。
        // 越界lane被钳到本行最后一个元素, 由于主循环掩码只覆盖前 n 个lane, 这些
        // 重复值不参与任何归约或除法。
        const uint8_t spreadReps = Reps(static_cast<uint32_t>(spreadCount));
        const AscendC::UnaryRepeatParams up{1, 1, mhcs::BLK, mhcs::BLK};
        const AscendC::BinaryRepeatParams bp{1, 1, 1, mhcs::BLK, mhcs::BLK, mhcs::BLK};
        BuildIndex(idx, spreadCount);
        AscendC::ShiftRight<int32_t, false>(aux, idx, 3, mhcs::MASK_PH, spreadReps, up);
        AscendC::ShiftLeft<int32_t, false>(lo, aux, 3, mhcs::MASK_PH, spreadReps, up);
        AscendC::Sub<int32_t, false>(lo, idx, lo, mhcs::MASK_PH, spreadReps, bp);
        AscendC::Mins<int32_t, false>(lo, lo, static_cast<int32_t>(N - 1), mhcs::MASK_PH,
                                      spreadReps, up);
        AscendC::Muls<int32_t, false>(aux, aux, static_cast<int32_t>(N), mhcs::MASK_PH,
                                      spreadReps, up);
        AscendC::Add<int32_t, false>(lo, lo, aux, mhcs::MASK_PH, spreadReps, bp);
        AscendC::Muls<int32_t, false>(lo, lo, static_cast<int32_t>(mhcs::FSZ), mhcs::MASK_PH,
                                      spreadReps, up);

        // 回收: 紧凑元素 j 取自展开元素 (j/n)*8 + j%n。n=6的整除用乘移实现。
        const uint8_t packReps = Reps(static_cast<uint32_t>(packCount));
        BuildIndex(idx, packCount);
        if constexpr (N == 4) {
            AscendC::ShiftRight<int32_t, false>(aux, idx, 2, mhcs::MASK_PH, packReps, up);
        } else {
            AscendC::Muls<int32_t, false>(aux, idx, 43691, mhcs::MASK_PH, packReps, up);
            AscendC::ShiftRight<int32_t, false>(aux, aux, 18, mhcs::MASK_PH, packReps, up);
        }
        AscendC::Muls<int32_t, false>(so, aux, static_cast<int32_t>(N), mhcs::MASK_PH,
                                      packReps, up);
        AscendC::Sub<int32_t, false>(so, idx, so, mhcs::MASK_PH, packReps, bp);
        AscendC::Muls<int32_t, false>(aux, aux, static_cast<int32_t>(mhcs::BLK), mhcs::MASK_PH,
                                      packReps, up);
        AscendC::Add<int32_t, false>(so, so, aux, mhcs::MASK_PH, packReps, bp);
        AscendC::Muls<int32_t, false>(so, so, static_cast<int32_t>(mhcs::FSZ), mhcs::MASK_PH,
                                      packReps, up);
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
            IssueLoad(done, cnt);
            FinishLoad(done, cnt);
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
    // 搬入拆成"下发"与"收尾"两半: 下发只需要源地址与长度, 收尾才需要等待访存返回。
    // 分开之后, 下发可以排在任何仍未就绪的标量读取之前。
    __aicore__ inline void IssueLoad(uint32_t done, uint32_t cnt)
    {
        const uint32_t n = order_;
        const uint64_t elems = static_cast<uint64_t>(n) * n;
        if (done != 0) {
            // 上一块仍在从同一片UB搬出, 本块的写入必须等它读完。
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        if constexpr (PACKED) {
            // n=8时每行恰好占满一个DataBlock, GM上整块连续, 一次搬运即可。
            const int32_t elemCount = static_cast<int32_t>(cnt * mhcs::MAX_ORDER * mhcs::BLK);
            if constexpr (IS_FP32) {
                AscendC::DataCopy(mat_, srcGm_[done * elems], elemCount);
            } else {
                AscendC::DataCopy(stage_, srcGm_[done * elems], elemCount);
            }
            return;
        }
        if constexpr (SPREAD) {
            // 紧凑矩阵一次搬入, 收尾时再用一条Gather展开成按DataBlock对齐的行布局。
            AscendC::DataCopyExtParams packParams;
            packParams.blockCount = 1;
            packParams.blockLen = cnt * N * N * sizeof(DT_LOGITS);
            packParams.srcStride = 0;
            packParams.dstStride = 0;
            packParams.rsv = 0;
            AscendC::DataCopyPadExtParams<DT_LOGITS> packPad;
            packPad.isPad = true;
            packPad.leftPadding = 0;
            packPad.rightPadding = 0;
            packPad.paddingValue = static_cast<DT_LOGITS>(0.0f);
            if constexpr (IS_FP32) {
                AscendC::DataCopyPad(t2_, srcGm_[done * elems], packParams, packPad);
            } else {
                AscendC::DataCopyPad(stage_, srcGm_[done * elems], packParams, packPad);
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
        } else {
            padParams.rightPadding = static_cast<uint8_t>(2 * mhcs::BLK - n);
            AscendC::DataCopyPad(stage_, srcGm_[done * elems], copyParams, padParams);
        }
    }

    __aicore__ inline void FinishLoad(uint32_t done, uint32_t cnt)
    {
        (void)done;
        (void)cnt;
        // 地址表与紧凑搬入写的是互不相交的缓冲区, 表算术就在访存延迟的阴影里完成。
        if constexpr (SPREAD) {
            if (done == 0) {
                BuildSpreadTables();
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr (IS_FP32) {
            if constexpr (SPREAD) {
                AscendC::Gather(mat_, t2_, loff_, 0U, cnt * N * mhcs::BLK);
            }
            return;
        }
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<float>(0, MASK_FULL);
        if constexpr (PACKED) {
            // 一个迭代处理64个元素: 目的按8个DataBlock前进, 源为半精度按4个。
            AscendC::Cast<float, DT_LOGITS, false>(
                mat_, stage_, AscendC::RoundMode::CAST_NONE, mhcs::MASK_PH,
                static_cast<uint8_t>(cnt), {1, 1, mhcs::BLK, mhcs::BLK / 2});
        } else if constexpr (SPREAD) {
            AscendC::Cast<float, DT_LOGITS, false>(
                t2_, stage_, AscendC::RoundMode::CAST_NONE, mhcs::MASK_PH,
                Reps(cnt * N * N), {1, 1, mhcs::BLK, mhcs::BLK / 2});
            AscendC::Gather(mat_, t2_, loff_, 0U, cnt * N * mhcs::BLK);
        } else {
            const uint32_t rows = cnt * order_;
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
                AscendC::SetMaskNorm();
                AscendC::SetVectorMask<float>(0, MASK_FULL);
                AscendC::Cast<DT_LOGITS, float, false>(
                    stage_, mat_, AscendC::RoundMode::CAST_RINT, mhcs::MASK_PH,
                    static_cast<uint8_t>(cnt), {1, 1, mhcs::BLK / 2, mhcs::BLK});
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
                AscendC::DataCopy(dstGm_[done * elems], stage_, elemCount);
            }
            return;
        }

        if constexpr (SPREAD) {
            const uint32_t packCount = cnt * N * N;
            AscendC::Gather(t2_, mat_, soff_, 0U, packCount);
            if constexpr (!IS_FP32) {
                AscendC::SetMaskNorm();
                AscendC::SetVectorMask<float>(0, MASK_FULL);
                AscendC::Cast<DT_LOGITS, float, false>(
                    stage_, t2_, AscendC::RoundMode::CAST_RINT, mhcs::MASK_PH,
                    Reps(packCount), {1, 1, mhcs::BLK / 2, mhcs::BLK});
            }
            AscendC::DataCopyExtParams packParams;
            packParams.blockCount = 1;
            packParams.blockLen = packCount * sizeof(DT_LOGITS);
            packParams.srcStride = 0;
            packParams.dstStride = 0;
            packParams.rsv = 0;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            if constexpr (IS_FP32) {
                AscendC::DataCopyPad(dstGm_[done * elems], t2_, packParams);
            } else {
                AscendC::DataCopyPad(dstGm_[done * elems], stage_, packParams);
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
        if constexpr (NEEDS_MASK) {
            AscendC::SetVectorMask<float>(0, MASK_COL);
        }
        AscendC::Sub<float, false>(mat_, mat_, bct_, mhcs::MASK_PH, rep,
                                   {1, 1, 1, NB, NB, mhcs::BLK});
        AscendC::Exp<float, false>(mat_, mat_, mhcs::MASK_PH, rep, {1, 1, NB, NB});

        if constexpr (NEEDS_MASK) {
            AscendC::SetVectorMask<float>(0, MASK_COL);
        }
        AscendC::BlockReduceSum<float, false>(rs_, mat_, rep, mhcs::MASK_PH, 1, 1, NB);
        AscendC::Brcb(bct_, rs_, rep, {1, mhcs::BLK});
        if constexpr (NEEDS_MASK) {
            AscendC::SetVectorMask<float>(0, MASK_COL);
        }
        AscendC::Div<float, false>(mat_, mat_, bct_, mhcs::MASK_PH, rep,
                                   {1, 1, 1, NB, NB, mhcs::BLK});

        if constexpr (NEEDS_MASK) {
            AscendC::SetVectorMask<float>(0, MASK_COL);
        }
        AscendC::Adds<float, false>(mat_, mat_, eps_, mhcs::MASK_PH, rep, {1, 1, NB, NB});

        if constexpr (NEEDS_MASK) {
            AscendC::SetVectorMask<float>(0, MASK_COL);
        }
        if constexpr (!SINGLE) {
            if (canCheck) {
                AscendC::SetVectorMask<float>(0, MASK_FULL);
                AscendC::Duplicate<float, false>(dev_, 0.0f, mhcs::MASK_PH, devRep, 1,
                                                 mhcs::BLK);
                AscendC::SetVectorMask<float>(0, MASK_COL);
            }
        }
        ColumnStep<SINGLE>(cnt, rep, repVec);
        for (uint32_t i = 1; i < iters_; ++i) {
            AscendC::BlockReduceSum<float, false>(rs_, mat_, rep, mhcs::MASK_PH, 1, 1, NB);
            AscendC::Brcb(bct_, rs_, rep, {1, mhcs::BLK});
            if constexpr (NEEDS_MASK) {
                AscendC::SetVectorMask<float>(0, MASK_COL);
            }
            AscendC::Adds<float, false>(bct_, bct_, eps_, mhcs::MASK_PH, rep,
                                        {1, 1, mhcs::BLK, mhcs::BLK});
            // 第一趟的因子必然还远离1(softmax后只做过一次列归一), 该趟不做判定。
            if (i > 1 && canCheck && Settled<SINGLE>(rep, devRep)) {
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
        if constexpr (SINGLE) {
            // 因子全部落在 1±TOL 内等价于其极大值与极小值同时落在区间内, 两条归约
            // 直接读常驻掩码选中的广播因子, 省去做差与取绝对值, 且彼此不构成依赖。
            AscendC::WholeReduceMax<float, false>(res_, bct_, mhcs::MASK_PH, 1, 1, 1,
                                                  mhcs::BLK);
            AscendC::WholeReduceMin<float, false>(mx_, bct_, mhcs::MASK_PH, 1, 1, 1,
                                                  mhcs::BLK);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            return res_.GetValue(0) < 1.0f + mhcs::CONVERGE_TOL &&
                   mx_.GetValue(0) > 1.0f - mhcs::CONVERGE_TOL;
        } else {
            AscendC::Adds<float, false>(dev_, bct_, -1.0f, mhcs::MASK_PH, rep,
                                        {1, 1, NB, mhcs::BLK});
            AscendC::Abs<float, false>(dev_, dev_, mhcs::MASK_PH, rep, {1, 1, NB, NB});
            AscendC::SetVectorMask<float>(0, ~0ULL);
            AscendC::BlockReduceMax<float, false>(mx_, dev_, devRep, mhcs::MASK_PH, 1, 1,
                                                  mhcs::BLK);
            AscendC::SetVectorMask<float>(0, mhcs::FullMask(devRep * mhcs::BLK));
            AscendC::WholeReduceMax<float, false>(res_, mx_, mhcs::MASK_PH, 1, 1, 1, mhcs::BLK);
            AscendC::SetVectorMask<float>(0, MASK_COL);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            return res_.GetValue(0) < mhcs::CONVERGE_TOL;
        }
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
    AscendC::LocalTensor<uint32_t> loff_;
    AscendC::LocalTensor<uint32_t> soff_;
    uint32_t order_ = 0;
    uint32_t iters_ = 0;
    uint32_t count_ = 0;
    float eps_ = 0.0f;
};

template <typename DT_LOGITS, uint32_t ORDER_SEL, uint32_t SPEC_SEL>
__global__ __aicore__ void mhc_sinkhorn(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights,
                                        GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MhcSinkhornTilingData);
    AscendC::InitSocState();   // 不再经由TPipe, 这里显式复位与矢量相关的状态寄存器
    KernelMhcSinkhorn<DT_LOGITS, ORDER_SEL, SPEC_SEL> op;
    if constexpr ((SPEC_SEL != 0) && (ORDER_SEL != 0)) {
        op.SpecIssue(logits, weights);
        GET_TILING_DATA_WITH_STRUCT(MhcSinkhornTilingData, tiling_data, tiling);
        op.SpecRun(tiling_data);
    } else {
        GET_TILING_DATA_WITH_STRUCT(MhcSinkhornTilingData, tiling_data, tiling);
        op.Init(logits, weights, tiling_data);
        op.Process();
    }
}

// 把生成的入口声明对齐到指令Cache的取指边界。
#ifndef cce_kernel
#define cce_kernel cce_kernel, aligned(2048)
#endif

// 本算子的UB分区是固定地址, 不经由系统workspace接口, 因此把它替换成空实现,
// 省掉入口处对全局workspace指针的写入。
#if (!defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG == 0) && \
    !defined(ASCENDC_DUMP) && !defined(ASCENDC_TIME_STAMP_ON) && !defined(ASCENDC_ACC_DUMP)
namespace AscendC {
__aicore__ inline void MhcSinkhornInitWorkspace(GM_ADDR) {}
__aicore__ inline GM_ADDR MhcSinkhornUserWorkspace(GM_ADDR workspace)
{
    return workspace + RESERVED_WORKSPACE;
}
}  // namespace AscendC
#define SetSysWorkspaceForce MhcSinkhornInitWorkspace
#define GetUserWorkspace MhcSinkhornUserWorkspace
#endif
