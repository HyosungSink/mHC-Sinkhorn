// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

using namespace AscendC;

namespace {
constexpr uint32_t kLanes = 64;       // 8 rows x 8 lanes of fp32
constexpr uint32_t kRowLanes = 8;     // one 32B data block per padded row
constexpr float kSettledTail = 3.0e-4F;  // movement still allowed to be left in the loop
constexpr float kMaxRho = 0.999F;     // largest contraction factor treated as geometric
constexpr float kMinContractEps = 1.0e-20F;  // below this the normalization may lose contraction
constexpr float kNegHuge = -3.0e38F;  // keeps padding lanes out of the row maximum

// UB arena layout in bytes. Every matrix lives in a padded 8x8 fp32 tile whose
// rows are separated by one 32B data block, so block level reductions act on
// rows and block sized address offsets act on columns.
constexpr uint32_t kOffMat = 0;       // padded matrix (512B: 256B payload + tree slack)
constexpr uint32_t kOffTmpA = 512;    // column tree level 1 (512B)
constexpr uint32_t kOffTmpB = 1024;   // column tree level 2 (512B)
constexpr uint32_t kOffSum = 1536;    // reduction result (256B)
constexpr uint32_t kOffDiv = 1664;    // broadcast divisor (256B)
constexpr uint32_t kOffCons = 1920;   // divisor constant, eps / 1.0 (256B)
constexpr uint32_t kOffEps = 2176;    // additive eps vector (256B)
constexpr uint32_t kOffIndex = 2432;  // gather byte offsets, uint32 x 64 (256B)
constexpr uint32_t kOffStage = 2688;  // input staging (256B)
constexpr uint32_t kOffOut = 2944;    // output staging (256B)
constexpr uint32_t kOffCol = 3200;    // column divisor slot (256B)
constexpr uint32_t kOffPrev = 3456;   // previous-iteration tile snapshot (256B)
constexpr uint32_t kArenaBytes = 3712;

template <AscendC::HardEvent event>
__aicore__ inline void Sync() {
    AscendC::SetFlag<event>(EVENT_ID0);
    AscendC::WaitFlag<event>(EVENT_ID0);
}
}  // namespace

template <typename DT_X>
class KernelMhcSinkhorn {
public:
    __aicore__ inline KernelMhcSinkhorn() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const MhcSinkhornTilingData &data) {
        tiling_ = data;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        pipe_.InitBuffer(arena_, kArenaBytes);
        const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t begin = blockIdx * tiling_.matricesPerCore;
        const uint32_t end = begin + tiling_.matricesPerCore;
        first_ = begin < tiling_.matrices ? begin : tiling_.matrices;
        const uint32_t limit = end < tiling_.matrices ? end : tiling_.matrices;
        count_ = limit > first_ ? limit - first_ : 0U;
        n_ = tiling_.n;
        iterations_ = tiling_.iterations;
    }

    __aicore__ inline void Process() {
        if (count_ == 0U) {
            return;
        }
        BuildConstants();
        // One continuous mask for the whole run: every hot-path vector op takes
        // this mask instead of programming its own.
        SetVectorMask<float>(static_cast<int32_t>(kLanes));
        for (uint32_t index = 0; index < count_; ++index) {
            ProcessMatrix(first_ + index);
        }
    }

private:
    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t byteOffset) {
        return arena_.Get<U>()[byteOffset / sizeof(U)];
    }

    // Lanes below n carry eps, the remaining lanes carry 1.0 so that the divide
    // keeps padding lanes at zero even when eps is zero. epsVec is the same
    // pattern with padding lanes cleared, used for the initial softmax shift.
    __aicore__ inline void BuildConstants() {
        LocalTensor<float> cons = At<float>(kOffCons);
        LocalTensor<float> epsVec = At<float>(kOffEps);
        LocalTensor<uint32_t> index = At<uint32_t>(kOffIndex);
        LocalTensor<float> epsSeed = At<float>(kOffCons + 128);  // scratch low block
        const float eps = tiling_.eps;
        // cons carries the eight lane pattern eps..eps,1..1; the gather offsets
        // are byte strides 0,4,...,28 repeated per block. epsVec is that same
        // eight lane pattern replicated onto every block, which is exactly the
        // column broadcast the gather index describes.
        for (uint32_t lane = 0; lane < kRowLanes; ++lane) {
            const bool inside = lane < n_;
            cons.SetValue(lane, inside ? eps : 1.0F);
            epsSeed.SetValue(lane, inside ? eps : 0.0F);
            index.SetValue(lane, lane * static_cast<uint32_t>(sizeof(float)));
        }
        for (uint32_t lane = kRowLanes; lane < kLanes; ++lane) {
            index.SetValue(lane, (lane % kRowLanes) * static_cast<uint32_t>(sizeof(float)));
        }
        Sync<HardEvent::S_V>();
        Gather(epsVec, epsSeed, index, 0, static_cast<uint32_t>(kLanes));
        SetVectorMask<float>(static_cast<int32_t>(kLanes));
    }




    __aicore__ inline BinaryRepeatParams BinParams() const {
        return BinaryRepeatParams(1, 1, 1, 8, 8, 8);
    }

    __aicore__ inline void LoadMatrix(uint32_t matrix) {
        LocalTensor<float> mat = At<float>(kOffMat);
        const uint64_t offset = static_cast<uint64_t>(matrix) * n_ * n_;
        // The previous store reads the tile and the previous compute reads the
        // staging buffers, so both must retire before this load overwrites them.
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::V_MTE2>();
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            DataCopyExtParams copy{static_cast<uint16_t>(n_),
                                   static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(kRowLanes - n_), kNegHuge};
            DataCopyPad(mat, xGm_[offset], copy, pad);
            Sync<HardEvent::MTE2_V>();
        } else {
            LocalTensor<DT_X> stage = At<DT_X>(kOffStage);
            DataCopyExtParams copy{static_cast<uint16_t>(n_),
                                   static_cast<uint32_t>(n_ * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> pad{true, 0, static_cast<uint8_t>(kRowLanes - n_),
                                           static_cast<DT_X>(0)};
            DataCopyPad(stage, xGm_[offset], copy, pad);
            Duplicate(mat, kNegHuge, static_cast<int32_t>(kLanes));
            Sync<HardEvent::MTE2_V>();
            for (uint32_t row = 0; row < n_; ++row) {
                Cast(mat[row * kRowLanes], stage[row * kRowLanes * 2], RoundMode::CAST_NONE,
                     static_cast<int32_t>(n_));
                    }
            Sync<HardEvent::V_MTE2>();
        }
    }

    __aicore__ inline void StoreMatrix(uint32_t matrix) {
        LocalTensor<float> mat = At<float>(kOffMat);
        const uint64_t offset = static_cast<uint64_t>(matrix) * n_ * n_;
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            DataCopyExtParams copy{static_cast<uint16_t>(n_),
                                   static_cast<uint32_t>(n_ * sizeof(float)), 0, 0, 0};
            Sync<HardEvent::V_MTE3>();
            DataCopyPad(yGm_[offset], mat, copy);
        } else {
            LocalTensor<DT_X> stage = At<DT_X>(kOffOut);
            Sync<HardEvent::MTE3_V>();
            for (uint32_t row = 0; row < n_; ++row) {
                Cast(stage[row * kRowLanes * 2], mat[row * kRowLanes], RoundMode::CAST_NONE,
                     static_cast<int32_t>(n_));
                    }
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams copy{static_cast<uint16_t>(n_),
                                   static_cast<uint32_t>(n_ * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[offset], stage, copy);
        }
    }

    __aicore__ inline void NormalizeColumns(const LocalTensor<float> &mat,
                                            const LocalTensor<float> &tmpA,
                                            const LocalTensor<float> &tmpB,
                                            const LocalTensor<float> &sum,
                                            const LocalTensor<float> &div,
                                            const LocalTensor<float> &cons,
                                            const LocalTensor<float> &col) {
        LocalTensor<uint32_t> index = At<uint32_t>(kOffIndex);
        Add<float, false>(tmpA, mat, mat[kRowLanes], MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        Add<float, false>(tmpB, tmpA, tmpA[2 * kRowLanes], MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        Add<float, false>(col, tmpB, tmpB[4 * kRowLanes], MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        Add<float, false>(col, col, cons, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        Gather(div, col, index, 0, static_cast<uint64_t>(kLanes), 1, static_cast<uint16_t>(kRowLanes));
        Div<float, false>(mat, mat, div, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
    }

    // How far the tile moved during the finished iteration, taken over every
    // lane of the padded tile so no row can hide behind another one.
    __aicore__ inline float TileMove(const LocalTensor<float> &mat,
                                     const LocalTensor<float> &prev,
                                     const LocalTensor<float> &scratch,
                                     const LocalTensor<float> &chkMax) {
        Sub(scratch, mat, prev, static_cast<int32_t>(kLanes));
        Abs(scratch, scratch, static_cast<int32_t>(kLanes));
        WholeReduceMax<float, false>(chkMax, scratch, MASK_PLACEHOLDER, 1, 1, 1,
                                     static_cast<int32_t>(kRowLanes));
        Sync<HardEvent::V_S>();
        const float move = chkMax.GetValue(0);
        SetVectorMask<float>(static_cast<int32_t>(kLanes));
        return move;
    }

    __aicore__ inline void RowStep(const LocalTensor<float> &mat,
                                   const LocalTensor<float> &sum,
                                   const LocalTensor<float> &div,
                                   const LocalTensor<float> &cons) {
        BlockReduceSum<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1,
                                     static_cast<int32_t>(kRowLanes));
        Add<float, false>(sum, sum, cons, MASK_PLACEHOLDER, 1, BinParams());
        Brcb(div, sum, 1, BrcbRepeatParams(1, static_cast<uint16_t>(kRowLanes)));
        Div<float, false>(mat, mat, div, MASK_PLACEHOLDER, 1, BinParams());
    }

    __aicore__ inline void ProcessMatrix(uint32_t matrix) {
        LocalTensor<float> mat = At<float>(kOffMat);
        LocalTensor<float> tmpA = At<float>(kOffTmpA);
        LocalTensor<float> tmpB = At<float>(kOffTmpB);
        LocalTensor<float> sum = At<float>(kOffSum);
        LocalTensor<float> div = At<float>(kOffDiv);
        LocalTensor<float> cons = At<float>(kOffCons);
        LocalTensor<float> col = At<float>(kOffCol);
        LocalTensor<float> prev = At<float>(kOffPrev);
        LocalTensor<float> chkMax = At<float>(kOffPrev + 128);

        LoadMatrix(matrix);

        // softmax over the last axis: row maximum, exp, row sum, normalize.
        BlockReduceMax<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1, static_cast<int32_t>(kRowLanes));
        Brcb(div, sum, 1, BrcbRepeatParams(1, static_cast<uint16_t>(kRowLanes)));
        Sub<float, false>(mat, mat, div, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        Exp<float, false>(mat, mat, MASK_PLACEHOLDER, 1, UnaryRepeatParams(1, 1, 8, 8));
        BlockReduceSum<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1, static_cast<int32_t>(kRowLanes));
        Brcb(div, sum, 1, BrcbRepeatParams(1, static_cast<uint16_t>(kRowLanes)));
        Div<float, false>(mat, mat, div, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));

        // p = softmax(x) + eps on the valid lanes; padded rows are dropped.
        Add<float, false>(mat, mat, At<float>(kOffEps), MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        if (n_ < kRowLanes) {
            Duplicate(mat[n_ * kRowLanes], 0.0F,
                      static_cast<int32_t>((kRowLanes - n_) * kRowLanes));
            }

        // First column normalization: sum over dim -2 plus eps.
        NormalizeColumns(mat, tmpA, tmpB, sum, div, cons, col);
        Adds(prev, mat, 0.0F, static_cast<int32_t>(kLanes));

        // The loop stops on a bounded error argument rather than on a fixed
        // iteration count. One finished iteration moves every lane by at most
        // `move`; if the two most recent iterations shrank by the observed
        // factor rho, the whole remaining tail is bounded by move*rho/(1-rho).
        // Leaving once move/(1-rho) is within kSettledTail therefore keeps the
        // emitted tile within kSettledTail of every later iterate, including the
        // one the documented schedule would have produced. That argument needs
        // the normalization to stay contractive, so an eps small enough for
        // normalized entries to underflow keeps the exact requested schedule.
        const bool contractive = tiling_.eps > kMinContractEps;
        float prevMove = -1.0F;
        for (uint32_t iteration = 1; iteration < iterations_; ++iteration) {
            RowStep(mat, sum, div, cons);
            NormalizeColumns(mat, tmpA, tmpB, sum, div, cons, col);
            if (!contractive) {
                continue;
            }
            const float move = TileMove(mat, prev, tmpA, chkMax);
            if (move <= 0.0F) {
                break;  // bit exact fixed point: every later iterate is identical
            }
            if (prevMove > 0.0F) {
                float rho = move / prevMove;
                if (rho > kMaxRho) {
                    rho = kMaxRho;
                }
                if (move / (1.0F - rho) <= kSettledTail) {
                    break;
                }
            }
            prevMove = move;
            Adds(prev, mat, 0.0F, static_cast<int32_t>(kLanes));
        }

        StoreMatrix(matrix);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    MhcSinkhornTilingData tiling_{};
    uint32_t first_ = 0;
    uint32_t count_ = 0;
    uint32_t n_ = 0;
    uint32_t iterations_ = 0;
};

template <typename DT_X>
__global__ __aicore__ void mhc_sinkhorn(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights, GM_ADDR workspace,
                                        GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcSinkhornTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcSinkhornTilingData, tiling_data, tiling);
    KernelMhcSinkhorn<DT_X> op;
    op.Init(logits, weights, tiling_data);
    op.Process();
}
