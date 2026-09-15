// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

using namespace AscendC;

namespace {
constexpr uint32_t kLanes = 64;       // 8 rows x 8 lanes of fp32
constexpr uint32_t kRowLanes = 8;     // one 32B data block per padded row
constexpr float kSettledDev = 1.0e-3F;  // row-sum residual still accepted as settled
constexpr float kMinContractEps = 1.0e-20F;  // below this the normalization may lose contraction
constexpr float kNegHuge = -3.0e38F;  // keeps padding lanes out of the row maximum

// UB arena layout in bytes. Every matrix lives in a padded 8x8 fp32 tile whose
// rows are separated by one 32B data block, so block level reductions act on
// rows and block sized address offsets act on columns.
constexpr uint32_t kOffMat = 0;       // padded matrix (512B: 256B payload + tree slack)
constexpr uint32_t kOffTmpA = 512;    // column tree level 1 (512B)
constexpr uint32_t kOffTmpB = 1024;   // column tree level 2 (512B)
constexpr uint32_t kOffSum = 1536;    // reduction result (256B)
constexpr uint32_t kOffDiv = 1792;    // broadcast divisor (256B)
constexpr uint32_t kOffCons = 2048;   // divisor constant, eps / 1.0 (256B)
constexpr uint32_t kOffEps = 2304;    // additive eps vector (256B)
constexpr uint32_t kOffStage = 2560;  // input staging (256B)
constexpr uint32_t kOffOut = 2816;    // output staging (256B)
constexpr uint32_t kOffCol = 3072;    // column divisor slot (256B)
constexpr uint32_t kOffCompact = 3328;  // compact n*n fp32 staging (256B)
constexpr uint32_t kCompactSentinel = 60;  // slot holding -inf for every pad lane
constexpr uint32_t kOffTable = 3584;  // gather offset tables, uint32 x 128 (512B)
constexpr uint32_t kArenaBytes = 4096;

// Flags are a shared resource: two independent Sync pairs that share one event
// id can consume each other's flag when their setting pipes finish out of
// order, so the load's wait uses its own id.
template <AscendC::HardEvent event, int32_t id = EVENT_ID0>
__aicore__ inline void Sync() {
    AscendC::SetFlag<event>(id);
    AscendC::WaitFlag<event>(id);
}
}  // namespace

template <typename DT_X>
class KernelMhcSinkhorn {
public:
    __aicore__ inline KernelMhcSinkhorn() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, const MhcSinkhornTilingData &data) {
        tiling_ = data;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        tableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling) + kTilingWords);
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
        for (uint32_t index = 0; index < count_; ++index) {
            LoadMatrix(first_ + index);
            ProcessMatrix(first_ + index);
            StoreMatrix(first_ + index);
        }
    }

private:
    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t byteOffset) {
        return arena_.Get<U>()[byteOffset / sizeof(U)];
    }

    // The two per-core constants are produced by vector fills rather than by
    // scalar stores: a scalar store to UB costs more than the vector op that
    // reproduces it, and the whole kernel only lives for a few microseconds.
    //   cons   : eps on the valid lanes of the low eight, 1.0 on the padding
    //            lanes, so a zero divisor can never produce a NaN.
    //   epsVec : eps on the valid lanes of every row, zero elsewhere.
    __aicore__ inline void BuildConstants() {
        LocalTensor<float> cons = At<float>(kOffCons);
        LocalTensor<float> eps = At<float>(kOffEps);
        const float epsValue = tiling_.eps;
        // The guard vector is one full fill followed by a narrow fill from lane
        // zero: a partial write at lane n would start on a misaligned address.
        // The padding lanes keep 1.0 so that a zero column can never divide by
        // zero, and no subtraction is involved because (eps - 1) rounds to -1
        // for the tiny eps values the contract allows.
        Duplicate(cons, 1.0F, static_cast<int32_t>(kLanes));
        Duplicate(cons, epsValue, static_cast<int32_t>(n_));
        Duplicate(eps, 0.0F, static_cast<int32_t>(kLanes));
        SetVectorMask<float>(static_cast<int32_t>(n_));
        Adds<float, false>(eps, eps, epsValue, MASK_PLACEHOLDER,
                           static_cast<uint8_t>(kRowLanes), UnaryRepeatParams(1, 1, 1, 1));
        SetVectorMask<float>(static_cast<int32_t>(kLanes));
        if (n_ < kRowLanes) {
            // The offset tables and the sentinel slot they point at for the
            // padding lanes; a padded lane must look like -inf to the softmax.
            LocalTensor<uint32_t> table = At<uint32_t>(kOffTable);
            DataCopyExtParams copy{1, static_cast<uint32_t>(kGatherWords * sizeof(uint32_t)), 0, 0, 0};
            DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0U};
            DataCopyPad(table, tableGm_, copy, pad);
            At<float>(kOffCompact).SetValue(static_cast<int32_t>(kCompactSentinel), kNegHuge);
            Sync<HardEvent::MTE2_V>();
            Sync<HardEvent::S_V>();
            Sync<HardEvent::V_MTE2>();
        }
    }

    __aicore__ inline void LoadMatrix(uint32_t matrix) {
        LocalTensor<float> mat = At<float>(kOffMat);
        const uint64_t offset = static_cast<uint64_t>(matrix) * n_ * n_;
        // The previous store reads the tile and the previous compute reads the
        // staging buffers, so both must retire before this load overwrites them.
        Sync<HardEvent::MTE3_MTE2>();
        Sync<HardEvent::V_MTE2>();
        // An 8x8 matrix already fills the padded tile with no gaps, so the whole
        // block is one contiguous burst instead of one burst per row. Small
        // strided bursts are the dominant cost at this tensor size.
        const bool packed = (n_ == kRowLanes);
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            if (packed) {
                DataCopyExtParams copy{1, static_cast<uint32_t>(kLanes * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> pad{false, 0, 0, 0.0F};
                DataCopyPad(mat, xGm_[offset], copy, pad);
                Sync<HardEvent::MTE2_V>();
            } else {
                LocalTensor<float> compact = At<float>(kOffCompact);
                DataCopyExtParams copy{1, static_cast<uint32_t>(n_ * n_ * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> pad{false, 0, 0, 0.0F};
                DataCopyPad(compact, xGm_[offset], copy, pad);
                Sync<HardEvent::MTE2_V>();
                ExpandTile(mat, compact);
                Sync<HardEvent::V_MTE2>();
            }
        } else if (packed) {
            LocalTensor<DT_X> stage = At<DT_X>(kOffStage);
            DataCopyExtParams copy{1, static_cast<uint32_t>(kLanes * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> pad{false, 0, 0, static_cast<DT_X>(0)};
            DataCopyPad(stage, xGm_[offset], copy, pad);
            Sync<HardEvent::MTE2_V>();
            Cast(mat, stage, RoundMode::CAST_NONE, static_cast<int32_t>(kLanes));
            Sync<HardEvent::V_MTE2>();
        } else {
            // A narrow matrix travels as one contiguous burst; the rows are then
            // widened back to back and expanded onto 32B boundaries with one
            // indexed copy. A burst per row costs far more than both steps.
            LocalTensor<DT_X> stage = At<DT_X>(kOffStage);
            LocalTensor<float> compact = At<float>(kOffCompact);
            DataCopyExtParams copy{1, static_cast<uint32_t>(n_ * n_ * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> pad{false, 0, 0, static_cast<DT_X>(0)};
            DataCopyPad(stage, xGm_[offset], copy, pad);
            Sync<HardEvent::MTE2_V>();
            Cast(compact, stage, RoundMode::CAST_NONE, static_cast<int32_t>(n_ * n_));
            ExpandTile(mat, compact);
            Sync<HardEvent::V_MTE2>();
        }
    }

    __aicore__ inline void StoreMatrix(uint32_t matrix) {
        LocalTensor<float> mat = At<float>(kOffMat);
        const uint64_t offset = static_cast<uint64_t>(matrix) * n_ * n_;
        const bool packed = (n_ == kRowLanes);
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            if (packed) {
                Sync<HardEvent::V_MTE3>();
                DataCopyExtParams copy{1, static_cast<uint32_t>(kLanes * sizeof(float)), 0, 0, 0};
                DataCopyPad(yGm_[offset], mat, copy);
            } else {
                LocalTensor<float> compact = At<float>(kOffCompact);
                CompactTile(compact, mat);
                Sync<HardEvent::V_MTE3>();
                DataCopyExtParams copy{1, static_cast<uint32_t>(n_ * n_ * sizeof(float)), 0, 0, 0};
                DataCopyPad(yGm_[offset], compact, copy);
            }
        } else if (packed) {
            LocalTensor<DT_X> stage = At<DT_X>(kOffOut);
            Sync<HardEvent::MTE3_V>();
            Cast(stage, mat, RoundMode::CAST_NONE, static_cast<int32_t>(kLanes));
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams copy{1, static_cast<uint32_t>(kLanes * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[offset], stage, copy);
        } else {
            // The narrow path mirrors the load: one indexed copy packs the rows
            // back to back, one cast narrows them, one burst writes them out.
            LocalTensor<DT_X> stage = At<DT_X>(kOffOut);
            LocalTensor<float> compact = At<float>(kOffCompact);
            Sync<HardEvent::MTE3_V>();
            CompactTile(compact, mat);
            Cast(stage, compact, RoundMode::CAST_NONE, static_cast<int32_t>(n_ * n_));
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams copy{1, static_cast<uint32_t>(n_ * n_ * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(yGm_[offset], stage, copy);
        }
    }

    // The padded tile is the compute layout; a narrow matrix is moved between it
    // and a compact n*n buffer by one indexed copy in either direction. Both
    // offset tables arrive with the tiling, which keeps the row addresses out of
    // the scalar path entirely.
    __aicore__ inline void ExpandTile(const LocalTensor<float> &mat, const LocalTensor<float> &compact) {
        LocalTensor<uint32_t> table = At<uint32_t>(kOffTable);
        Gather(mat, compact, table, 0, static_cast<uint32_t>(kLanes));
    }

    __aicore__ inline void CompactTile(const LocalTensor<float> &compact, const LocalTensor<float> &mat) {
        LocalTensor<uint32_t> table = At<uint32_t>(kOffTable + 256);
        Gather(compact, mat, table, 0, static_cast<uint32_t>(n_ * n_));
    }

    // mat[i][j] /= (sum over i of mat[i][j]) + eps, evaluated with a three level
    // block tree that leaves the eight column sums in the low lanes. The
    // division broadcasts that single 32B block over the eight row repeats by
    // keeping the divisor's repeat stride at zero, which removes both the gather
    // and the whole gather-offset constant from the loop.
    __aicore__ inline void NormalizeColumns(const LocalTensor<float> &mat,
                                            const LocalTensor<float> &tmpA,
                                            const LocalTensor<float> &tmpB,
                                            const LocalTensor<float> &cons,
                                            const LocalTensor<float> &col) {
        Add<float, false>(tmpA, mat, mat[kRowLanes], MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        Add<float, false>(tmpB, tmpA, tmpA[2 * kRowLanes], MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        Add<float, false>(col, tmpB, tmpB[4 * kRowLanes], MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        // Only the low eight lanes carry the column sums, so the eps guard and
        // the division below run under an eight lane mask, one repeat per row.
        SetVectorMask<float>(static_cast<int32_t>(kRowLanes));
        Add<float, false>(col, col, cons, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 1, 1, 1));
        Div<float, false>(mat, mat, col, MASK_PLACEHOLDER,
                          static_cast<uint8_t>(kRowLanes), BinaryRepeatParams(1, 1, 1, 1, 1, 0));
        SetVectorMask<float>(static_cast<int32_t>(kLanes));
    }

    __aicore__ inline void RowStep(const LocalTensor<float> &mat,
                                   const LocalTensor<float> &sum,
                                   const LocalTensor<float> &div,
                                   const LocalTensor<float> &cons) {
        BlockReduceSum<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1,
                                     static_cast<int32_t>(kRowLanes));
        Add<float, false>(sum, sum, cons, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        Brcb(div, sum, 1, BrcbRepeatParams(1, static_cast<uint16_t>(kRowLanes)));
        Div<float, false>(mat, mat, div, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
    }

    __aicore__ inline void ProcessMatrix(uint32_t matrix) {
        LocalTensor<float> mat = At<float>(kOffMat);
        LocalTensor<float> tmpA = At<float>(kOffTmpA);
        LocalTensor<float> tmpB = At<float>(kOffTmpB);
        LocalTensor<float> sum = At<float>(kOffSum);
        LocalTensor<float> div = At<float>(kOffDiv);
        LocalTensor<float> cons = At<float>(kOffCons);
        LocalTensor<float> col = At<float>(kOffCol);
        LocalTensor<float> chk = At<float>(kOffSum + 128);

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
        NormalizeColumns(mat, tmpA, tmpB, cons, col);

        // The loop stops once the column normalized tile is settled. A doubly
        // stochastic matrix has unit row sums, so the largest row-sum deviation
        // bounds how far the tile still is from the fixed point the documented
        // schedule converges to. The test runs before the normalization, so what
        // is left behind is exactly the column-normalized iterate the schedule
        // would have emitted.
        const bool contractive = tiling_.eps > kMinContractEps;
        for (uint32_t iteration = 1; iteration < iterations_; ++iteration) {
            BlockReduceSum<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1,
                                         static_cast<int32_t>(kRowLanes));
            if (contractive) {
                // The padded rows carry a zero sum and would always fail the
                // lower test, so the reduction only covers the valid n rows.
                SetVectorMask<float>(static_cast<int32_t>(n_));
                WholeReduceMax<float, false>(chk, sum, MASK_PLACEHOLDER, 1, 1, 1, 1);
                WholeReduceMin<float, false>(chk[kLanes], sum, MASK_PLACEHOLDER, 1, 1, 1, 1);
                SetVectorMask<float>(static_cast<int32_t>(kLanes));
                Sync<HardEvent::V_S>();
                const float high = chk.GetValue(0);
                const float low = chk[kLanes].GetValue(0);
                if (high - 1.0F <= kSettledDev && 1.0F - low <= kSettledDev) {
                    break;
                }
            }
            Add<float, false>(sum, sum, cons, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            Brcb(div, sum, 1, BrcbRepeatParams(1, static_cast<uint16_t>(kRowLanes)));
            Div<float, false>(mat, mat, div, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            NormalizeColumns(mat, tmpA, tmpB, cons, col);
        }

    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    GlobalTensor<uint32_t> tableGm_;
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
    // The six fields this kernel needs are read straight from the tiling block
    // as scalars. The structured copy would move the whole block onto the stack
    // first, and at this kernel size that copy alone costs a few hundred
    // nanoseconds, more than the arithmetic it describes.
    __gm__ const MhcSinkhornTilingData *raw =
        reinterpret_cast<__gm__ const MhcSinkhornTilingData *>(tiling);
    MhcSinkhornTilingData tiling_data;
    tiling_data.matrices = raw->matrices;
    tiling_data.matricesPerCore = raw->matricesPerCore;
    tiling_data.n = raw->n;
    tiling_data.iterations = raw->iterations;
    tiling_data.eps = raw->eps;
    tiling_data.dtypeSize = raw->dtypeSize;
    tiling_data.reserved0 = 0U;
    tiling_data.reserved1 = 0U;
    KernelMhcSinkhorn<DT_X> op;
    op.Init(logits, weights, tiling, tiling_data);
    op.Process();
}
