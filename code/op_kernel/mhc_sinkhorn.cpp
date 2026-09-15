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
constexpr float kNegHuge = -__builtin_inff();  // keeps padding lanes out of the row maximum

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
        eps_ = data.eps;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y));
        tableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling) + kTilingWords);
        const uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t begin = blockIdx * data.matricesPerCore;
        const uint32_t end = begin + data.matricesPerCore;
        first_ = begin < data.matrices ? begin : data.matrices;
        const uint32_t limit = end < data.matrices ? end : data.matrices;
        count_ = limit > first_ ? limit - first_ : 0U;
        n_ = data.n;
        iterations_ = data.iterations;
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
        return LocalTensor<U>(TPosition::VECCALC, byteOffset,
                              (kArenaBytes - byteOffset) / sizeof(U));
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
        const float epsValue = eps_;
        // The guard vector is one full fill followed by a narrow fill from lane
        // zero: a partial fill at lane n would start off the 32B boundary.
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
        SetVectorMask<float>(static_cast<int32_t>(kLanes));
        Div<float, false>(mat, mat, col, MASK_PLACEHOLDER,
                          1, BinaryRepeatParams(1, 1, 0, 8, 8, 8));
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
        // schedule converges to. The test follows the normalization, so the tile
        // that is left behind is the column-normalized iterate the schedule would
        // have emitted, and no probe is spent on a tile that cannot be settled.
        const bool contractive = eps_ > kMinContractEps;
        for (uint32_t iteration = 1; iteration < iterations_; ++iteration) {
            BlockReduceSum<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1,
                                         static_cast<int32_t>(kRowLanes));
            Add<float, false>(sum, sum, cons, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            Brcb(div, sum, 1, BrcbRepeatParams(1, static_cast<uint16_t>(kRowLanes)));
            Div<float, false>(mat, mat, div, MASK_PLACEHOLDER, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            NormalizeColumns(mat, tmpA, tmpB, cons, col);
            if (iteration < 2U || !contractive || iteration + 1U >= iterations_) {
                // Either the eps regime no longer contracts, so the documented
                // schedule has to run in full, or this was the last permitted
                // step and there is nothing left to probe.
                continue;
            }
            // The padded rows carry a zero sum and would always fail the lower
            // test, so only the valid n rows are read back.
            BlockReduceSum<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1,
                                         static_cast<int32_t>(kRowLanes));
            if (n_ == 6U) {
                Sync<HardEvent::V_S>();
                float high = sum.GetValue(0);
                float low = high;
                for (uint32_t row = 1; row < n_; ++row) {
                    const float value = sum.GetValue(static_cast<int32_t>(row));
                    if (value > high) {
                        high = value;
                    }
                    if (value < low) {
                        low = value;
                    }
                }
                if (high - 1.0F <= kSettledDev && 1.0F - low <= kSettledDev) {
                    break;
                }
            } else {
                SetVectorMask<float>(static_cast<int32_t>(n_));
                Adds<float, false>(tmpA, sum, -1.0F, MASK_PLACEHOLDER, 1,
                                    UnaryRepeatParams(1, 1, 8, 8));
                Abs<float, false>(tmpA, tmpA, MASK_PLACEHOLDER, 1,
                                   UnaryRepeatParams(1, 1, 8, 8));
                WholeReduceMax<float, false>(tmpB, tmpA, MASK_PLACEHOLDER,
                    1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
                Sync<HardEvent::V_S>();
                const float deviation = tmpB.GetValue(0);
                SetVectorMask<float>(static_cast<int32_t>(kLanes));
                if (deviation <= kSettledDev) {
                    break;
                }
            }
        }

    }

    GlobalTensor<DT_X> xGm_;
    GlobalTensor<DT_X> yGm_;
    GlobalTensor<uint32_t> tableGm_;
    float eps_;
    uint32_t first_ = 0;
    uint32_t count_ = 0;
    uint32_t n_ = 0;
    uint32_t iterations_ = 0;
};

template <typename DT_X, uint32_t N>
class KernelMhcSinkhornPair {
    static constexpr uint32_t kBatch = N == 4U ? 2U : 4U;
    static constexpr uint32_t kElements = kBatch * N * N;
    static constexpr uint32_t kRows = N == 6U ? 6U : 8U;
    static constexpr uint32_t kRepeats = kRows * kBatch / 8U;
    static constexpr uint32_t kPairLanes = 2U * kRows * 8U;
    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t offset) {
        return LocalTensor<U>(TPosition::VECCALC, offset, 2048U / sizeof(U));
    }
    __aicore__ inline void Columns(const LocalTensor<float> &mat) {
        auto a = At<float>(2048);
        auto b = At<float>(4096);
        auto col = At<float>(10240);
        auto div = At<float>(8192);
        if constexpr (N == 6U) {
            Add<float, false>(a, mat, mat[8], MASK_PLACEHOLDER, 2, {1, 2, 2, 8, 16, 16});
            SetVectorMask<float>(32);
            Add<float, false>(b, a, a[8], MASK_PLACEHOLDER, 1, {1, 3, 3, 8, 24, 24});
            Add<float, false>(col, b, a[16], MASK_PLACEHOLDER, 1, {1, 1, 3, 8, 8, 24});
            Add<float, false>(col, col, At<float>(12288), MASK_PLACEHOLDER, 1, {1, 1, 0, 8, 8, 0});
            SetVectorMask<float>(48);
            Div<float, false>(mat, mat, col, MASK_PLACEHOLDER, kBatch, {1, 1, 0, 6, 6, 1});
            SetVectorMask<float>(64);
        } else {
            Add<float, false>(a, mat, mat[8], MASK_PLACEHOLDER, kBatch, {1, 1, 1, 8, 8, 8});
            Add<float, false>(b, a, a[16], MASK_PLACEHOLDER, kBatch, {1, 1, 1, 8, 8, 8});
            Add<float, false>(col, b, b[32], MASK_PLACEHOLDER, kBatch, {1, 1, 1, 8, 8, 8});
            SetVectorMask<float>(8);
            Add<float, false>(col, col, At<float>(12288), MASK_PLACEHOLDER, kBatch, {1, 1, 1, 8, 8, 0});
            SetVectorMask<float>(64);
            Div<float, false>(mat, mat, col, MASK_PLACEHOLDER, kBatch, {1, 1, 0, 8, 8, 8});
        }
    }
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
                                    const MhcSinkhornTilingData &data) {
        GlobalTensor<DT_X> xGm;
        GlobalTensor<DT_X> yGm;
        GlobalTensor<uint32_t> tables;
        const uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * kBatch;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + first * N * N);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + first * N * N);
        tables.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling) + kTilingWords + kGatherWords);
        auto mat = At<float>(0);
        auto sum = At<float>(6144);
        auto div = At<float>(8192);
        auto cons = At<float>(12288);
        auto eps = At<float>(14336);
        auto stage = At<DT_X>(16384);
        auto compact = At<float>(18432);
        auto table = At<uint32_t>(20480);
        auto probe = At<float>(22528);
        auto output = At<DT_X>(24576);
        if constexpr (N < 8U) {
            DataCopyPad(table, tables, DataCopyExtParams{1, (N == 6U ? 168U : 208U) * sizeof(uint32_t), 0, 0, 0},
                        DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
        }
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            auto input = N < 8U ? compact : mat;
            DataCopyPad(input, xGm, DataCopyExtParams{1, kElements * sizeof(float), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0.0F});
        } else {
            DataCopyPad(stage, xGm, DataCopyExtParams{1, kElements * sizeof(DT_X), 0, 0, 0},
                        DataCopyPadExtParams<DT_X>{false, 0, 0, static_cast<DT_X>(0)});
        }
        if constexpr (N < 8U) {
            Duplicate(cons, 1.0F, 8);
            Duplicate(cons, data.eps, N);
            for (uint32_t pair = 0; pair < kBatch / 2U; ++pair) {
                compact.SetValue(252U + pair * 2U * N * N, kNegHuge);
            }
            Sync<HardEvent::S_V>();
        } else {
            Duplicate(cons, data.eps, 8U * kBatch);
            Duplicate(eps, data.eps, 64U * kBatch);
        }
        Sync<HardEvent::MTE2_V>();
        if constexpr (sizeof(DT_X) != sizeof(float)) {
            auto input = N < 8U ? compact : mat;
            Cast(input, stage, RoundMode::CAST_NONE, kElements);
        }
        if constexpr (N < 8U) {
            for (uint32_t pair = 0; pair < kBatch / 2U; ++pair) {
                Gather(mat[pair * kPairLanes], compact[pair * 2U * N * N], table, 0, kPairLanes);
            }
        }
        SetVectorMask<float>(64);
        BlockReduceMax<float, false>(sum, mat, kRepeats, MASK_PLACEHOLDER, 1, 1, 8);
        Brcb(div, sum, kRepeats, BrcbRepeatParams(1, 8));
        Sub<float, false>(mat, mat, div, MASK_PLACEHOLDER, kRepeats, {1, 1, 1, 8, 8, 8});
        Exp<float, false>(mat, mat, MASK_PLACEHOLDER, kRepeats, {1, 1, 8, 8});
        BlockReduceSum<float, false>(sum, mat, kRepeats, MASK_PLACEHOLDER, 1, 1, 8);
        Brcb(div, sum, kRepeats, BrcbRepeatParams(1, 8));
        Div<float, false>(mat, mat, div, MASK_PLACEHOLDER, kRepeats, {1, 1, 1, 8, 8, 8});
        if constexpr (N == 6U) {
            SetVectorMask<float>(6);
            Adds<float, false>(mat, mat, data.eps, MASK_PLACEHOLDER, 24, {1, 1, 1, 1});
            SetVectorMask<float>(64);
        } else {
            Add<float, false>(mat, mat, eps, MASK_PLACEHOLDER, kRepeats, {1, 1, 1, 8, 8, 8});
        }
        Columns(mat);
        for (uint32_t iteration = 1; iteration < data.iterations; ++iteration) {
            BlockReduceSum<float, false>(sum, mat, kRepeats, MASK_PLACEHOLDER, 1, 1, 8);
            SetVectorMask<float>(kRows * kBatch);
            if constexpr (N == 6U) {
                Adds<float, false>(sum, sum, data.eps, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            } else {
                Add<float, false>(sum, sum, cons, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
            }
            SetVectorMask<float>(64);
            Brcb(div, sum, kRepeats, BrcbRepeatParams(1, 8));
            Div<float, false>(mat, mat, div, MASK_PLACEHOLDER, kRepeats, {1, 1, 1, 8, 8, 8});
            Columns(mat);
            if (iteration >= 2U && data.eps > kMinContractEps && iteration + 1U < data.iterations) {
                BlockReduceSum<float, false>(sum, mat, kRepeats, MASK_PLACEHOLDER, 1, 1, 8);
                SetVectorMask<float>(kRows * kBatch);
                Adds<float, false>(probe, sum, -1.0F, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                Abs<float, false>(probe, probe, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                WholeReduceMax<float, false>(probe[64], probe, MASK_PLACEHOLDER,
                    1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
                Sync<HardEvent::V_S>();
                const float deviation = probe.GetValue(64);
                SetVectorMask<float>(64);
                if (deviation <= kSettledDev) {
                    break;
                }
            }
        }
        if constexpr (N < 8U) {
            for (uint32_t pair = 0; pair < kBatch / 2U; ++pair) {
                Gather(compact[pair * 2U * N * N], mat[pair * kPairLanes], table[kPairLanes], 0, 2U * N * N);
            }
        }
        auto packed = N < 8U ? compact : mat;
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            Sync<HardEvent::V_MTE3>();
            DataCopyPad(yGm, packed, DataCopyExtParams{1, kElements * sizeof(float), 0, 0, 0});
        } else {
            Cast(output, packed, RoundMode::CAST_NONE, kElements);
            Sync<HardEvent::V_MTE3>();
            DataCopyPad(yGm, output, DataCopyExtParams{1, kElements * sizeof(DT_X), 0, 0, 0});
        }
    }
};

template <typename DT_X>
class KernelMhcSinkhornNarrowPair {
    template <typename U>
    __aicore__ inline LocalTensor<U> At(uint32_t offset) {
        return LocalTensor<U>(TPosition::VECCALC, offset, 1024U / sizeof(U));
    }
    __aicore__ inline void Columns(const LocalTensor<float> &mat) {
        auto a = At<float>(1024);
        auto col = At<float>(2048);
        auto div = At<float>(4096);
        Add<float, false>(a, mat, mat[8], MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        Add<float, false>(col, a, a[16], MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        SetVectorMask<float>(8);
        Add<float, false>(col, col, At<float>(5120), MASK_PLACEHOLDER, 2, {1, 1, 1, 4, 4, 0});
        SetVectorMask<float>(32);
        Div<float, false>(mat, mat, col, MASK_PLACEHOLDER, 2, {1, 1, 0, 4, 4, 4});
        SetVectorMask<float>(64);
    }
public:
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
                                    const MhcSinkhornTilingData &data) {
        GlobalTensor<DT_X> xGm;
        GlobalTensor<DT_X> yGm;
        GlobalTensor<uint32_t> tables;
        const uint64_t first = static_cast<uint64_t>(GetBlockIdx()) * 2U;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(x) + first * 16U);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(y) + first * 16U);
        tables.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling) + kTilingWords + kGatherWords);
        auto mat = At<float>(0);
        auto sum = At<float>(3072);
        auto div = At<float>(4096);
        auto cons = At<float>(5120);
        auto stage = At<DT_X>(6144);
        auto compact = At<float>(7168);
        auto table = At<uint32_t>(8192);
        auto probe = At<float>(9216);
        auto output = At<DT_X>(10240);
        DataCopyPad(table, tables, DataCopyExtParams{1, 96U * sizeof(uint32_t), 0, 0, 0},
                    DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            DataCopyPad(compact, xGm, DataCopyExtParams{1, 32U * sizeof(float), 0, 0, 0},
                        DataCopyPadExtParams<float>{false, 0, 0, 0.0F});
        } else {
            DataCopyPad(stage, xGm, DataCopyExtParams{1, 32U * sizeof(DT_X), 0, 0, 0},
                        DataCopyPadExtParams<DT_X>{false, 0, 0, static_cast<DT_X>(0)});
        }
        Duplicate(cons, 1.0F, 8);
        Duplicate(cons, data.eps, 4);
        compact.SetValue(252, kNegHuge);
        Sync<HardEvent::S_V>();
        Sync<HardEvent::MTE2_V>();
        if constexpr (sizeof(DT_X) != sizeof(float)) {
            Cast(compact, stage, RoundMode::CAST_NONE, 32);
        }
        Gather(mat, compact, table, 0, 64);
        SetVectorMask<float>(64);
        BlockReduceMax<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1, 8);
        Brcb(div, sum, 1, BrcbRepeatParams(1, 8));
        Sub<float, false>(mat, mat, div, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        Exp<float, false>(mat, mat, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
        BlockReduceSum<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1, 8);
        Brcb(div, sum, 1, BrcbRepeatParams(1, 8));
        Div<float, false>(mat, mat, div, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
        SetVectorMask<float>(4);
        Adds<float, false>(mat, mat, data.eps, MASK_PLACEHOLDER, 8, {1, 1, 1, 1});
        SetVectorMask<float>(64);
        Columns(mat);
        for (uint32_t iteration = 1; iteration < data.iterations; ++iteration) {
            BlockReduceSum<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1, 8);
            SetVectorMask<float>(8);
            Adds<float, false>(sum, sum, data.eps, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
            SetVectorMask<float>(64);
            Brcb(div, sum, 1, BrcbRepeatParams(1, 8));
            Div<float, false>(mat, mat, div, MASK_PLACEHOLDER, 1, {1, 1, 1, 8, 8, 8});
            Columns(mat);
            if (iteration >= 2U && data.eps > kMinContractEps && iteration + 1U < data.iterations) {
                BlockReduceSum<float, false>(sum, mat, 1, MASK_PLACEHOLDER, 1, 1, 8);
                SetVectorMask<float>(8);
                Adds<float, false>(probe, sum, -1.0F, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                Abs<float, false>(probe, probe, MASK_PLACEHOLDER, 1, {1, 1, 8, 8});
                WholeReduceMax<float, false>(probe[64], probe, MASK_PLACEHOLDER,
                    1, 1, 1, 8, ReduceOrder::ORDER_ONLY_VALUE);
                Sync<HardEvent::V_S>();
                const float deviation = probe.GetValue(64);
                SetVectorMask<float>(64);
                if (deviation <= kSettledDev) {
                    break;
                }
            }
        }
        Gather(compact, mat, table[64], 0, 32);
        if constexpr (sizeof(DT_X) == sizeof(float)) {
            Sync<HardEvent::V_MTE3>();
            DataCopyPad(yGm, compact, DataCopyExtParams{1, 32U * sizeof(float), 0, 0, 0});
        } else {
            Cast(output, compact, RoundMode::CAST_NONE, 32);
            Sync<HardEvent::V_MTE3>();
            DataCopyPad(yGm, output, DataCopyExtParams{1, 32U * sizeof(DT_X), 0, 0, 0});
        }
    }
};

template <typename DT_X, uint32_t GROUP_ORDER, uint32_t ACTIVE_GROUPS>
__global__ __aicore__ void mhc_sinkhorn(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights, GM_ADDR workspace,
                                        GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcSinkhornTilingData);
    if constexpr (GROUP_ORDER != 0U && ACTIVE_GROUPS != 0U) {
        if (static_cast<uint32_t>(GetBlockIdx()) >= ACTIVE_GROUPS) {
            return;
        }
    }
    // The six fields this kernel needs are read straight from the tiling block
    // as scalars. The structured copy would move the whole block onto the stack
    // first, and at this kernel size that copy alone costs a few hundred
    // nanoseconds, more than the arithmetic it describes.
    __gm__ const MhcSinkhornTilingData *raw =
        reinterpret_cast<__gm__ const MhcSinkhornTilingData *>(tiling);
    if constexpr (GROUP_ORDER == 8U && ACTIVE_GROUPS == 0U) {
        constexpr uint32_t groupMatrices = GROUP_ORDER == 4U ? 2U : 4U;
        if (static_cast<uint32_t>(GetBlockIdx()) * groupMatrices >= raw->matrices) {
            return;
        }
    }
    MhcSinkhornTilingData tiling_data;
    tiling_data.matrices = raw->matrices;
    tiling_data.matricesPerCore = raw->matricesPerCore;
    tiling_data.n = raw->n;
    tiling_data.iterations = raw->iterations;
    tiling_data.eps = raw->eps;
    tiling_data.dtypeSize = raw->dtypeSize;
    tiling_data.reserved0 = 0U;
    tiling_data.reserved1 = 0U;
    if constexpr (GROUP_ORDER != 0U) {
        constexpr uint32_t groupMatrices = GROUP_ORDER == 4U ? 2U : 4U;
        if (static_cast<uint32_t>(GetBlockIdx()) * groupMatrices + groupMatrices <= tiling_data.matrices) {
            if constexpr (GROUP_ORDER == 4U) {
                KernelMhcSinkhornNarrowPair<DT_X> op;
                op.Process(logits, weights, tiling, tiling_data);
            } else {
                KernelMhcSinkhornPair<DT_X, GROUP_ORDER> op;
                op.Process(logits, weights, tiling, tiling_data);
            }
            PipeBarrier<PIPE_ALL>();
            return;
        }
    }
    KernelMhcSinkhorn<DT_X> op;
    op.Init(logits, weights, tiling, tiling_data);
    op.Process();
    PipeBarrier<PIPE_ALL>();
}
