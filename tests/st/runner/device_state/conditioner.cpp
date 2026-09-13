// Common test precondition, run in a separate process before each measurement.
// It models nonzero scratch left by preceding device work. It is not part of
// any submitted operator and its duration is never included in Mock timing.
#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void mock_ub_condition(GM_ADDR result, uint32_t pattern) {
    LocalTensor<uint32_t> values(TPosition::VECCALC, 0, 32768);
    Duplicate(values, pattern, 16384);
    Duplicate(values[16384], pattern, 16384);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    GlobalTensor<uint32_t> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(result) + GetBlockIdx()*8, 8);
    DataCopy(output, values, 8);
}

extern "C" void mock_prepare_ub(void *stream, void *result, uint32_t pattern) {
    mock_ub_condition<<<40, nullptr, stream>>>(static_cast<uint8_t*>(result), pattern);
}
