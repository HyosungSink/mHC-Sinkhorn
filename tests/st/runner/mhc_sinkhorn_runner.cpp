// MhcSinkhorn ACLNN fixture runner.
//
// Executes one or more fixture directories against the locally built custom
// operator package. Each fixture descriptor selects an input shape/layout,
// the two operator attributes, and an optional float32 reference output used
// for result verification. Warmup can require both a launch count and elapsed
// preparation time after the first synchronized launch. Python reports the median
// of three process means. Every timed kernel record is required; later drain
// launches have a separate correlation ID.
// Workspace queries, allocation, and result copies are outside kernel timing.
//
// Four timing modes are available (--mode):
//   sync   : device events around `inner` launches, synchronise once, event
//            elapsed time divided by `inner`. Includes the device idle gaps
//            produced by host-side launch overhead.
//   launch : host monotonic clock around the enqueue of `inner` launches,
//            synchronise once, divided by `inner`. Pure host enqueue cost.
//   batch  : identical to sync except that the event pair is created once for
//            the whole case instead of once per timed sample.
//   mspti  : kernel activity duration (end - start), matched by correlation ID.
//            Requires libmspti.so preloading.

#include <acl/acl.h>
#include <dlfcn.h>

#include "aclnn_mhc_sinkhorn.h"
#include "mspti.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using Config = std::unordered_map<std::string, std::string>;

constexpr std::size_t kMsptiBufferSize = 8u * 1024u * 1024u;
constexpr std::size_t kMsptiBufferAlign = 8u;
constexpr int kMsptiFlushAttempts = 400;
constexpr int kMsptiFlushIntervalUs = 5000;

[[noreturn]] void Fail(const std::string &message)
{
    throw std::runtime_error(message);
}

void Check(aclError status, const std::string &operation)
{
    if (status != ACL_SUCCESS) {
        Fail(operation + " failed with status " + std::to_string(status));
    }
}


Config ReadConfig(const std::filesystem::path &path)
{
    std::ifstream stream(path);
    if (!stream) {
        Fail("cannot open fixture descriptor: " + path.string());
    }
    Config config;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const auto separator = line.find('=');
        if (separator == std::string::npos || separator == 0) {
            Fail("invalid fixture descriptor line");
        }
        config.emplace(line.substr(0, separator), line.substr(separator + 1));
    }
    return config;
}

const std::string &Required(const Config &config, const std::string &key)
{
    const auto item = config.find(key);
    if (item == config.end()) {
        Fail("fixture descriptor is missing " + key);
    }
    return item->second;
}

std::string Optional(const Config &config, const std::string &key,
                     const std::string &fallback)
{
    const auto item = config.find(key);
    return item == config.end() ? fallback : item->second;
}

std::vector<int64_t> ParseShape(const std::string &text)
{
    std::vector<int64_t> shape;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = text.find(',', begin);
        const auto token = text.substr(begin, end == std::string::npos ? end : end - begin);
        if (token.empty()) {
            break;
        }
        shape.push_back(std::stoll(token));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    if (shape.empty()) {
        Fail("empty tensor shape: " + text);
    }
    return shape;
}

std::size_t TypeSize(const std::string &dtype)
{
    if (dtype == "float16" || dtype == "bfloat16") {
        return 2;
    }
    if (dtype == "float32") {
        return 4;
    }
    Fail("unsupported dtype " + dtype);
}

aclDataType AclType(const std::string &dtype)
{
    if (dtype == "float16") {
        return ACL_FLOAT16;
    }
    if (dtype == "bfloat16") {
        return ACL_BF16;
    }
    if (dtype == "float32") {
        return ACL_FLOAT;
    }
    Fail("unsupported dtype " + dtype);
}

std::size_t TensorBytes(const std::vector<int64_t> &shape, const std::string &dtype)
{
    std::size_t elements = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0 ||
            elements > std::numeric_limits<std::size_t>::max() /
                           static_cast<std::size_t>(dimension)) {
            Fail("invalid tensor shape");
        }
        elements *= static_cast<std::size_t>(dimension);
    }
    return elements * TypeSize(dtype);
}

std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &shape)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index-- > 1;) {
        strides[index - 1] = strides[index] * shape[index];
    }
    return strides;
}

std::vector<std::uint8_t> ReadBinary(const std::filesystem::path &path,
                                     std::size_t expectedBytes)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() < 0 ||
        static_cast<std::size_t>(stream.tellg()) != expectedBytes) {
        Fail("fixture tensor byte count mismatch: " + path.string());
    }
    std::vector<std::uint8_t> bytes(expectedBytes);
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()))) {
        Fail("cannot read fixture tensor: " + path.string());
    }
    return bytes;
}

class DeviceTensor {
public:
    DeviceTensor() = default;
    DeviceTensor(const DeviceTensor &) = delete;
    DeviceTensor &operator=(const DeviceTensor &) = delete;

    ~DeviceTensor()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
        if (data_ != nullptr) {
            aclrtFree(data_);
        }
    }

    void Create(const std::vector<int64_t> &viewShape,
                const std::vector<int64_t> &storageShape,
                const std::string &dtype,
                const std::vector<std::uint8_t> *host)
    {
        const std::size_t bytes = TensorBytes(storageShape, dtype);
        if (TensorBytes(viewShape, dtype) != bytes) {
            Fail("view and storage tensor byte counts differ");
        }
        Check(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate tensor");
        if (host == nullptr) {
            Check(aclrtMemset(data_, bytes, 0, bytes), "initialize device tensor");
        } else {
            if (host->size() != bytes) {
                Fail("host tensor byte count mismatch");
            }
            Check(aclrtMemcpy(data_, bytes, host->data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE),
                  "copy fixture tensor");
        }
        const auto strides = ContiguousStrides(viewShape);
        tensor_ = aclCreateTensor(viewShape.data(), static_cast<size_t>(viewShape.size()),
                                  AclType(dtype), strides.data(), 0, ACL_FORMAT_ND,
                                  storageShape.data(), static_cast<size_t>(storageShape.size()),
                                  data_);
        if (tensor_ == nullptr) {
            Fail("create ACL tensor failed");
        }
    }

    aclTensor *Get() const { return tensor_; }

    std::vector<std::uint8_t> CopyToHost(std::size_t bytes) const
    {
        std::vector<std::uint8_t> host(bytes);
        Check(aclrtMemcpy(host.data(), bytes, data_, bytes, ACL_MEMCPY_DEVICE_TO_HOST),
              "copy output tensor");
        return host;
    }

private:
    void *data_ = nullptr;
    aclTensor *tensor_ = nullptr;
};

class Workspace {
public:
    explicit Workspace(std::size_t bytes)
    {
        if (bytes != 0) {
            Check(aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate workspace");
        }
    }
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    ~Workspace()
    {
        if (data_ != nullptr) {
            aclrtFree(data_);
        }
    }
    void *Get() const { return data_; }

private:
    void *data_ = nullptr;
};

class Session {
public:
    Session() = default;
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    ~Session() { Cleanup(); }

    void Initialize(int32_t deviceId)
    {
        deviceId_ = deviceId;
        try {
            Check(aclInit(nullptr), "initialize ACL");
            initialized_ = true;
            Check(aclrtSetDevice(deviceId_), "select device");
            deviceSet_ = true;
            Check(aclrtCreateContext(&context_, deviceId_), "create context");
            Check(aclrtSetCurrentContext(context_), "select context");
            Check(aclrtCreateStream(&stream_), "create stream");
        } catch (...) {
            Cleanup();
            throw;
        }
    }

    aclrtStream Stream() const { return stream_; }

private:
    void Cleanup() noexcept
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (context_ != nullptr) {
            aclrtDestroyContext(context_);
            context_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (initialized_) {
            aclFinalize();
            initialized_ = false;
        }
    }

    int32_t deviceId_ = 0;
    bool initialized_ = false;
    bool deviceSet_ = false;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
};

enum class TimingMode { kSync, kLaunch, kBatch, kMspti };

const char *ModeName(TimingMode mode)
{
    switch (mode) {
        case TimingMode::kSync:
            return "sync";
        case TimingMode::kLaunch:
            return "launch";
        case TimingMode::kBatch:
            return "batch";
        case TimingMode::kMspti:
            return "mspti";
    }
    return "sync";
}

TimingMode ParseMode(const std::string &text)
{
    if (text == "sync") {
        return TimingMode::kSync;
    }
    if (text == "launch") {
        return TimingMode::kLaunch;
    }
    if (text == "batch") {
        return TimingMode::kBatch;
    }
    if (text == "mspti") {
        return TimingMode::kMspti;
    }
    Fail("unsupported timing mode: " + text);
}

struct Options {
    int32_t deviceId = 0;
    int warmup = 5;
    int warmupUs = 0;
    int repeat = 60;
    int intervalUs = 0;
    bool prepareOnly = false;
    std::string stateLibrary;
    int inner = 1;
    TimingMode mode = TimingMode::kSync;
    std::filesystem::path report;
    std::filesystem::path outputDir;
    std::vector<std::filesystem::path> fixtures;
};

Options ParseArguments(int argc, char **argv)
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--device" && index + 1 < argc) {
            options.deviceId = std::stoi(argv[++index]);
        } else if (argument == "--warmup" && index + 1 < argc) {
            options.warmup = std::stoi(argv[++index]);
        } else if (argument == "--warmup-us" && index + 1 < argc) {
            options.warmupUs = std::stoi(argv[++index]);
        } else if (argument == "--repeat" && index + 1 < argc) {
            options.repeat = std::stoi(argv[++index]);
        } else if (argument == "--interval-us" && index + 1 < argc) {
            options.intervalUs = std::stoi(argv[++index]);
        } else if (argument == "--state-library" && index + 1 < argc) {
            options.stateLibrary = argv[++index];
        } else if (argument == "--prepare-only") {
            options.prepareOnly = true;
        } else if (argument == "--inner" && index + 1 < argc) {
            options.inner = std::stoi(argv[++index]);
        } else if (argument == "--mode" && index + 1 < argc) {
            options.mode = ParseMode(argv[++index]);
        } else if (argument == "--output-dir" && index + 1 < argc) {
            options.outputDir = argv[++index];
        } else if (argument == "--report" && index + 1 < argc) {
            options.report = argv[++index];
        } else {
            options.fixtures.emplace_back(argument);
        }
    }
    if (options.report.empty() || options.fixtures.empty() || options.warmup < 0 ||
        options.repeat <= 0 || options.inner <= 0 || options.intervalUs < 0 || options.warmupUs < 0) {
        Fail("usage: mhc_sinkhorn_runner --report <path> [--device id] "
             "[--warmup n] [--warmup-us us] [--repeat n] [--inner k] "
             "[--interval-us n] [--state-library path] [--prepare-only] "
             "[--mode {sync,launch,batch,mspti}] <fixture>...");
    }
    return options;
}

struct CaseResult {
    int warmupCount = 0;
    double warmupElapsedUs = 0;
    std::string caseId;
    std::string status = "Runtime Error";
    std::string stage = "infrastructure";
    int32_t aclStatus = 0;
    bool hasTiming = false;
    double deviceUsP50 = 0.0;
    double deviceUsMin = 0.0;
    double deviceUsMax = 0.0;
    double deviceUsMean = 0.0;
    std::uint64_t workspaceBytes = 0;
    std::size_t sampleCount = 0;
    std::size_t mismatchCount = 0;
    std::size_t elementCount = 0;
};

struct Invocation {
    DeviceTensor logits;
    DeviceTensor weights;
    std::vector<int64_t> weightsShape;
    std::string weightsDtype;
    std::size_t weightsBytes = 0;
    int64_t iterations = 20;
    double eps = 1.0e-6;
    bool hasExpected = false;
    std::string expectedFile;
    double atol = 1.0e-5;
    double rtol = 1.0e-5;
    bool verify = true;
};

// ---------------------------------------------------------------------------
// MSPTI device-duration collection.
//
// Activity delivery is asynchronous. External correlation IDs separate timed
// launches from warmups without depending on arrival order or complete capture.
// One process owns one case. Every valid correlated record contributes to the
// median; no duration-based filtering or timing correction is performed.
// ---------------------------------------------------------------------------

struct KernelSample {
    std::string name;
    std::uint64_t durationNs = 0;
    std::uint64_t startNs = 0;
    std::uint64_t correlationId = 0;
};

std::mutex g_msptiMutex;
std::vector<KernelSample> g_kernelSamples;
bool g_msptiReady = false;
msptiSubscriberHandle g_subscriber = nullptr;
std::unordered_map<std::uint64_t, bool> g_correlations;

void MsptiBufferRequest(std::uint8_t **buffer, std::size_t *size, std::size_t *maxNumRecords)
{
    const std::size_t requested = kMsptiBufferSize;
    auto *allocated = static_cast<std::uint8_t *>(std::aligned_alloc(kMsptiBufferAlign, requested));
    if (allocated == nullptr) {
        *buffer = nullptr;
        *size = 0;
        *maxNumRecords = 0;
        return;
    }
    *buffer = allocated;
    *size = requested;
    *maxNumRecords = 0;
}

void MsptiBufferComplete(std::uint8_t *buffer, std::size_t size, std::size_t validSize)
{
    (void)size;
    if (validSize != 0) {
        msptiActivity *record = nullptr;
        while (msptiActivityGetNextRecord(buffer, validSize, &record) == MSPTI_SUCCESS) {
            if (record != nullptr && record->kind == MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
                const auto *external = reinterpret_cast<const msptiActivityExternalCorrelation *>(record);
                std::lock_guard<std::mutex> guard(g_msptiMutex);
                g_correlations[external->correlationId] = external->externalId == 1;
            }
            if (record == nullptr || record->kind != MSPTI_ACTIVITY_KIND_KERNEL) {
                continue;
            }
            const auto *kernel = reinterpret_cast<const msptiActivityKernel *>(record);
            std::lock_guard<std::mutex> guard(g_msptiMutex);
            g_kernelSamples.push_back(KernelSample{kernel->name == nullptr ? "<unknown>" : kernel->name,
                kernel->end > kernel->start ? kernel->end - kernel->start : 0, kernel->start, kernel->correlationId});
        }
    }
    std::free(buffer);
}

void EnableMspti()
{
    if (g_msptiReady) {
        return;
    }
    if (msptiSubscribe(&g_subscriber, nullptr, nullptr) != MSPTI_SUCCESS) {
        Fail("cannot subscribe to the MSPTI activity source");
    }
    if (msptiActivityRegisterCallbacks(MsptiBufferRequest, MsptiBufferComplete) != MSPTI_SUCCESS) {
        Fail("cannot register the MSPTI activity callbacks");
    }
    if (msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL) != MSPTI_SUCCESS) {
        Fail("cannot enable the MSPTI kernel activity");
    }
    if (msptiActivityEnable(MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) != MSPTI_SUCCESS) {
        Fail("cannot enable external correlation records");
    }
    if (msptiActivityEnable(MSPTI_ACTIVITY_KIND_API) != MSPTI_SUCCESS) {
        Fail("cannot enable runtime activity correlation");
    }
    g_msptiReady = true;
}

void DisableMspti()
{
    if (!g_msptiReady) {
        return;
    }
    msptiActivityDisable(MSPTI_ACTIVITY_KIND_KERNEL);
    msptiActivityDisable(MSPTI_ACTIVITY_KIND_API);
    msptiActivityDisable(MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION);
    msptiActivityFlushAll(0);
    msptiUnsubscribe(g_subscriber);
    g_subscriber = nullptr;
    g_msptiReady = false;
}

// Moves every record collected so far out of the shared collector.
std::vector<KernelSample> TakeKernelSamples()
{
    std::lock_guard<std::mutex> guard(g_msptiMutex);
    std::vector<KernelSample> taken;
    taken.swap(g_kernelSamples);
    return taken;
}

std::size_t BufferedKernelSamples()
{
    std::lock_guard<std::mutex> guard(g_msptiMutex);
    return g_kernelSamples.size();
}

std::size_t BufferedTimedSamples()
{
    std::lock_guard<std::mutex> guard(g_msptiMutex);
    std::size_t count = 0;
    for (const auto &sample : g_kernelSamples) {
        const auto correlation = g_correlations.find(sample.correlationId);
        if (sample.durationNs > 0 && correlation != g_correlations.end() && correlation->second) {
            ++count;
        }
    }
    return count;
}

bool NameMentionsSinkhorn(const std::string &name)
{
    std::string lowered(name);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return lowered.find("sinkhorn") != std::string::npos;
}

// Waits for the records of one sample and pops exactly that many from the
// front of the collector: MSPTI delivers whole buffers at a time, so a single
// flush frequently contains records of several samples.
std::vector<KernelSample> TakeKernelSampleBatch(std::size_t expected, std::size_t requiredTimed = 0)
{
    for (int attempt = 0; attempt < kMsptiFlushAttempts; ++attempt) {
        msptiActivityFlushAll(1);
        if (BufferedKernelSamples() >= expected ||
            (requiredTimed != 0 && BufferedTimedSamples() >= requiredTimed)) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(kMsptiFlushIntervalUs));
    }
    std::lock_guard<std::mutex> guard(g_msptiMutex);
    const std::size_t count = std::min(expected, g_kernelSamples.size());
    std::vector<KernelSample> batch(g_kernelSamples.begin(),
                                    g_kernelSamples.begin() + static_cast<std::ptrdiff_t>(count));
    g_kernelSamples.erase(g_kernelSamples.begin(),
                          g_kernelSamples.begin() + static_cast<std::ptrdiff_t>(count));
    return batch;
}

// Preserve individual hardware timestamps, including untimed warmups.
void ReportKernelSamples(const std::string &caseId, const std::vector<KernelSample> &samples,
                         const std::filesystem::path &reportPath)
{
    std::ofstream correlations(reportPath.string() + ".correlations.tsv");
    for (const auto &entry : g_correlations) {
        correlations << entry.first << '\t' << entry.second << '\n';
    }
    std::ofstream report(reportPath.string() + ".kernels.tsv");
    report << "case_id\tstart_ns\tduration_ns\tcorrelation_id\ttimed\tkernel\n";
    for (const auto &sample : samples) {
        report << caseId << '\t' << sample.startNs << '\t' << sample.durationNs
               << '\t' << sample.correlationId << '\t' << g_correlations[sample.correlationId]
               << '\t' << sample.name << '\n';
    }
}

float ReadOutputValue(const std::vector<std::uint8_t> &bytes, std::size_t index,
                      const std::string &dtype)
{
    if (dtype == "float32") {
        float value;
        std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
        return value;
    }
    std::uint16_t bits;
    std::memcpy(&bits, bytes.data() + index * sizeof(bits), sizeof(bits));
    const unsigned exponent = (bits >> 10) & 31;
    const unsigned fraction = bits & 1023;
    const float sign = bits & 32768 ? -1.0F : 1.0F;
    if (exponent == 31) {
        return fraction ? std::numeric_limits<float>::quiet_NaN()
                        : sign * std::numeric_limits<float>::infinity();
    }
    return sign * (exponent ? std::ldexp(1.0F + fraction / 1024.0F, static_cast<int>(exponent) - 15)
                            : std::ldexp(static_cast<float>(fraction), -24));
}

bool VerifyOutput(const Invocation &invocation, const std::vector<std::uint8_t> &actual,
                  std::size_t bytes, CaseResult &result)
{
    if (!invocation.hasExpected) {
        return true;
    }
    const auto expected = ReadBinary(std::filesystem::path(invocation.expectedFile), bytes);
    const auto count = bytes / TypeSize(invocation.weightsDtype);
    std::size_t mismatched = 0;
    double worst = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
        const float a = ReadOutputValue(actual, index, invocation.weightsDtype);
        const float e = ReadOutputValue(expected, index, invocation.weightsDtype);
        const bool bothNaN = std::isnan(a) && std::isnan(e);
        const bool bothInf = std::isinf(a) && std::isinf(e) && ((a > 0) == (e > 0));
        if (bothNaN || bothInf) {
            continue;
        }
        if (!std::isfinite(a) || !std::isfinite(e)) {
            ++mismatched;
            continue;
        }
        const double tolerance = invocation.atol + invocation.rtol * std::fabs(static_cast<double>(e));
        const double error = std::fabs(static_cast<double>(a) - static_cast<double>(e));
        if (error > tolerance) {
            ++mismatched;
            worst = std::max(worst, error / std::max(1.0e-30, std::fabs(static_cast<double>(e))));
        }
    }
    result.mismatchCount = mismatched;
    result.elementCount = count;
    return mismatched == 0;
}

CaseResult RunCase(const std::filesystem::path &fixture, const Options &options,
                   aclrtStream stream)
{
    const Config config = ReadConfig(fixture / "case.conf");
    CaseResult result;
    result.caseId = Required(config, "case_id");

    Invocation invocation;
    const auto logitsShape = ParseShape(Required(config, "input.logits.shape"));
    const auto logitsStorageShape =
        ParseShape(Optional(config, "input.logits.storage_shape", ""));
    const auto logitsDtype = Optional(config, "input.logits.dtype", "float32");
    const auto logitsFile = Required(config, "input.logits.file");
    invocation.iterations = std::stoll(Optional(config, "iterations", "20"));
    invocation.eps = std::stod(Optional(config, "eps", "1e-6"));
    invocation.weightsShape = ParseShape(
        Optional(config, "output.weights.shape", Required(config, "input.logits.shape")));
    invocation.weightsDtype = Optional(config, "output.weights.dtype", logitsDtype);
    invocation.weightsBytes = TensorBytes(invocation.weightsShape, invocation.weightsDtype);
    invocation.atol = std::stod(Optional(config, "verify.atol", "1e-5"));
    invocation.rtol = std::stod(Optional(config, "verify.rtol", "1e-5"));
    const auto expected = Optional(config, "expected.weights.file", "");
    invocation.hasExpected = !expected.empty();
    invocation.expectedFile = (fixture / expected).string();
    invocation.verify = Optional(config, "verify.skip", "0") != "1";

    const auto logitsHost = ReadBinary(fixture / logitsFile,
                                        TensorBytes(logitsShape, logitsDtype));
    invocation.logits.Create(logitsShape,
                             logitsStorageShape.empty() ? logitsShape : logitsStorageShape,
                             logitsDtype, &logitsHost);
    invocation.weights.Create(invocation.weightsShape, invocation.weightsShape,
                              invocation.weightsDtype, nullptr);

    // One staged launch: the host-side workspace query has already produced the
    // executor, so enqueueing it is the only remaining step.
    struct StagedLaunch {
        aclOpExecutor *executor = nullptr;
        std::uint64_t workspaceBytes = 0;
    };

    std::unique_ptr<Workspace> workspace;
    std::size_t workspaceBytes = 0;

    // Host-side staging of `count` launches. This is deliberately outside of
    // every timed region: only the enqueue belongs to the measured window.
    auto stageLaunches = [&](int count, std::vector<StagedLaunch> &staged) -> bool {
        staged.clear();
        staged.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) {
            std::uint64_t bytes = 0;
            aclOpExecutor *executor = nullptr;
            const aclnnStatus queryStatus = aclnnMhcSinkhornGetWorkspaceSize(
                invocation.logits.Get(), nullptr, invocation.iterations, invocation.eps,
                invocation.weights.Get(), &bytes, &executor);
            if (queryStatus != 0) {
                result.status = "Runtime Error";
                result.stage = "workspace_query";
                result.aclStatus = queryStatus;
                return false;
            }
            result.workspaceBytes = std::max(result.workspaceBytes, bytes);
            if (workspace == nullptr || workspaceBytes != static_cast<std::size_t>(bytes)) {
                workspace = std::make_unique<Workspace>(static_cast<std::size_t>(bytes));
                workspaceBytes = static_cast<std::size_t>(bytes);
            }
            staged.push_back(StagedLaunch{executor, bytes});
        }
        return true;
    };

    // Enqueues the staged launches back to back on one workspace allocation.
    auto enqueueStaged = [&](const std::vector<StagedLaunch> &staged) -> bool {
        for (const auto &launch : staged) {
            const aclnnStatus launchStatus =
                aclnnMhcSinkhorn(workspace->Get(), launch.workspaceBytes, launch.executor, stream);
            if (launchStatus != 0) {
                result.status = "Runtime Error";
                result.stage = "kernel_launch";
                result.aclStatus = launchStatus;
                return false;
            }
        }
        return true;
    };

    // Stages and enqueues `count` launches.
    auto runLaunches = [&](int count) -> bool {
        std::vector<StagedLaunch> staged;
        return stageLaunches(count, staged) && enqueueStaged(staged);
    };

    auto syncStream = [&]() -> bool {
        const aclError syncStatus = aclrtSynchronizeStream(stream);
        if (syncStatus != ACL_SUCCESS) {
            result.status = "Runtime Error";
            result.stage = "stream_sync";
            result.aclStatus = syncStatus;
            return false;
        }
        return true;
    };

    if (options.mode == TimingMode::kMspti) {
        EnableMspti();
        TakeKernelSamples();
    }
    std::vector<std::uint8_t> firstOutput;
    std::vector<StagedLaunch> warmupStaged;
    std::chrono::steady_clock::time_point warmupBegin;
    while (result.warmupCount < options.warmup ||
           (options.warmupUs > 0 && (result.warmupCount == 0 ||
                                    result.warmupElapsedUs < options.warmupUs))) {
        if (options.intervalUs > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(options.intervalUs));
        }
        if (!stageLaunches(options.inner, warmupStaged)) return result;
        if (!enqueueStaged(warmupStaged) || !syncStream()) return result;
        if (result.warmupCount == 0) warmupBegin = std::chrono::steady_clock::now();
        if (result.warmupCount == 0 && invocation.verify) {
            firstOutput = invocation.weights.CopyToHost(invocation.weightsBytes);
        }
        ++result.warmupCount;
        result.warmupElapsedUs = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - warmupBegin).count();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(options.repeat));
    std::vector<StagedLaunch> staged;
    if (options.mode == TimingMode::kMspti) {
        // Runtime correlation IDs identify measured launches independently of
        // asynchronous delivery order and missing warmup records.
        if (msptiActivityPushExternalCorrelationId(MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 1) != MSPTI_SUCCESS) {
            Fail("cannot identify measured launch phase");
        }
        for (int repetition = 0; repetition < options.repeat; ++repetition) {
            if (options.intervalUs > 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(options.intervalUs));
            }
            if (!stageLaunches(options.inner, staged) || !enqueueStaged(staged) || !syncStream()) {
                DisableMspti();
                return result;
            }
        }
        std::uint64_t externalId = 0;
        if (msptiActivityPopExternalCorrelationId(MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &externalId) != MSPTI_SUCCESS) {
            Fail("cannot close measured launch phase");
        }
        // Short captures need later real launches to drain hardware activity
        // packets. A separate correlation phase excludes every drain launch.
        std::size_t drainLaunches = 0;
        if (options.repeat * options.inner <= 30) {
            drainLaunches = 48;
            if (msptiActivityPushExternalCorrelationId(MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, 2) != MSPTI_SUCCESS) {
                Fail("cannot identify activity drain phase");
            }
            if (!runLaunches(static_cast<int>(drainLaunches)) || !syncStream()) {
                DisableMspti();
                return result;
            }
            if (msptiActivityPopExternalCorrelationId(MSPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &externalId) != MSPTI_SUCCESS) {
                Fail("cannot close activity drain phase");
            }
        }
        const std::size_t expected = static_cast<std::size_t>(result.warmupCount + options.repeat) * options.inner + drainLaunches;
        auto records = TakeKernelSampleBatch(expected, drainLaunches == 0 ? 0 : options.repeat * options.inner);
        DisableMspti();
        const auto remaining = TakeKernelSamples();
        records.insert(records.end(), remaining.begin(), remaining.end());
        ReportKernelSamples(result.caseId, records, options.report);
        std::sort(records.begin(), records.end(), [](const KernelSample &a, const KernelSample &b) {
            return a.startNs < b.startNs;
        });
        for (const auto &record : records) {
            if (NameMentionsSinkhorn(record.name) && record.durationNs > 0 &&
                g_correlations[record.correlationId]) {
                samples.push_back(static_cast<double>(record.durationNs) / 1000.0);
            }
        }
        result.sampleCount = samples.size();
        const std::size_t measuredLaunches = static_cast<std::size_t>(options.repeat * options.inner);
        if (samples.size() != measuredLaunches) {
            result.status = "Runner Error";
            result.stage = "mspti_incomplete_capture";
            return result;
        }
    } else if (options.mode == TimingMode::kLaunch) {
        for (int repetition = 0; repetition < options.repeat; ++repetition) {
            if (!stageLaunches(options.inner, staged)) {
                return result;
            }
            const auto begin = std::chrono::steady_clock::now();
            if (!enqueueStaged(staged)) {
                return result;
            }
            const auto end = std::chrono::steady_clock::now();
            if (!syncStream()) {
                return result;
            }
            const auto elapsed = std::chrono::duration<double, std::micro>(end - begin).count();
            samples.push_back(elapsed / static_cast<double>(options.inner));
        }
    } else {
        const bool reuseEvents = options.mode == TimingMode::kBatch;
        aclrtEvent begin = nullptr;
        aclrtEvent end = nullptr;
        auto destroyEvents = [&begin, &end]() {
            if (begin != nullptr) {
                aclrtDestroyEvent(begin);
                begin = nullptr;
            }
            if (end != nullptr) {
                aclrtDestroyEvent(end);
                end = nullptr;
            }
        };
        if (reuseEvents) {
            Check(aclrtCreateEvent(&begin), "create begin event");
            Check(aclrtCreateEvent(&end), "create end event");
        }
        for (int repetition = 0; repetition < options.repeat; ++repetition) {
            if (!stageLaunches(options.inner, staged)) {
                destroyEvents();
                return result;
            }
            if (!reuseEvents) {
                Check(aclrtCreateEvent(&begin), "create begin event");
                Check(aclrtCreateEvent(&end), "create end event");
            }
            Check(aclrtRecordEvent(begin, stream), "record begin event");
            if (!enqueueStaged(staged)) {
                destroyEvents();
                return result;
            }
            Check(aclrtRecordEvent(end, stream), "record end event");
            if (!syncStream()) {
                destroyEvents();
                return result;
            }
            float elapsedMs = 0.0F;
            Check(aclrtEventElapsedTime(&elapsedMs, begin, end), "read event elapsed time");
            if (!reuseEvents) {
                destroyEvents();
            }
            samples.push_back(static_cast<double>(elapsedMs) * 1000.0 /
                              static_cast<double>(options.inner));
        }
        destroyEvents();
    }
    std::sort(samples.begin(), samples.end());
    result.deviceUsP50 = samples[(samples.size() - 1) / 2];
    result.deviceUsMin = samples.front();
    result.deviceUsMax = samples.back();
    result.deviceUsMean = 0.0;
    for (const double sample : samples) {
        result.deviceUsMean += sample;
    }
    result.deviceUsMean /= static_cast<double>(samples.size());
    // Numerical failure does not invalidate a completed timing measurement.
    result.hasTiming = true;

    if (invocation.verify && invocation.hasExpected) {
        const auto actual = invocation.weights.CopyToHost(invocation.weightsBytes);
        bool numericalMatch = VerifyOutput(invocation, actual, invocation.weightsBytes, result);
        const std::vector<std::uint8_t> *compared = &actual;
        if (!firstOutput.empty()) {
            CaseResult firstResult = result;
            if (!VerifyOutput(invocation, firstOutput, invocation.weightsBytes, firstResult)) {
                // Report the first observed numerical failure. Later execution
                // must not replace its error count with a different failure.
                result = firstResult;
                numericalMatch = false;
                compared = &firstOutput;
            }
        }
        if (!options.outputDir.empty()) {
            std::filesystem::create_directories(options.outputDir);
            auto writeOutput = [&](const std::vector<std::uint8_t> &bytes, const std::string &suffix) {
                std::ofstream values(options.outputDir / (result.caseId + suffix));
                values << std::setprecision(std::numeric_limits<float>::max_digits10);
                const auto count = invocation.weightsBytes / TypeSize(invocation.weightsDtype);
                for (std::size_t index = 0; index < count; ++index) {
                    values << ReadOutputValue(bytes, index, invocation.weightsDtype) << '\n';
                }
            };
            writeOutput(*compared, ".output.tsv");
            if (!firstOutput.empty() && firstOutput != actual) {
                writeOutput(compared == &actual ? firstOutput : actual,
                            compared == &actual ? ".first_output.tsv" : ".last_output.tsv");
            }
            std::ifstream maps("/proc/self/maps");
            std::ofstream libraries(options.outputDir / (result.caseId + ".libraries.tsv"));
            std::string mapping;
            while (std::getline(maps, mapping)) {
                if (mapping.find("libcust_opapi.so") != std::string::npos ||
                    mapping.find("libcustom_ascendc_cust_optiling.so") != std::string::npos) {
                    libraries << mapping << '\n';
                }
            }
        }
        if (!firstOutput.empty() && firstOutput != actual) {
            result.status = "Wrong Answer";
            result.stage = "determinism";
            return result;
        }
        if (!numericalMatch) {
            result.status = "Wrong Answer";
            result.stage = "result_verify";
            return result;
        }
    }
    result.status = "Passed";
    result.stage = "benchmark";
    return result;
}

void EmitResult(std::ofstream &report, const CaseResult &result, const Options &options)
{
    report << result.caseId << '\t' << result.status << '\t' << result.stage << '\t'
           << result.aclStatus << '\t';
    if (result.hasTiming) {
        report << result.deviceUsP50;
    } else {
        report << '-';
    }
    report << '\t' << result.warmupCount << '\t' << options.repeat << '\t'
           << result.workspaceBytes;
    if (result.hasTiming) {
        report << '\t' << result.deviceUsMin << '\t' << result.deviceUsMax << '\t'
               << result.deviceUsMean;
    } else {
        report << "\t-\t-\t-";
    }
    report << '\t' << ModeName(options.mode) << '\t' << options.inner;
    report << '\t' << result.sampleCount << '\t' << result.mismatchCount
           << '\t' << result.elementCount << '\t' << result.warmupElapsedUs << '\n';
}

}  // namespace

int main(int argc, char **argv)
{
    std::filesystem::path reportPath;
    try {
        const Options options = ParseArguments(argc, argv);
        reportPath = options.report;
        std::ofstream report(reportPath, std::ios::trunc);
        if (!report) {
            return 2;
        }
        report << "case_id\tstatus\tstage\tacl_status\tdevice_us_p50\twarmup\trepeat"
                  "\tworkspace\tus_min\tus_max\tus_mean\tmode\tinner"
                  "\tsample_count\tmismatch_count\telement_count\twarmup_elapsed_us\n";
        if (options.mode == TimingMode::kMspti) {
            if (options.fixtures.size() != 1) {
                Fail("MSPTI requires one fixture per process");
            }
            EnableMspti();
        }
        Session session;
        session.Initialize(options.deviceId);
        if (options.prepareOnly) {
            if (options.stateLibrary.empty()) Fail("missing device state library");
            // The library remains registered until process exit. The following
            // target measurement starts in a different process and ACL context.
            void *handle = dlopen(options.stateLibrary.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!handle) Fail(dlerror());
            auto prepare = reinterpret_cast<void(*)(void*, void*, uint32_t)>(dlsym(handle, "mock_prepare_ub"));
            if (!prepare) Fail("missing device state entry point");
            void *scratch = nullptr;
            Check(aclrtMalloc(&scratch, 40*32, ACL_MEM_MALLOC_HUGE_FIRST), "state allocation");
            prepare(session.Stream(), scratch, 0x3f800000U);
            Check(aclrtSynchronizeStream(session.Stream()), "state preparation");
            std::vector<uint32_t> receipt(40*8);
            Check(aclrtMemcpy(receipt.data(), 40*32, scratch, 40*32, ACL_MEMCPY_DEVICE_TO_HOST), "state receipt");
            Check(aclrtFree(scratch), "state release");
            if (!std::all_of(receipt.begin(), receipt.end(), [](uint32_t v) {return v==0x3f800000U;})) {
                Fail("not every device preparation block completed");
            }
            return 0;
        }
        for (const auto &fixture : options.fixtures) {
            try {
                EmitResult(report, RunCase(fixture, options, session.Stream()), options);
            } catch (...) {
                CaseResult result;
                result.caseId = fixture.filename().string();
                EmitResult(report, result, options);
            }
        }
        DisableMspti();
        return 0;
    } catch (...) {
        if (!reportPath.empty()) {
            std::ofstream report(reportPath, std::ios::app);
            if (report) {
                report << "__runner__\tRunner Error\tinfrastructure\t0\t-\t0\t0\t0\n";
            }
        }
        return 2;
    }
}
