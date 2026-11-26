// PrivateLM laptop worker: serves this computer's GPU (or CPU) to a phone
// cluster master, like a phone set to Worker. Built from the same llama.cpp
// tree as the app, so it speaks the same RPC protocol version.
//
//   privatelm-worker --info              print this computer's specs as JSON
//   privatelm-worker [--port 50052] [--cpu] [--threads N] [--cache DIR]
//
// Usually started by privatelm_worker.py, which also advertises it on the
// Wi-Fi so the master's "Find nearby" sees it.

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach/mach.h>
#  include <sys/sysctl.h>
#else
#  include <fstream>
#  include <sstream>
#endif

/// System memory available to new allocations, in MB (what the OS could hand
/// out now without swapping), or 0 if unknown.
static long long availableRamMb() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return 0;
    return (long long)(status.ullAvailPhys >> 20);
#elif defined(__APPLE__)
    vm_size_t page = 0;
    host_page_size(mach_host_self(), &page);
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count) != KERN_SUCCESS) {
        return 0;
    }
    // Free + cached files macOS drops on demand.
    const uint64_t pages = (uint64_t)vm.free_count + vm.inactive_count + vm.purgeable_count;
    return (long long)((pages * page) >> 20);
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            std::istringstream fields(line.substr(13));
            long long kb = 0;
            fields >> kb;
            return kb / 1024;
        }
    }
    return 0;
#endif
}

/// Same multi-threaded memory read as the phones' benchmark (GB/s), so the
/// master compares laptop and phones on one scale.
static double memoryBandwidth(int threads) {
    const int n = std::max(1, threads);
    const size_t per_thread = (256u << 20) / sizeof(uint64_t) / n;  // 256 MB total
    std::vector<uint64_t> data(per_thread * n, 1);
    std::vector<uint64_t> sums(n, 0);
    const int passes = 4;
    auto run = [&](int t) {
        const uint64_t* p = data.data() + per_thread * t;
        uint64_t a = 0, b = 0, c = 0, d = 0;
        for (int pass = 0; pass < passes; pass++) {
            for (size_t i = 0; i + 4 <= per_thread; i += 4) {
                a += p[i]; b += p[i + 1]; c += p[i + 2]; d += p[i + 3];
            }
        }
        sums[t] = a + b + c + d;
    };
    run(0);  // warm up: page in the buffer
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < n; t++) pool.emplace_back(run, t);
    for (auto& th : pool) th.join();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (sums[0] == 42) std::fprintf(stderr, " ");  // keep the loop from being optimized away
    return secs > 0 ? (double)data.size() * sizeof(uint64_t) * passes / secs / 1e9 : 0;
}

static int defaultThreads() {
#if defined(__APPLE__)
    int count = 0;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &count, &size, nullptr, 0) == 0 && count > 0) {
        return count;
    }
#endif
    const unsigned hw = std::thread::hardware_concurrency();
    return std::max(2, (int)(hw / 2));  // physical cores, roughly
}

/// The device to serve: the first GPU, else an integrated GPU, else the CPU.
static ggml_backend_dev_t pickDevice(bool cpu_only) {
    ggml_backend_dev_t igpu = nullptr, cpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_GPU:
                if (!cpu_only) return dev;
                break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:
                if (!igpu) igpu = dev;
                break;
            case GGML_BACKEND_DEVICE_TYPE_CPU:
                if (!cpu) cpu = dev;
                break;
            default:
                break;
        }
    }
    return (!cpu_only && igpu) ? igpu : cpu;
}

/// What the served device can hold, in MB. A discrete GPU holds layers in its
/// own memory; the CPU, and GPUs that share system RAM (Apple silicon,
/// integrated), are limited by the RAM free right now.
static long long usableMb(ggml_backend_dev_t dev) {
    size_t free = 0, total = 0;
    ggml_backend_dev_memory(dev, &free, &total);
    const long long dev_mb = (long long)(free >> 20);
    const long long ram_mb = availableRamMb();
    const auto type = ggml_backend_dev_type(dev);
#if defined(__APPLE__)
    const bool shares_ram = true;
#else
    const bool shares_ram = type != GGML_BACKEND_DEVICE_TYPE_GPU;
#endif
    if (type == GGML_BACKEND_DEVICE_TYPE_CPU) return ram_mb > 0 ? ram_mb : dev_mb;
    if (shares_ram && ram_mb > 0) return std::min(dev_mb > 0 ? dev_mb : ram_mb, ram_mb);
    return dev_mb;
}

static std::string jsonEscape(const char* s) {
    std::string out;
    for (; s && *s; s++) {
        if (*s == '"' || *s == '\\') out += '\\';
        if ((unsigned char)*s >= 0x20) out += *s;
    }
    return out;
}

static void usage() {
    std::fprintf(stderr,
        "usage: privatelm-worker [--info] [--host 0.0.0.0] [--port 50052] [--cpu]\n"
        "                        [--threads N] [--cache DIR]\n");
}

int main(int argc, char** argv) {
    bool info = false, cpu_only = false;
    std::string host = "0.0.0.0", cache;
    int port = 50052, threads = defaultThreads();
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (arg == "--info") info = true;
        else if (arg == "--cpu") cpu_only = true;
        else if (arg == "--host") host = next();
        else if (arg == "--port") port = std::atoi(next());
        else if (arg == "--threads") threads = std::max(1, std::atoi(next()));
        else if (arg == "--cache") cache = next();
        else { usage(); return 2; }
    }

    ggml_backend_dev_t dev = pickDevice(cpu_only);
    if (!dev) {
        std::fprintf(stderr, "No device to serve: this build found no CPU or GPU backend\n");
        return 1;
    }

    if (info) {
        std::printf("{\"device\": \"%s\", \"description\": \"%s\", \"type\": \"%s\", "
                    "\"usableMb\": %lld, \"bandwidth\": %.1f, \"threads\": %d}\n",
                    jsonEscape(ggml_backend_dev_name(dev)).c_str(),
                    jsonEscape(ggml_backend_dev_description(dev)).c_str(),
                    ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU ? "cpu" : "gpu",
                    usableMb(dev), memoryBandwidth(threads), threads);
        return 0;
    }

    const std::string endpoint = host + ":" + std::to_string(port);
    std::fprintf(stderr, "PrivateLM worker: serving %s (%s), %lld MB usable, on %s (protocol %d.%d.%d)\n",
                 ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), usableMb(dev),
                 endpoint.c_str(), RPC_PROTO_MAJOR_VERSION, RPC_PROTO_MINOR_VERSION,
                 RPC_PROTO_PATCH_VERSION);
    // Returns only if the server can't start (the port is usually taken).
    ggml_backend_rpc_start_server(endpoint.c_str(), cache.empty() ? nullptr : cache.c_str(),
                                  (size_t)threads, 1, &dev);
    std::fprintf(stderr, "Worker stopped: port %d may already be in use\n", port);
    return 1;
}
