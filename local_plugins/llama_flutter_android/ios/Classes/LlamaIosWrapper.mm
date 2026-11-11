#import "LlamaIosWrapper.h"
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "gguf.h"
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

#define LOGI(...) NSLog(@"[LlamaIOS] " __VA_ARGS__)
#define LOGE(...) NSLog(@"[LlamaIOS ERROR] " __VA_ARGS__)

// Global inference state (mirrors jni_wrapper.cpp)
static llama_model*        g_model    = nullptr;
static llama_context*      g_ctx      = nullptr;
static const llama_vocab*  g_vocab    = nullptr;
static llama_sampler*      g_sampler  = nullptr;
static std::atomic<bool>   g_stop_flag{false};
static int                 g_n_past   = 0;
static int                 g_system_prompt_length = 0;

// ---------------------------------------------------------------------------
// Phone cluster master (mirrors jni_wrapper.cpp)
//
// Workers serve their processors over ggml's RPC protocol. The master adds
// each worker as a remote device and offloads the model's last layers to
// them; this device runs the first layers on its CPU. Workers only ever see
// tensors, never the prompt text. The cluster set here applies to the next
// model load.
// ---------------------------------------------------------------------------
static std::mutex g_cluster_mutex;
static std::vector<std::string> g_cluster_endpoints;
static std::vector<float> g_cluster_weights;  // one per endpoint
static float g_cluster_local_weight = 0.0f;
static std::vector<std::string> g_registered_endpoints;
// Name and MiB of each weight buffer of the last cluster load, from
// llama.cpp's "model buffer size" lines.
static std::mutex g_load_log_mutex;
static std::vector<std::pair<std::string, double>> g_load_buffers;
static std::string g_load_error;

/// Cluster workers show up as GPU devices named "RPC<n>[host:port]"; they
/// must never be mistaken for this device's own GPU.
static bool isRpcDevice(ggml_backend_dev_t dev) {
    const char* name = ggml_backend_dev_name(dev);
    return name && std::strncmp(name, "RPC", 3) == 0;
}

/// llama.cpp's log during a cluster load: keeps where the weights went and
/// the last error, and passes everything on to stderr as the default does.
static void clusterLoadLog(ggml_log_level level, const char* text, void*) {
    if (!text) return;
    fputs(text, stderr);
    std::lock_guard<std::mutex> lock(g_load_log_mutex);
    static const char kSize[] = " model buffer size = ";
    if (const char* at = std::strstr(text, kSize)) {
        // "load_tensors: <name> model buffer size = <MiB> MiB"
        const char* colon = std::strchr(text, ':');
        const char* start = colon && colon < at ? colon + 1 : text;
        while (start < at && *start == ' ') start++;
        const char* end = at;
        while (end > start && end[-1] == ' ') end--;
        if (end > start) {
            g_load_buffers.emplace_back(std::string(start, end),
                                        std::strtod(at + sizeof(kSize) - 1, nullptr));
        }
    }
    if (level == GGML_LOG_LEVEL_ERROR) {
        g_load_error.assign(text);
    } else if (level == GGML_LOG_LEVEL_CONT && !g_load_error.empty()) {
        g_load_error.append(text);
    }
    if (g_load_error.size() > 4096) g_load_error.erase(0, g_load_error.size() - 4096);
}

/// The model's transformer block count from its GGUF header, or -1.
static int readBlockCount(const char* model_path) {
    gguf_init_params params = {/* .no_alloc = */ true, /* .ctx = */ nullptr};
    gguf_context* gguf = gguf_init_from_file(model_path, params);
    if (!gguf) return -1;
    int count = -1;
    const int64_t arch_id = gguf_find_key(gguf, "general.architecture");
    if (arch_id >= 0) {
        const std::string key = std::string(gguf_get_val_str(gguf, arch_id)) + ".block_count";
        const int64_t id = gguf_find_key(gguf, key.c_str());
        if (id >= 0) {
            switch (gguf_get_kv_type(gguf, id)) {
                case GGUF_TYPE_UINT32: count = (int)gguf_get_val_u32(gguf, id); break;
                case GGUF_TYPE_INT32:  count = gguf_get_val_i32(gguf, id); break;
                default: break;
            }
        }
    }
    gguf_free(gguf);
    return count;
}

struct ClusterPlan {
    std::vector<ggml_backend_dev_t> devices;  // null-terminated
    std::vector<float> split;                 // llama_max_devices() entries
    int n_gpu_layers = 0;
    llama_model_tensor_buft_override overrides[2] = {};
};

/// Connects to every configured worker and decides which layers go where:
/// this device keeps the first layers (by its weight), workers get the rest
/// split by theirs. Returns an error message, or empty on success.
static std::string planCluster(const char* model_path,
                               const std::vector<std::string>& endpoints,
                               const std::vector<float>& weights,
                               float local_weight,
                               ClusterPlan& plan) {
    const int n_layer = readBlockCount(model_path);
    if (n_layer <= 0) return "Could not read the model's layer count";

    float remote_weight = 0.0f;
    std::vector<float> device_weights;
    for (size_t i = 0; i < endpoints.size(); i++) {
        const float weight = std::max(i < weights.size() ? weights[i] : 1.0f, 0.01f);
        // Aborts the app if the worker can't be reached, so ClusterChannel
        // probes each worker over TCP before setting the cluster.
        ggml_backend_reg_t reg = ggml_backend_rpc_add_server(endpoints[i].c_str());
        if (!reg) return "Worker " + endpoints[i] + " reported no devices";
        if (std::find(g_registered_endpoints.begin(), g_registered_endpoints.end(),
                      endpoints[i]) == g_registered_endpoints.end()) {
            ggml_backend_register(reg);
            g_registered_endpoints.push_back(endpoints[i]);
        }
        const size_t n_dev = ggml_backend_reg_dev_count(reg);
        if (n_dev == 0) return "Worker " + endpoints[i] + " reported no devices";
        for (size_t d = 0; d < n_dev; d++) {
            plan.devices.push_back(ggml_backend_reg_dev_get(reg, d));
            device_weights.push_back(weight / n_dev);
        }
        remote_weight += weight;
    }
    if (plan.devices.empty()) return "No cluster workers";
    if (plan.devices.size() > llama_max_devices()) return "Too many cluster workers";
    plan.devices.push_back(nullptr);

    plan.split.assign(llama_max_devices(), 0.0f);
    for (size_t d = 0; d < device_weights.size(); d++) {
        plan.split[d] = device_weights[d];
    }

    // llama.cpp offloads the *last* n_gpu_layers blocks (+1 for the output
    // layer), so keeping `local` blocks here means offloading the rest.
    const float total = std::max(local_weight, 0.0f) + remote_weight;
    int local = (int)std::lround(n_layer * std::max(local_weight, 0.0f) / total);
    local = std::clamp(local, 0, n_layer);
    plan.n_gpu_layers = n_layer + 1 - local;

    // The output layer goes with the last offloaded layer, i.e. a worker.
    // It's the biggest single matmul per token (vocab x hidden) and its
    // result, a score per vocab entry (~600 KB), would cross Wi-Fi on every
    // token. Keep its weights here instead (tied models reuse token_embd).
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu) {
        plan.overrides[0] = {"^(output|token_embd)\\.weight$", ggml_backend_dev_buffer_type(cpu)};
        plan.overrides[1] = {nullptr, nullptr};
    }
    LOGI("Cluster: %d of %d layers local, rest on %zu worker device(s), output layer local",
         local, n_layer, device_weights.size());
    return {};
}

// UTF-8 helpers (identical logic to jni_wrapper.cpp)
static bool isValidUTF8(const char* str, size_t len) {
    if (!str) return false;
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str);
    size_t i = 0;
    while (i < len) {
        unsigned char c = bytes[i];
        if ((c & 0x80) == 0) { i++; continue; }
        int nb = 0;
        if      ((c & 0xE0) == 0xC0) nb = 2;
        else if ((c & 0xF0) == 0xE0) nb = 3;
        else if ((c & 0xF8) == 0xF0) nb = 4;
        else return false;
        if (i + nb > len) return false;
        for (int j = 1; j < nb; j++)
            if ((bytes[i+j] & 0xC0) != 0x80) return false;
        if (nb == 2 && (c & 0x1E) == 0) return false;
        if (nb == 3) {
            if (c == 0xED && (bytes[i+1] & 0x20) == 0x20) return false;
            if (c == 0xE0 && (bytes[i+1] & 0x20) == 0) return false;
        }
        if (nb == 4) {
            if (c > 0xF4) return false;
            if (c == 0xF0 && (bytes[i+1] & 0x30) == 0) return false;
            if (c == 0xF4 && bytes[i+1] > 0x8F) return false;
        }
        i += nb;
    }
    return true;
}

static std::string sanitizeUTF8(const char* str, size_t len) {
    if (!str || len == 0) return "";
    if (isValidUTF8(str, len)) return std::string(str, len);
    std::string result;
    result.reserve(len);
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(str);
    size_t i = 0;
    while (i < len) {
        unsigned char c = bytes[i];
        if ((c & 0x80) == 0) { result += c; i++; continue; }
        int nb = 0;
        if      ((c & 0xE0) == 0xC0) nb = 2;
        else if ((c & 0xF0) == 0xE0) nb = 3;
        else if ((c & 0xF8) == 0xF0) nb = 4;
        else { result += "\xEF\xBF\xBD"; i++; continue; }
        if (i + nb > len) { result += "\xEF\xBF\xBD"; break; }
        std::string seq(reinterpret_cast<const char*>(bytes + i), nb);
        if (isValidUTF8(seq.c_str(), nb)) result += seq;
        else result += "\xEF\xBF\xBD";
        i += nb;
    }
    return result;
}

@implementation LlamaIosWrapper

- (void)loadModelAtPath:(NSString*)path
               nThreads:(int)nThreads
            contextSize:(int)contextSize
             nGpuLayers:(int)nGpuLayers
       progressCallback:(LlamaProgressCallback)progressCallback {

    const char* model_path = [path UTF8String];
    LOGI("Loading model: %s", model_path);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = nGpuLayers;

    // Once cluster workers are registered, llama.cpp's default device list
    // would include them too: name this device's own GPUs instead.
    std::vector<ggml_backend_dev_t> local_devices;
    if (!g_registered_endpoints.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto type = ggml_backend_dev_type(dev);
            if ((type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                 type == GGML_BACKEND_DEVICE_TYPE_IGPU) && !isRpcDevice(dev)) {
                local_devices.push_back(dev);
            }
        }
        local_devices.push_back(nullptr);
        model_params.devices = local_devices.data();
    }

    // Cluster mode: this device runs on its CPU and the workers hold the rest.
    std::vector<std::string> endpoints;
    std::vector<float> weights;
    float local_weight;
    {
        std::lock_guard<std::mutex> lock(g_cluster_mutex);
        endpoints = g_cluster_endpoints;
        weights = g_cluster_weights;
        local_weight = g_cluster_local_weight;
    }
    const bool has_cluster = !endpoints.empty();
    ClusterPlan cluster;
    {
        std::lock_guard<std::mutex> lock(g_load_log_mutex);
        g_load_buffers.clear();
        g_load_error.clear();
    }
    if (has_cluster) {
        const std::string error = planCluster(model_path, endpoints, weights, local_weight, cluster);
        if (!error.empty()) {
            LOGE("Cluster: %s", error.c_str());
            [NSException raise:@"LlamaClusterError" format:@"Cluster: %s", error.c_str()];
        }
        model_params.devices = cluster.devices.data();
        model_params.tensor_split = cluster.split.data();
        model_params.n_gpu_layers = cluster.n_gpu_layers;
        if (cluster.overrides[0].pattern) {
            model_params.tensor_buft_overrides = cluster.overrides;
        }
        llama_log_set(clusterLoadLog, nullptr);
    }

    g_model = llama_model_load_from_file(model_path, model_params);
    if (!g_model) {
        LOGE("Failed to load model");
        if (has_cluster) {
            std::string detail;
            {
                std::lock_guard<std::mutex> lock(g_load_log_mutex);
                detail = g_load_error;
            }
            while (!detail.empty() && isspace((unsigned char)detail.back())) detail.pop_back();
            [NSException raise:@"LlamaLoadError"
                        format:@"Cluster: failed to load the model across the phone cluster%s%s",
                               detail.empty() ? "" : ": ", detail.c_str()];
        }
        [NSException raise:@"LlamaLoadError" format:@"Failed to load model at path: %@", path];
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = contextSize;
    ctx_params.n_threads       = nThreads;
    ctx_params.n_threads_batch = nThreads;
    ctx_params.n_batch         = 512;
    // Don't push this device's own tensor ops over Wi-Fi to a worker.
    if (has_cluster) ctx_params.op_offload = false;

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (!g_ctx) {
        llama_model_free(g_model);
        g_model = nullptr;
        [NSException raise:@"LlamaContextError" format:@"Failed to create context"];
    }

    g_vocab = llama_model_get_vocab(g_model);
    if (!g_vocab) {
        llama_free(g_ctx);
        llama_model_free(g_model);
        g_ctx = nullptr;
        g_model = nullptr;
        [NSException raise:@"LlamaVocabError" format:@"Failed to get vocab"];
    }

    g_n_past = 0;
    LOGI("Model loaded successfully");
    if (progressCallback) progressCallback(1.0);
}

- (void)generateWithPrompt:(NSString*)prompt
                 maxTokens:(int)maxTokens
               temperature:(double)temperature
                      topP:(double)topP
                      topK:(int)topK
                      minP:(double)minP
                  typicalP:(double)typicalP
             repeatPenalty:(double)repeatPenalty
         frequencyPenalty:(double)frequencyPenalty
          presencePenalty:(double)presencePenalty
              repeatLastN:(int)repeatLastN
                 mirostat:(int)mirostat
              mirostatTau:(double)mirostatTau
              mirostatEta:(double)mirostatEta
                     seed:(long long)seed
         penalizeNewline:(BOOL)penalizeNewline
            tokenCallback:(LlamaTokenCallback)tokenCallback {

    if (!g_model || !g_ctx || !g_vocab) {
        [NSException raise:@"LlamaStateError" format:@"Model not loaded"];
    }

    g_stop_flag = false;

    const char* prompt_str = [prompt UTF8String];
    const int prompt_len = (int)strlen(prompt_str);
    std::string sanitized = sanitizeUTF8(prompt_str, prompt_len);

    // Count tokens
    const int n_prompt_tokens = -llama_tokenize(g_vocab, sanitized.c_str(), (int)sanitized.size(), nullptr, 0, true, true);
    if (n_prompt_tokens <= 0) {
        [NSException raise:@"LlamaTokenizeError" format:@"Failed to tokenize prompt (%d tokens)", n_prompt_tokens];
    }

    std::vector<llama_token> tokens(n_prompt_tokens);
    const int actual = llama_tokenize(g_vocab, sanitized.c_str(), (int)sanitized.size(), tokens.data(), (int)tokens.size(), true, true);
    if (actual < 0) {
        [NSException raise:@"LlamaTokenizeError" format:@"Tokenize failed"];
    }
    tokens.resize(actual);

    const int n_ctx = llama_n_ctx(g_ctx);

    // Sliding window KV cache (mirrors Android: discard oldest 25%)
    if (g_n_past + (int)tokens.size() > n_ctx) {
        const int n_discard = n_ctx / 4;
        LOGI("Context full, shifting KV cache by %d tokens", n_discard);
        llama_memory_seq_rm(llama_get_memory(g_ctx), 0, 0, n_discard);
        llama_memory_seq_add(llama_get_memory(g_ctx), 0, n_discard, -1, -n_discard);
        g_n_past = std::max(0, g_n_past - n_discard);
    }

    // Decode prompt in batches
    {
        llama_batch batch = llama_batch_init(512, 0, 1);
        int i = 0;
        while (i < (int)tokens.size()) {
            batch.n_tokens = 0;
            while (batch.n_tokens < 512 && i < (int)tokens.size()) {
                batch.token[batch.n_tokens]     = tokens[i];
                batch.pos[batch.n_tokens]       = g_n_past + i;
                batch.n_seq_id[batch.n_tokens]  = 1;
                batch.seq_id[batch.n_tokens][0] = 0;
                batch.logits[batch.n_tokens]    = (i == (int)tokens.size() - 1) ? 1 : 0;
                batch.n_tokens++;
                i++;
            }
            if (llama_decode(g_ctx, batch) != 0) {
                llama_batch_free(batch);
                [NSException raise:@"LlamaDecodeError" format:@"llama_decode failed during prompt evaluation"];
            }
        }
        g_n_past += (int)tokens.size();
        llama_batch_free(batch);
    }

    // Build sampler chain — mirrors jni_wrapper.cpp exactly
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    g_sampler = llama_sampler_chain_init(chain_params);

    uint32_t seed_val = (seed < 0) ? (uint32_t)time(nullptr) : (uint32_t)seed;

    // Penalties first (applied to logits before sampling)
    if (repeatPenalty != 1.0 || frequencyPenalty != 0.0 || presencePenalty != 0.0) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_penalties(
            llama_vocab_n_tokens(g_vocab),
            repeatLastN,
            (float)repeatPenalty,
            (float)frequencyPenalty,
            (float)presencePenalty
        ));
    }

    // Temperature
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp((float)temperature));

    if (mirostat == 1) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_mirostat(
            llama_vocab_n_tokens(g_vocab),
            seed_val,
            (float)mirostatTau,
            (float)mirostatEta,
            100
        ));
    } else if (mirostat == 2) {
        llama_sampler_chain_add(g_sampler, llama_sampler_init_mirostat_v2(
            seed_val,
            (float)mirostatTau,
            (float)mirostatEta
        ));
    } else {
        if (minP > 0.0 && minP < 1.0)
            llama_sampler_chain_add(g_sampler, llama_sampler_init_min_p((float)minP, 1));
        if (typicalP < 1.0)
            llama_sampler_chain_add(g_sampler, llama_sampler_init_typical((float)typicalP, 1));
        if (topK > 0)
            llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(topK));
        if (topP < 1.0)
            llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p((float)topP, 1));
    }

    // Final distribution sampler
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(seed_val));

    // Generation loop
    const llama_token eos_token = llama_vocab_eos(g_vocab);
    std::string partial_token;

    for (int i = 0; i < maxTokens && !g_stop_flag; i++) {
        llama_token token_id = llama_sampler_sample(g_sampler, g_ctx, -1);

        if (token_id == eos_token || llama_vocab_is_eog(g_vocab, token_id)) break;

        // Decode token to text
        char buf[256] = {0};
        int n = llama_token_to_piece(g_vocab, token_id, buf, sizeof(buf) - 1, 0, true);
        if (n < 0) n = 0;
        buf[n] = '\0';

        partial_token += std::string(buf, n);

        // Emit complete UTF-8 sequences only
        if (isValidUTF8(partial_token.c_str(), partial_token.size())) {
            NSString* tokenStr = [NSString stringWithUTF8String:partial_token.c_str()];
            if (tokenStr && tokenCallback && !g_stop_flag) tokenCallback(tokenStr);
            partial_token.clear();
        }

        // Advance KV cache
        llama_batch next_batch = llama_batch_init(1, 0, 1);
        next_batch.token[0]     = token_id;
        next_batch.pos[0]       = g_n_past;
        next_batch.n_seq_id[0]  = 1;
        next_batch.seq_id[0][0] = 0;
        next_batch.logits[0]    = 1;
        next_batch.n_tokens     = 1;
        if (llama_decode(g_ctx, next_batch) != 0) {
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
        g_n_past++;
    }

    // Emit remaining partial token
    if (!partial_token.empty() && !g_stop_flag) {
        std::string safe = sanitizeUTF8(partial_token.c_str(), partial_token.size());
        NSString* tokenStr = [NSString stringWithUTF8String:safe.c_str()];
        if (tokenStr && tokenCallback) tokenCallback(tokenStr);
    }
}

- (void)stop {
    g_stop_flag = true;
}

- (void)freeModel {
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_ctx)     { llama_free(g_ctx);              g_ctx     = nullptr; }
    if (g_model)   { llama_model_free(g_model);      g_model   = nullptr; }
    g_vocab  = nullptr;
    g_n_past = 0;
}

- (BOOL)isModelLoaded {
    return g_model != nullptr && g_ctx != nullptr;
}

- (int)tokensUsed {
    return g_n_past;
}

- (int)contextSize {
    if (!g_ctx) return 0;
    return (int)llama_n_ctx(g_ctx);
}

- (void)clearContext {
    if (g_ctx) {
        llama_memory_seq_rm(llama_get_memory(g_ctx), 0, g_system_prompt_length, -1);
        g_n_past = g_system_prompt_length;
        LOGI("Context cleared, keeping %d system prompt tokens", g_system_prompt_length);
    }
}

- (void)setSystemPromptLength:(int)length {
    g_system_prompt_length = length;
}

// MARK: Phone cluster master

+ (void)setClusterEndpoints:(NSArray<NSString*>*)endpoints
                    weights:(NSArray<NSNumber*>*)weights
                localWeight:(float)localWeight {
    std::lock_guard<std::mutex> lock(g_cluster_mutex);
    g_cluster_endpoints.clear();
    g_cluster_weights.clear();
    g_cluster_local_weight = localWeight;
    for (NSUInteger i = 0; i < endpoints.count; i++) {
        g_cluster_endpoints.push_back(endpoints[i].UTF8String);
        g_cluster_weights.push_back(i < weights.count ? weights[i].floatValue : 1.0f);
    }
    LOGI("Cluster set: %zu worker(s)", g_cluster_endpoints.size());
}

+ (NSArray<NSDictionary<NSString*, id>*>*)lastLoadBuffers {
    std::lock_guard<std::mutex> lock(g_load_log_mutex);
    NSMutableArray<NSDictionary<NSString*, id>*>* out = [NSMutableArray array];
    for (const auto& buffer : g_load_buffers) {
        NSString* name = [NSString stringWithUTF8String:buffer.first.c_str()];
        if (name) [out addObject:@{@"name": name, @"mb": @(buffer.second)}];
    }
    return out;
}

+ (BOOL)probeEndpoint:(NSString*)endpoint timeoutMs:(int)timeoutMs {
    const std::string text = endpoint.UTF8String ?: "";
    const size_t colon = text.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= text.size()) return NO;
    const std::string host = text.substr(0, colon);
    const std::string port = text.substr(colon + 1);

    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* list = nullptr;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &list) != 0 || !list) return NO;

    BOOL reachable = NO;
    for (addrinfo* ai = list; ai && !reachable; ai = ai->ai_next) {
        const int fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
            reachable = YES;
        } else if (errno == EINPROGRESS) {
            pollfd waiting = {fd, POLLOUT, 0};
            if (poll(&waiting, 1, timeoutMs) == 1) {
                int error = 0;
                socklen_t size = sizeof(error);
                reachable = getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &size) == 0 && error == 0;
            }
        }
        close(fd);
    }
    freeaddrinfo(list);
    return reachable;
}

@end
