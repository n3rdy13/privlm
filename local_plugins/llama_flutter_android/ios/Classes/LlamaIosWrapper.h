#ifndef LlamaIosWrapper_h
#define LlamaIosWrapper_h

#import <Foundation/Foundation.h>

typedef void (^LlamaTokenCallback)(NSString* _Nonnull token);
typedef void (^LlamaProgressCallback)(double progress);

NS_ASSUME_NONNULL_BEGIN

@interface LlamaIosWrapper : NSObject

/// Load a GGUF model file. Throws NSException on failure.
- (void)loadModelAtPath:(NSString*)path
               nThreads:(int)nThreads
            contextSize:(int)contextSize
             nGpuLayers:(int)nGpuLayers
       progressCallback:(LlamaProgressCallback)progressCallback;

/// Start text generation. Calls tokenCallback for each token synchronously (runs on caller thread).
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
            tokenCallback:(LlamaTokenCallback)tokenCallback;

/// Signal the current generation to stop.
- (void)stop;

/// Free model and context resources.
- (void)freeModel;

/// Returns YES if a model is currently loaded.
- (BOOL)isModelLoaded;

/// Returns number of tokens currently in KV cache.
- (int)tokensUsed;

/// Returns context window size.
- (int)contextSize;

/// Resets the KV cache (clears conversation history, keeps model loaded).
- (void)clearContext;

/// Sets system prompt length for smart context management.
- (void)setSystemPromptLength:(int)length;

// MARK: Phone cluster master

/// Sets the workers ("host:port") the next model load spreads its layers
/// across. Each worker gets a share of the layers by its weight (e.g. free
/// RAM); this device keeps a share by `localWeight`. An empty list turns
/// cluster mode off. Every endpoint must already be reachable (see
/// `probeEndpoint:timeoutMs:`): llama.cpp aborts the app if it can't connect.
+ (void)setClusterEndpoints:(NSArray<NSString*>*)endpoints
                    weights:(NSArray<NSNumber*>*)weights
                localWeight:(float)localWeight;

/// Where the last cluster load put its weights: `name` (llama.cpp buffer
/// name, "RPC0[host:port]" for a worker) and `mb` per buffer.
+ (NSArray<NSDictionary<NSString*, id>*>*)lastLoadBuffers;

/// Whether something accepts a TCP connection on `endpoint` ("host:port")
/// within `timeoutMs`. Blocks; call off the main thread.
+ (BOOL)probeEndpoint:(NSString*)endpoint timeoutMs:(int)timeoutMs;

@end

NS_ASSUME_NONNULL_END

#endif /* LlamaIosWrapper_h */
