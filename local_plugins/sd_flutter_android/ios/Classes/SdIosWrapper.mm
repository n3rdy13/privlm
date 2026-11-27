#import "SdIosWrapper.h"
#include "stable-diffusion.h"
#include <string>

static sd_ctx_t* g_sd_ctx = nullptr;
static SdProgressBlock g_progress_block = nil;
static NSMutableString *g_last_logs = nil;

static void ios_sd_progress_cb(int step, int steps, float time, void* data) {
    if (g_progress_block) {
        g_progress_block(step, steps);
    }
}

static void ios_sd_log_cb(sd_log_level_t level, const char* text, void* data) {
    NSString *line = [NSString stringWithUTF8String:text];
    NSLog(@"[SD-CPP] %@", line);
    if (g_last_logs) {
        [g_last_logs appendString:line];
        [g_last_logs appendString:@"\n"];
    }
}

@implementation SdIosWrapper

- (NSString *)loadModel:(NSString *)path {
    if (g_sd_ctx) {
        free_sd_ctx(g_sd_ctx);
        g_sd_ctx = nullptr;
    }

    g_last_logs = [NSMutableString string];

    NSFileManager *fm = [NSFileManager defaultManager];
    if (![fm fileExistsAtPath:path]) {
        NSString *err = [NSString stringWithFormat:@"Model file does NOT exist at path: %@", path];
        NSLog(@"[SD-iOS] ERROR: %@", err);
        return err;
    }

    NSError *err = nil;
    NSDictionary *attrs = [fm attributesOfItemAtPath:path error:&err];
    if (attrs) {
        NSLog(@"[SD-iOS] Model file exists, size: %llu bytes", [attrs fileSize]);
    } else {
        NSLog(@"[SD-iOS] WARNING: Cannot read file attributes: %@", err);
    }

    sd_set_log_callback(ios_sd_log_cb, NULL);

    sd_ctx_params_t params;
    sd_ctx_params_init(&params);
    params.model_path = [path UTF8String];
    params.n_threads = sd_get_num_physical_cores();
    params.free_params_immediately = false;  // Keep buffers alive between generations

    NSLog(@"[SD-iOS] Calling new_sd_ctx...");
    g_sd_ctx = new_sd_ctx(&params);

    if (!g_sd_ctx) {
        NSString *errorMsg = [NSString stringWithFormat:@"new_sd_ctx returned nullptr. Logs:\n%@", g_last_logs];
        NSLog(@"[SD-iOS] ERROR: %@", errorMsg);
        return errorMsg;
    }

    NSLog(@"[SD-iOS] SUCCESS: new_sd_ctx returned valid context");
    return @"true";
}

- (NSData *)generateImage:(NSString *)prompt steps:(int)steps {
    NSDictionary *out = [self generateImage:prompt steps:steps width:512 height:512 seed:-1
                             negativePrompt:@"" cfgScale:-1 sampler:@""];
    return out[@"data"];
}

- (NSDictionary *)generateImage:(NSString *)prompt
                          steps:(int)steps
                          width:(int)width
                         height:(int)height
                           seed:(int64_t)seed
                 negativePrompt:(NSString *)negativePrompt
                       cfgScale:(float)cfgScale
                        sampler:(NSString *)sampler {
    if (!g_sd_ctx) return nil;

    // Keep the prompt strings alive for the whole call.
    std::string prompt_s = prompt ? [prompt UTF8String] : "";
    std::string negative_s = negativePrompt ? [negativePrompt UTF8String] : "";

    sd_img_gen_params_t params;
    sd_img_gen_params_init(&params);
    params.prompt = prompt_s.c_str();
    params.negative_prompt = negative_s.c_str();
    params.sample_params.sample_steps = steps;
    params.width = width > 0 ? width : 512;
    params.height = height > 0 ? height : 512;
    if (seed < 0) {
        // Pick it here so the caller learns which seed made the image.
        seed = (int64_t)arc4random_uniform(0x7fffffff);
    }
    params.seed = seed;
    params.sample_params.sample_method = EULER_A_SAMPLE_METHOD;
    if (sampler.length > 0) {
        enum sample_method_t method = str_to_sample_method([sampler UTF8String]);
        if (method != SAMPLE_METHOD_COUNT) params.sample_params.sample_method = method;
    }
    if (cfgScale > 0) params.sample_params.guidance.txt_cfg = cfgScale;

    g_progress_block = self.onProgress;
    sd_set_progress_callback(ios_sd_progress_cb, NULL);

    sd_image_t* result = generate_image(g_sd_ctx, &params);

    g_progress_block = nil;
    sd_set_progress_callback(NULL, NULL);

    if (!result) return nil;

    size_t size = result->width * result->height * result->channel;
    NSData *data = [NSData dataWithBytes:result->data length:size];
    NSDictionary *out = @{@"data": data, @"width": @(result->width), @"height": @(result->height),
                          @"seed": @(seed)};

    free(result->data);
    free(result);

    return out;
}

- (void)unloadModel {
    if (g_sd_ctx) {
        free_sd_ctx(g_sd_ctx);
        g_sd_ctx = nullptr;
    }
}

@end
