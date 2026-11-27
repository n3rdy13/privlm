#import <Foundation/Foundation.h>

typedef void (^SdProgressBlock)(int step, int totalSteps);

@interface SdIosWrapper : NSObject

@property (nonatomic, copy) SdProgressBlock onProgress;

/// Returns "true" on success, or an error message string on failure.
- (NSString *)loadModel:(NSString *)path;
- (NSData *)generateImage:(NSString *)prompt steps:(int)steps;
/// Full control, for the phone cluster's image worker. width/height <= 0
/// mean 512, seed < 0 means random, an empty or unknown sampler means
/// Euler a. Returns @{@"data": RGB bytes, @"width": @w, @"height": @h,
/// @"seed": @seed} or nil. With initImage (width x height x 3 RGB bytes)
/// it's image-to-image, changing it by strength (0..1).
- (NSDictionary *)generateImage:(NSString *)prompt
                          steps:(int)steps
                          width:(int)width
                         height:(int)height
                           seed:(int64_t)seed
                 negativePrompt:(NSString *)negativePrompt
                       cfgScale:(float)cfgScale
                        sampler:(NSString *)sampler
                      initImage:(NSData *)initImage
                       strength:(float)strength
                      loraPaths:(NSArray<NSString *> *)loraPaths
                    loraWeights:(NSArray<NSNumber *> *)loraWeights;
- (void)unloadModel;

@end
