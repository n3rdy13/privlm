import 'package:flutter/services.dart';

class SdFlutterAndroid {
  static const MethodChannel _channel = MethodChannel('sd_flutter_android');

  Future<String?> getPlatformVersion() {
    return _channel.invokeMethod<String>('getPlatformVersion');
  }

  static Future<String> detectGpuVendor() async {
    final result = await _channel.invokeMethod<String>('detectGpuVendor');
    return result ?? 'unknown';
  }

  static Future<int> getDeviceMemory() async {
    final result = await _channel.invokeMethod<int>('getDeviceMemory');
    return result ?? 4096; // fallback: assume 4GB
  }

  static Future<int> getAvailableMemory() async {
    final result = await _channel.invokeMethod<int>('getAvailableMemory');
    return result ?? 0;
  }

  static Future<dynamic> initModelRaw(String path, {bool useGpu = true}) async {
    final result = await _channel.invokeMethod<dynamic>('initModel', {
      'path': path,
      'useGpu': useGpu,
    });
    return result;
  }

  static Future<bool> initModel(String path, {bool useGpu = true}) async {
    final result = await initModelRaw(path, useGpu: useGpu);
    if (result is bool) {
      return result;
    }
    if (result is String && result == 'true') {
      return true;
    }
    return false;
  }

  static Function(int step, int total)? _onProgress;

  static void _ensureInitialized() {
    _channel.setMethodCallHandler((call) async {
      if (call.method == 'onProgress') {
        final step = call.arguments['step'] as int;
        final total = call.arguments['total'] as int;
        _onProgress?.call(step, total);
      }
    });
  }

  static Future<Uint8List?> generateImage(
    String prompt, {
    int steps = 20,
    Function(int step, int total)? onProgress,
  }) async {
    _ensureInitialized();
    _onProgress = onProgress;

    final bytes = await _channel.invokeMethod<Uint8List>('generateImage', {
      'prompt': prompt,
      'steps': steps,
    });
    return bytes;
  }

  /// iOS: generates with full control (the phone cluster's image worker).
  /// [seed] < 0 picks a random one; an empty [sampler] means Euler a.
  /// Returns `{rgb: Uint8List, width: int, height: int, seed: int}`.
  static Future<Map<String, Object>?> generateImageEx(
    String prompt, {
    int steps = 20,
    int width = 512,
    int height = 512,
    int seed = -1,
    String negativePrompt = '',
    double cfgScale = -1,
    String sampler = '',

    /// Image-to-image: [width] x [height] RGB bytes, changed by [strength].
    Uint8List? initImage,
    double strength = 0.75,
    Function(int step, int total)? onProgress,
  }) async {
    _ensureInitialized();
    _onProgress = onProgress;

    final out = await _channel.invokeMethod<Map<Object?, Object?>>('generateImageEx', {
      if (initImage != null) 'initImage': initImage,
      'strength': strength,
      'prompt': prompt,
      'steps': steps,
      'width': width,
      'height': height,
      'seed': seed,
      'negativePrompt': negativePrompt,
      'cfgScale': cfgScale,
      'sampler': sampler,
    });
    final rgb = out?['rgb'];
    if (out == null || rgb is! Uint8List) return null;
    return {
      'rgb': rgb,
      'width': (out['width'] as num?)?.toInt() ?? width,
      'height': (out['height'] as num?)?.toInt() ?? height,
      'seed': (out['seed'] as num?)?.toInt() ?? seed,
    };
  }

  static Future<void> unloadModel() async {
    await _channel.invokeMethod('unloadModel');
  }
}
