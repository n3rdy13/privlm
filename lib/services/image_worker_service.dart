import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:get/get.dart';
import 'package:llama_flutter_android/llama_flutter_android.dart';

import 'app_log_service.dart';
import 'cluster_service.dart';
import 'local_image_service.dart';

/// Image worker: lets a laptop running PrivateLM Laptop send this phone
/// image jobs over the Wi-Fi. The phone makes each image with the image
/// model loaded in the app and sends it back, so several devices make
/// images at the same time.
///
/// Speaks the A1111-style `POST /sdapi/v1/txt2img` that the laptop's own
/// stable-diffusion.cpp server speaks, plus `GET /status`. Like the chat
/// worker, it has no password: anyone on this Wi-Fi can send it jobs while
/// it's on.
class ImageWorkerService extends GetxService {
  static const int port = 50061;
  static const String protocol = 'img1';
  static const int maxBodyBytes = 64 * 1024;

  final running = false.obs;
  final busy = false.obs;
  final served = 0.obs;
  final status = 'Off'.obs;

  /// The device whose job is running (its IP address), if any.
  final client = RxnString();

  HttpServer? _server;
  LocalImageService get _images => Get.find<LocalImageService>();

  String get _deviceName {
    if (Get.isRegistered<ClusterService>()) {
      final name = Get.find<ClusterService>().deviceName;
      if (name.isNotEmpty) return name;
    }
    return Platform.isIOS ? 'iPhone' : 'Android phone';
  }

  /// Throws a [StateError] with a readable message if it can't start.
  Future<void> start() async {
    if (_server != null) return;
    if (!_images.isModelLoaded.value) {
      throw StateError('Load an image model first (Models, then an image '
          'model), then turn this on.');
    }
    try {
      _server = await HttpServer.bind(InternetAddress.anyIPv4, port);
    } on SocketException catch (e) {
      throw StateError('Couldn\'t listen on port $port: ${e.message}');
    }
    unawaited(_serve(_server!));
    running.value = true;
    status.value = 'Waiting for jobs';
    _setKeepAwake(true);
    try {
      await LlamaCluster.advertiseImageWorker(port, {
        'model': _deviceName,
        'img': _images.loadedModelName.value,
        'v': protocol,
      });
    } catch (e) {
      Get.find<AppLogService>()
          .error('Image worker: couldn\'t announce on the Wi-Fi', details: e);
    }
  }

  Future<void> stop() async {
    final server = _server;
    _server = null;
    running.value = false;
    busy.value = false;
    client.value = null;
    status.value = 'Off';
    _setKeepAwake(false);
    try {
      await LlamaCluster.stopImageWorkerAdvert();
    } catch (_) {}
    await server?.close(force: true);
  }

  void _setKeepAwake(bool on) {
    // iOS suspends the app (and its server) when the screen locks, and
    // Android throttles it: keep the screen on while taking jobs.
    if (Get.isRegistered<ClusterService>()) {
      Get.find<ClusterService>().imageWorker.value = on;
    }
  }

  Future<void> _serve(HttpServer server) async {
    await for (final request in server) {
      unawaited(_handle(request));
    }
  }

  Map<String, Object?> _statusJson() => {
        'name': _deviceName,
        'model': _images.loadedModelName.value,
        'loaded': _images.isModelLoaded.value,
        'busy': busy.value || _images.isGenerating.value,
        'served': served.value,
        'platform': Platform.isIOS ? 'ios' : 'android',
        'maxPixels': 512 * 768,
        'v': protocol,
      };

  Future<void> _handle(HttpRequest request) async {
    try {
      final path = request.uri.path;
      if (request.method == 'GET' && (path == '/status' || path == '/')) {
        return await _json(request, _statusJson());
      }
      if (request.method == 'POST' && path == '/sdapi/v1/txt2img') {
        return await _txt2img(request);
      }
      return await _json(request, {'error': 'Not found'}, HttpStatus.notFound);
    } catch (e) {
      try {
        await _json(request, {'error': '$e'}, HttpStatus.internalServerError);
      } catch (_) {}
    }
  }

  Future<void> _txt2img(HttpRequest request) async {
    if (!_images.isModelLoaded.value) {
      return await _json(request, {'error': 'No image model is loaded on this phone.'},
          HttpStatus.serviceUnavailable);
    }
    if (busy.value || _images.isGenerating.value) {
      return await _json(request, {'error': 'This phone is already making an image.'},
          HttpStatus.conflict);
    }
    if ((request.contentLength) > maxBodyBytes) {
      return await _json(request, {'error': 'Request too large'},
          HttpStatus.requestEntityTooLarge);
    }
    final Map<String, dynamic> body;
    try {
      final text = await utf8.decoder.bind(request).join();
      body = jsonDecode(text) as Map<String, dynamic>;
    } catch (_) {
      return await _json(request, {'error': 'Body must be JSON'}, HttpStatus.badRequest);
    }
    final prompt = (body['prompt'] as String? ?? '').trim();
    if (prompt.isEmpty) {
      return await _json(request, {'error': 'prompt is required'}, HttpStatus.badRequest);
    }

    busy.value = true;
    client.value = request.connectionInfo?.remoteAddress.address;
    status.value = 'Making an image for ${client.value ?? 'a laptop'}';
    final started = DateTime.now();
    try {
      final image = await _images.generateForWorker(
        prompt: prompt,
        negativePrompt: body['negative_prompt'] as String? ?? '',
        width: (body['width'] as num?)?.toInt() ?? 512,
        height: (body['height'] as num?)?.toInt() ?? 512,
        steps: (body['steps'] as num?)?.toInt() ?? 20,
        seed: (body['seed'] as num?)?.toInt() ?? -1,
        cfgScale: (body['cfg_scale'] as num?)?.toDouble() ?? 7.0,
        sampler: body['sampler_name'] as String? ?? 'Euler a',
        onProgress: (step, total) =>
            status.value = 'Making an image for ${client.value ?? 'a laptop'}: '
                'step $step of $total',
      );
      if (image == null) {
        return await _json(
            request,
            {'error': _images.lastFailureReason ?? 'The image couldn\'t be made.'},
            HttpStatus.internalServerError);
      }
      served.value++;
      final seconds = DateTime.now().difference(started).inMilliseconds / 1000;
      return await _json(request, {
        'images': [base64Encode(image.png)],
        'info': {
          'seed': image.seed,
          'width': image.width,
          'height': image.height,
          'seconds': seconds,
          'device': _deviceName,
          'model': _images.loadedModelName.value,
        },
      });
    } finally {
      busy.value = false;
      client.value = null;
      if (running.value) status.value = 'Waiting for jobs · ${served.value} made';
    }
  }

  Future<void> _json(HttpRequest request, Object body,
      [int code = HttpStatus.ok]) async {
    request.response
      ..statusCode = code
      ..headers.contentType = ContentType.json
      ..write(jsonEncode(body));
    await request.response.close();
  }

  @override
  void onClose() {
    unawaited(stop());
    super.onClose();
  }
}
