import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:get/get.dart';
import 'package:path_provider/path_provider.dart';
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
  static const int maxBodyBytes = 16 * 1024 * 1024; // img2img carries a picture
  static const int maxLoraBytes = 2 * 1024 * 1024 * 1024;
  static const _loraExtensions = {'safetensors', 'gguf', 'pt', 'ckpt'};
  static final _sha256 = RegExp(r'^[0-9a-f]{64}$');

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
        'img2img': true,
        'lora': true,
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
      if (request.method == 'POST' && path == '/sdapi/v1/img2img') {
        return await _txt2img(request, img2img: true);
      }
      if (request.method == 'GET' && path == '/loras') {
        return await _json(request, {'loras': (await _cachedLoras()).keys.toList()});
      }
      if (request.method == 'PUT' && path.startsWith('/loras/')) {
        return await _receiveLora(request, path.substring('/loras/'.length));
      }
      return await _json(request, {'error': 'Not found'}, HttpStatus.notFound);
    } catch (e) {
      try {
        await _json(request, {'error': '$e'}, HttpStatus.internalServerError);
      } catch (_) {}
    }
  }

  Future<void> _txt2img(HttpRequest request, {bool img2img = false}) async {
    if (!_images.isModelLoaded.value) {
      return await _json(request, {'error': 'No image model is loaded on this phone.'},
          HttpStatus.serviceUnavailable);
    }
    if (busy.value || _images.isGenerating.value) {
      return await _json(request, {'error': 'This phone is already making an image.'},
          HttpStatus.conflict);
    }
    if (request.contentLength > maxBodyBytes) {
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
    // LoRAs the laptop sent earlier, by their SHA-256.
    final loraPaths = <String>[];
    final loraWeights = <double>[];
    final wanted = body['lora'];
    if (wanted is List && wanted.isNotEmpty) {
      final cached = await _cachedLoras();
      final missing = <String>[];
      for (final item in wanted.whereType<Map>()) {
        final sha = (item['sha256'] as String? ?? '').toLowerCase();
        final file = cached[sha];
        if (file == null) {
          missing.add(sha);
        } else {
          loraPaths.add(file.path);
          loraWeights.add((item['multiplier'] as num?)?.toDouble() ?? 1.0);
        }
      }
      if (missing.isNotEmpty) {
        return await _json(request, {'error': 'Missing LoRA', 'missing': missing},
            HttpStatus.preconditionFailed);
      }
    }
    Uint8List? initImage;
    if (img2img) {
      final images = body['init_images'];
      try {
        var data = (images as List).first as String;
        final comma = data.indexOf(',');
        if (data.startsWith('data:') && comma > 0) data = data.substring(comma + 1);
        initImage = base64Decode(data);
      } catch (_) {
        return await _json(request, {'error': 'init_images must hold one base64 image'},
            HttpStatus.badRequest);
      }
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
        initImage: initImage,
        strength: (body['denoising_strength'] as num?)?.toDouble() ?? 0.75,
        loraPaths: loraPaths,
        loraWeights: loraWeights,
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

  Future<Directory> _loraDir() async {
    final base = await getApplicationSupportDirectory();
    final dir = Directory('${base.path}/image_worker_loras');
    await dir.create(recursive: true);
    return dir;
  }

  /// LoRA files received from laptops, by SHA-256 (they're named `<sha>.<ext>`).
  Future<Map<String, File>> _cachedLoras() async {
    final out = <String, File>{};
    await for (final entity in (await _loraDir()).list()) {
      if (entity is! File) continue;
      final name = entity.uri.pathSegments.last;
      final dot = name.indexOf('.');
      final sha = dot > 0 ? name.substring(0, dot) : name;
      if (_sha256.hasMatch(sha) && !name.endsWith('.part')) out[sha] = entity;
    }
    return out;
  }

  /// Stores a LoRA file a laptop sends, keeping it only if its SHA-256
  /// matches the name it was sent under.
  Future<void> _receiveLora(HttpRequest request, String name) async {
    final sha = name.toLowerCase();
    final ext = (request.uri.queryParameters['ext'] ?? 'safetensors').toLowerCase();
    if (!_sha256.hasMatch(sha) || !_loraExtensions.contains(ext)) {
      return await _json(request, {'error': 'Bad LoRA name'}, HttpStatus.badRequest);
    }
    if (request.contentLength > maxLoraBytes) {
      return await _json(request, {'error': 'LoRA too large'}, HttpStatus.requestEntityTooLarge);
    }
    final dir = await _loraDir();
    final part = File('${dir.path}/$sha.$ext.part');
    final sink = part.openWrite();
    try {
      await sink.addStream(request);
    } finally {
      await sink.close();
    }
    final digest = await sha256.bind(part.openRead()).first;
    if (digest.toString() != sha) {
      await part.delete();
      return await _json(request, {'error': 'The file arrived damaged; send it again.'},
          HttpStatus.badRequest);
    }
    await part.rename('${dir.path}/$sha.$ext');
    status.value = 'Received a LoRA';
    return await _json(request, {'ok': true});
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
