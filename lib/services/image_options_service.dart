import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

import 'hive_service.dart';

/// A LoRA file kept in the app's storage.
class LoraEntry {
  final String path;
  final String name;
  bool use;
  double weight;

  LoraEntry({required this.path, required this.name, this.use = true, this.weight = 0.8});

  Map<String, dynamic> toMap() => {'path': path, 'name': name, 'use': use, 'weight': weight};

  factory LoraEntry.fromMap(Map m) => LoraEntry(
        path: m['path'] as String? ?? '',
        name: m['name'] as String? ?? '',
        use: m['use'] as bool? ?? true,
        weight: (m['weight'] as num?)?.toDouble() ?? 0.8,
      );
}

/// Options for images made in the chat: a negative prompt ("Avoid"), a
/// starting image (image-to-image) with how much to change it, and LoRAs.
class ImageOptionsService extends GetxService {
  static const _keyNegative = 'image_opt_negative';
  static const _keyStrength = 'image_opt_strength';
  static const _keyLoras = 'image_opt_loras';
  static const loraExtensions = {'safetensors', 'gguf', 'pt', 'ckpt'};

  final HiveService _hive = Get.find<HiveService>();

  final negative = ''.obs;

  /// How much of the starting image to change, 0.1 (a little) to 1 (a lot).
  final strength = 0.55.obs;

  /// The starting image (PNG/JPEG bytes), or null for text-to-image.
  final initImage = Rxn<Uint8List>();
  final initLabel = ''.obs;

  final loras = <LoraEntry>[].obs;

  @override
  void onInit() {
    super.onInit();
    negative.value = _hive.getSetting<String>(_keyNegative) ?? '';
    strength.value = (_hive.getSetting<num>(_keyStrength) ?? 0.55).toDouble();
    final saved = _hive.getSetting<List>(_keyLoras) ?? const [];
    loras.assignAll(saved
        .whereType<Map>()
        .map(LoraEntry.fromMap)
        .where((l) => l.path.isNotEmpty && File(l.path).existsSync()));
  }

  List<LoraEntry> get activeLoras =>
      loras.where((l) => l.use && File(l.path).existsSync()).toList();

  /// True when any option changes how the next image is made.
  bool get inUse =>
      initImage.value != null || activeLoras.isNotEmpty || negative.value.trim().isNotEmpty;

  Future<void> save() async {
    await _hive.setSetting(_keyNegative, negative.value);
    await _hive.setSetting(_keyStrength, strength.value);
    await _hive.setSetting(_keyLoras, [for (final l in loras) l.toMap()]);
    loras.refresh();
  }

  // ── Starting image ────────────────────────────────────────────────────

  void setInitImage(Uint8List bytes, String label) {
    initImage.value = bytes;
    initLabel.value = label;
  }

  void clearInitImage() {
    initImage.value = null;
    initLabel.value = '';
  }

  /// Picks the starting image from photos or the camera. Returns false if
  /// the person cancelled.
  Future<bool> pickInitImage(ImageSource source) async {
    final file = await ImagePicker().pickImage(
      source: source,
      maxWidth: 1024,
      maxHeight: 1024,
      imageQuality: 95,
    );
    if (file == null) return false;
    setInitImage(await file.readAsBytes(), file.name);
    return true;
  }

  // ── LoRAs ─────────────────────────────────────────────────────────────

  Future<Directory> _loraDir() async {
    final base = await getApplicationSupportDirectory();
    final dir = Directory('${base.path}/loras');
    await dir.create(recursive: true);
    return dir;
  }

  /// Lets the person pick a LoRA file and copies it into the app. Returns
  /// an error message, or null on success (or when cancelled).
  Future<String?> addLora() async {
    // Any type: phones don't know the .safetensors type, so a filter would
    // grey the files out.
    final result = await FilePicker.pickFiles(type: FileType.any);
    final picked = result?.files.single;
    if (picked == null || picked.path == null) return null;
    final name = picked.name;
    final ext = name.contains('.') ? name.split('.').last.toLowerCase() : '';
    if (!loraExtensions.contains(ext)) {
      return '$name isn\'t a LoRA file. LoRAs end in .safetensors.';
    }
    final dir = await _loraDir();
    var target = File('${dir.path}/$name');
    var n = 2;
    while (await target.exists()) {
      target = File('${dir.path}/${name.substring(0, name.length - ext.length - 1)}_$n.$ext');
      n++;
    }
    await File(picked.path!).copy(target.path);
    loras.add(LoraEntry(path: target.path, name: name));
    await save();
    return null;
  }

  Future<void> removeLora(LoraEntry lora) async {
    loras.remove(lora);
    try {
      await File(lora.path).delete();
    } catch (_) {}
    await save();
  }
}
