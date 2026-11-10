import 'dart:async';
import 'dart:convert';
import 'dart:ffi' show Abi;
import 'dart:io';

import 'package:flutter/foundation.dart' show kIsWeb, debugPrint;
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:get/get.dart';
import 'package:http/http.dart' as http;
import 'package:package_info_plus/package_info_plus.dart';
import 'package:path_provider/path_provider.dart';

import '../core/constants.dart';
import '../core/ui.dart';
import '../widgets/update_sheet.dart';
import 'hive_service.dart';

/// A published GitHub release and the APK chosen for this device.
class AppRelease {
  final String version;
  final String name;
  final String notes;
  final String pageUrl;
  final DateTime? publishedAt;
  final String? apkName;
  final String? apkUrl;
  final int apkSize;

  const AppRelease({
    required this.version,
    required this.name,
    required this.notes,
    required this.pageUrl,
    this.publishedAt,
    this.apkName,
    this.apkUrl,
    this.apkSize = 0,
  });
}

enum UpdateState { idle, checking, downloading, installing, error }

/// Checks GitHub Releases for a newer APK, downloads it and hands it to the
/// Android package installer. Android only.
class UpdateService extends GetxService with WidgetsBindingObserver {
  static const _channel = MethodChannel('com.aichat.ai_chat/model_import');
  static const _promptInterval = Duration(hours: 24);

  final HiveService _hive = Get.find<HiveService>();

  final currentVersion = ''.obs;
  final availableUpdate = Rxn<AppRelease>();
  final state = UpdateState.idle.obs;
  final progress = 0.0.obs;
  final downloadedBytes = 0.obs;
  final totalBytes = 0.obs;
  final lastError = ''.obs;
  final needsInstallPermission = false.obs;
  final autoCheck = true.obs;
  final lastCheck = Rxn<DateTime>();

  int? _downloadId;
  String? _pendingInstallPath;
  bool _sheetOpen = false;

  static bool get isSupported => !kIsWeb && Platform.isAndroid;

  @override
  void onInit() {
    super.onInit();
    autoCheck.value = _hive.getSetting<bool>(AppConstants.keyAutoUpdateCheck,
            defaultValue: true) ??
        true;
    final last = _hive.getSetting<int>(AppConstants.keyLastUpdateCheck);
    if (last != null) {
      lastCheck.value = DateTime.fromMillisecondsSinceEpoch(last);
    }
    if (!isSupported) return;
    WidgetsBinding.instance.addObserver(this);
    unawaited(_start());
  }

  @override
  void onClose() {
    if (isSupported) WidgetsBinding.instance.removeObserver(this);
    super.onClose();
  }

  Future<void> _start() async {
    try {
      currentVersion.value = (await PackageInfo.fromPlatform()).version;
    } catch (_) {}
    unawaited(_cleanupOldApks());
    if (!autoCheck.value) return;
    await Future.delayed(const Duration(seconds: 5));
    // Check quietly on every launch so Settings shows the right state, but
    // pop the sheet up at most once a day.
    final release = await checkForUpdate();
    if (release == null) return;
    final skipped =
        _hive.getSetting<String>(AppConstants.keySkippedUpdateVersion);
    final lastPrompt =
        _hive.getSetting<int>(AppConstants.keyLastUpdatePrompt) ?? 0;
    final due = DateTime.now()
            .difference(DateTime.fromMillisecondsSinceEpoch(lastPrompt)) >
        _promptInterval;
    if (release.version != skipped && due) {
      await _hive.setSetting(AppConstants.keyLastUpdatePrompt,
          DateTime.now().millisecondsSinceEpoch);
      showUpdateSheet();
    }
  }

  Future<void> setAutoCheck(bool value) async {
    autoCheck.value = value;
    await _hive.setSetting(AppConstants.keyAutoUpdateCheck, value);
  }

  Future<void> skipVersion(String version) =>
      _hive.setSetting(AppConstants.keySkippedUpdateVersion, version);

  /// Fetches the latest release. Returns it when it is newer than the
  /// installed version. A manual check reports the result with a toast.
  Future<AppRelease?> checkForUpdate({bool manual = false}) async {
    if (!isSupported || state.value == UpdateState.checking) return null;
    if (state.value == UpdateState.downloading) return availableUpdate.value;
    state.value = UpdateState.checking;
    try {
      if (currentVersion.value.isEmpty) {
        currentVersion.value = (await PackageInfo.fromPlatform()).version;
      }
      final res = await http.get(
        Uri.parse(
            'https://api.github.com/repos/${AppConstants.githubRepo}/releases/latest'),
        headers: const {'Accept': 'application/vnd.github+json'},
      ).timeout(const Duration(seconds: 15));
      if (res.statusCode != 200) {
        throw HttpException('GitHub returned ${res.statusCode}');
      }
      final release =
          _parseRelease(jsonDecode(res.body) as Map<String, dynamic>);
      final now = DateTime.now();
      lastCheck.value = now;
      await _hive.setSetting(
          AppConstants.keyLastUpdateCheck, now.millisecondsSinceEpoch);

      final newer = compareVersions(release.version, currentVersion.value) > 0;
      availableUpdate.value = newer ? release : null;
      state.value = UpdateState.idle;
      if (manual && !newer) {
        showToast('You\'re up to date',
            'PrivateLM ${currentVersion.value} is the latest version.',
            tone: ToastTone.success);
      }
      return availableUpdate.value;
    } catch (e) {
      debugPrint('Update check failed: $e');
      state.value = UpdateState.idle;
      if (manual) {
        showToast('Couldn\'t check for updates',
            'Check your connection and try again.',
            tone: ToastTone.error);
      }
      return null;
    }
  }

  AppRelease _parseRelease(Map<String, dynamic> json) {
    final tag = (json['tag_name'] as String? ?? '').trim();
    final assets = (json['assets'] as List? ?? [])
        .cast<Map<String, dynamic>>()
        .where((a) => (a['name'] as String? ?? '').endsWith('.apk'))
        .toList();
    Map<String, dynamic>? pick(String abi) =>
        assets.firstWhereOrNull((a) => (a['name'] as String).contains(abi));
    final asset = pick(_abiTag()) ?? pick('universal');
    return AppRelease(
      version: _normalize(tag),
      name: (json['name'] as String?)?.trim().isNotEmpty == true
          ? json['name'] as String
          : 'PrivateLM $tag',
      notes: json['body'] as String? ?? '',
      pageUrl: json['html_url'] as String? ?? '',
      publishedAt: DateTime.tryParse(json['published_at'] as String? ?? ''),
      apkName: asset?['name'] as String?,
      apkUrl: asset?['browser_download_url'] as String?,
      apkSize: (asset?['size'] as num?)?.toInt() ?? 0,
    );
  }

  static String _abiTag() {
    switch (Abi.current()) {
      case Abi.androidArm64:
        return 'arm64-v8a';
      case Abi.androidArm:
        return 'armeabi-v7a';
      case Abi.androidX64:
        return 'x86_64';
      default:
        return 'universal';
    }
  }

  static String _normalize(String v) {
    var s = v.trim();
    if (s.startsWith('v') || s.startsWith('V')) s = s.substring(1);
    final plus = s.indexOf('+');
    if (plus >= 0) s = s.substring(0, plus);
    return s;
  }

  /// Numeric comparison of dotted versions ("1.0.10" > "1.0.9").
  static int compareVersions(String a, String b) {
    List<int> parts(String v) => _normalize(v)
        .split(RegExp(r'[.\-]'))
        .map((p) => int.tryParse(p) ?? 0)
        .toList();
    final pa = parts(a), pb = parts(b);
    for (var i = 0; i < 3; i++) {
      final x = i < pa.length ? pa[i] : 0;
      final y = i < pb.length ? pb[i] : 0;
      if (x != y) return x.compareTo(y);
    }
    return 0;
  }

  // ── Download & install ────────────────────────────────────────────────

  Future<void> downloadAndInstall() async {
    final release = availableUpdate.value;
    if (release?.apkUrl == null || release?.apkName == null) return;
    if (state.value == UpdateState.downloading) return;

    // Already downloaded earlier (e.g. the user backed out of the installer).
    final existing = await _apkFile(release!.apkName!, release.version);
    if (existing != null &&
        await existing.exists() &&
        (release.apkSize == 0 || await existing.length() == release.apkSize)) {
      await _install(existing.path);
      return;
    }

    lastError.value = '';
    needsInstallPermission.value = false;
    progress.value = 0;
    downloadedBytes.value = 0;
    totalBytes.value = release.apkSize;
    state.value = UpdateState.downloading;
    try {
      final id = await _channel.invokeMethod<num>('downloadUpdate', {
        'url': release.apkUrl,
        'filename': _fileName(release.apkName!, release.version),
      });
      _downloadId = id?.toInt();
    } catch (e) {
      _fail('Couldn\'t start the download.');
    }
  }

  Future<void> cancelDownload() async {
    final id = _downloadId;
    _downloadId = null;
    state.value = UpdateState.idle;
    if (id != null) {
      try {
        await _channel.invokeMethod('cancelUpdateDownload', {'downloadId': id});
      } catch (_) {}
    }
  }

  /// Progress events for "update:<filename>", routed from DownloadService.
  void onNativeProgress(
      String filename, int downloaded, int total, String status) {
    if (status == 'Downloading update...') {
      if (state.value != UpdateState.downloading) return;
      downloadedBytes.value = downloaded;
      if (total > 0) {
        totalBytes.value = total;
        progress.value = downloaded / total;
      }
    } else if (status == 'Update downloaded') {
      _downloadId = null;
      progress.value = 1;
      unawaited(_installDownloaded(filename));
    } else if (status.startsWith('Update failed')) {
      _downloadId = null;
      _fail('The download failed. Try again.');
    } else if (status == 'Update cancelled') {
      _downloadId = null;
      if (state.value == UpdateState.downloading) {
        state.value = UpdateState.idle;
      }
    }
  }

  Future<void> _installDownloaded(String filename) async {
    final dir = await _updatesDir();
    if (dir == null) return _fail('Update folder unavailable.');
    await _install('${dir.path}/$filename');
  }

  Future<void> _install(String path) async {
    state.value = UpdateState.installing;
    try {
      final result =
          await _channel.invokeMethod<String>('installApk', {'path': path});
      if (result == 'needs_permission') {
        _pendingInstallPath = path;
        needsInstallPermission.value = true;
      } else {
        _pendingInstallPath = null;
        needsInstallPermission.value = false;
      }
      state.value = UpdateState.idle;
    } catch (e) {
      _fail('Couldn\'t open the installer.');
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Back from the "Install unknown apps" screen: try again.
    if (state == AppLifecycleState.resumed && _pendingInstallPath != null) {
      final path = _pendingInstallPath!;
      _pendingInstallPath = null;
      unawaited(_install(path));
    }
  }

  void _fail(String message) {
    lastError.value = message;
    state.value = UpdateState.error;
  }

  // ── Files ─────────────────────────────────────────────────────────────

  static String _fileName(String apkName, String version) =>
      'PrivateLM-$version-$apkName';

  Future<Directory?> _updatesDir() async {
    final base = await getExternalStorageDirectory();
    if (base == null) return null;
    return Directory('${base.path}/updates');
  }

  Future<File?> _apkFile(String apkName, String version) async {
    final dir = await _updatesDir();
    return dir == null
        ? null
        : File('${dir.path}/${_fileName(apkName, version)}');
  }

  /// Removes downloaded APKs for versions that are already installed.
  Future<void> _cleanupOldApks() async {
    try {
      final dir = await _updatesDir();
      if (dir == null || !await dir.exists()) return;
      await for (final f in dir.list()) {
        if (f is! File) continue;
        final m =
            RegExp(r'PrivateLM-([\d.]+)-').firstMatch(f.uri.pathSegments.last);
        if (m == null ||
            compareVersions(m.group(1)!, currentVersion.value) <= 0) {
          await f.delete();
        }
      }
    } catch (_) {}
  }

  // ── UI ────────────────────────────────────────────────────────────────

  Future<void> showUpdateSheet([BuildContext? context]) async {
    final ctx = context ?? Get.context;
    if (ctx == null || availableUpdate.value == null || _sheetOpen) return;
    _sheetOpen = true;
    try {
      await showAppSheet(ctx, builder: (_) => const UpdateSheet());
    } finally {
      _sheetOpen = false;
    }
  }
}
