import 'package:flutter/services.dart';
import 'package:get/get.dart';
import '../core/redact.dart';
import 'crash_reporting_service.dart';

class AppLogEntry {
  final DateTime timestamp;
  final String level;
  final String message;
  final String? details;

  AppLogEntry({
    required this.level,
    required this.message,
    this.details,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  bool get isImportant => level == 'ERROR' || level == 'WARNING';

  String format() {
    final buffer = StringBuffer()
      ..write('[${timestamp.toIso8601String()}] ')
      ..write('$level: $message');
    if (details != null && details!.trim().isNotEmpty) {
      buffer.write('\n$details');
    }
    return buffer.toString();
  }
}

class AppLogService extends GetxService {
  final entries = <AppLogEntry>[].obs;

  void warning(String message, {Object? details}) {
    _add('WARNING', message, details);
  }

  void error(String message, {Object? details}) {
    _add('ERROR', message, details);
  }

  void info(String message, {Object? details}) {
    _add('INFO', message, details);
  }

  void debug(String message, {Object? details}) {
    _add('DEBUG', message, details);
  }

  void _add(String level, String rawMessage, Object? details) {
    // Error text can embed request URLs or headers; never keep keys in logs.
    final message = redactSecrets(rawMessage);
    final detailText = details == null ? null : redactSecrets('$details');
    entries.insert(
      0,
      AppLogEntry(
        level: level,
        message: message,
        details: detailText,
      ),
    );
    if (entries.length > 200) {
      entries.removeRange(200, entries.length);
    }
    if ((level == 'ERROR' || level == 'WARNING') &&
        Get.isRegistered<CrashReportingService>()) {
      Get.find<CrashReportingService>().recordNonFatal(
        details ?? message,
        reason: message,
        extra: {'app_log_level': level},
      );
    }
  }

  List<AppLogEntry> get importantEntries =>
      entries.where((entry) => entry.isImportant).toList();

  String get shareText {
    final selected = importantEntries.isEmpty ? entries : importantEntries;
    return selected.map((entry) => entry.format()).join('\n\n');
  }

  Future<void> copyImportantLogs() async {
    await Clipboard.setData(ClipboardData(text: shareText));
  }

  void clear() {
    entries.clear();
  }
}
