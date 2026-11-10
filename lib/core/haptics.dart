import 'package:flutter/services.dart';

/// App-wide haptic feedback. Every vibration goes through here so the
/// "Haptic feedback" setting can turn them all off in one place.
class Haptics {
  Haptics._();

  /// Mirrors the user's setting; loaded at startup by SettingsController.
  static bool enabled = true;

  static void selectionClick() {
    if (enabled) HapticFeedback.selectionClick();
  }

  static void lightImpact() {
    if (enabled) HapticFeedback.lightImpact();
  }

  static void mediumImpact() {
    if (enabled) HapticFeedback.mediumImpact();
  }

  static void heavyImpact() {
    if (enabled) HapticFeedback.heavyImpact();
  }
}
