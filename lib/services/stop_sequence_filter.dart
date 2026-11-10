/// Cuts a token stream at the first stop sequence (OpenAI semantics: the
/// stop sequence itself is never emitted).
///
/// Tokens can split a stop sequence across chunks, so the last
/// `longestStop - 1` characters are held back until it's clear they aren't
/// the start of one.
class StopSequenceFilter {
  StopSequenceFilter(Iterable<String> stops)
      : _stops = stops.where((s) => s.isNotEmpty).toSet().toList();

  final List<String> _stops;
  late final int _holdBack = _stops.isEmpty
      ? 0
      : _stops.map((s) => s.length).reduce((a, b) => a > b ? a : b) - 1;
  String _text = '';
  int _emitted = 0;
  bool _stopped = false;

  bool get isActive => _stops.isNotEmpty;

  /// True once a stop sequence has been seen; further input is ignored.
  bool get stopped => _stopped;

  /// All accepted text so far, excluding any stop sequence.
  String get text => _text;

  /// Feeds a token and returns the text that is now safe to emit.
  String add(String token) {
    if (_stopped || token.isEmpty) return '';
    _text += token;

    // Only text around the new token can contain a fresh match.
    final searchFrom = _emitted - _holdBack < 0 ? 0 : _emitted - _holdBack;
    var cut = -1;
    for (final stop in _stops) {
      final i = _text.indexOf(stop, searchFrom);
      if (i >= 0 && (cut < 0 || i < cut)) cut = i;
    }
    if (cut >= 0) {
      _stopped = true;
      _text = _text.substring(0, cut);
      final out = cut > _emitted ? _text.substring(_emitted) : '';
      _emitted = _text.length;
      return out;
    }

    final safeEnd = _text.length - _holdBack;
    if (safeEnd <= _emitted) return '';
    final out = _text.substring(_emitted, safeEnd);
    _emitted = safeEnd;
    return out;
  }

  /// Releases any held-back text once generation has ended normally.
  String flush() {
    if (_stopped || _emitted >= _text.length) return '';
    final out = _text.substring(_emitted);
    _emitted = _text.length;
    return out;
  }
}
