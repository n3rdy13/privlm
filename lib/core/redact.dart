/// Masks API keys and tokens in free-form text before it is logged, shown in
/// the log viewer, copied, or uploaded to crash reporting.
library;

// Characters that end a credential value: whitespace, quotes, separators.
const _valueEnd = '\\s,;&"\'}';

final List<(RegExp, String Function(Match))> _rules = [
  // Query-string credentials: ?key=..., &api_key=..., access_token=...
  (
    RegExp('([?&](?:key|api_key|apikey|token|access_token)=)[^$_valueEnd]+',
        caseSensitive: false),
    (m) => '${m[1]}[REDACTED]',
  ),
  // Authorization: Bearer <token>
  (
    RegExp(r'(Bearer\s+)[A-Za-z0-9._~+/=\-]+', caseSensitive: false),
    (m) => '${m[1]}[REDACTED]',
  ),
  // Header-style credentials: x-api-key: ..., "x-goog-api-key": "..."
  (
    RegExp('((?:x-goog-api-key|x-api-key)["\']?\\s*[:=]\\s*["\']?)[^$_valueEnd]+',
        caseSensitive: false),
    (m) => '${m[1]}[REDACTED]',
  ),
  // Bare provider keys that may appear on their own.
  (RegExp(r'\bsk-[A-Za-z0-9_\-]{8,}'), (_) => 'sk-[REDACTED]'),
  (RegExp(r'\bAIza[0-9A-Za-z_\-]{20,}'), (_) => 'AIza[REDACTED]'),
  (RegExp(r'\bnvapi-[A-Za-z0-9_\-]{8,}'), (_) => 'nvapi-[REDACTED]'),
  (RegExp(r'\baichat_[0-9a-f]{16,}'), (_) => 'aichat_[REDACTED]'),
];

String redactSecrets(String input) {
  var out = input;
  for (final (pattern, replace) in _rules) {
    out = out.replaceAllMapped(pattern, replace);
  }
  return out;
}
