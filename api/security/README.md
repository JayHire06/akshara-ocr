# Security — Joshua's module

Authentication, input validation, hashing, rate limiting, and security middleware for the backend API. Findings and outstanding risks are tracked in [`../../docs/security-audit.md`](../../docs/security-audit.md).

## Modules

- `auth.py` — authentication handlers.
- `file_validator.py` — rejects malformed or unsafe uploads before they reach the OCR path.
- `hashing.py` — password/secret hashing helpers.
- `rate_limiter.py` — `slowapi` integration; attached to `app.state.limiter` in `api/main.py`.
- `middleware.py` — request-level hooks (security headers, etc.).

The rate limiter and its exception handler are both optional imports in `api/main.py` — if `slowapi` isn't installed the API still boots, without rate limiting. The dev stack always ships it.
