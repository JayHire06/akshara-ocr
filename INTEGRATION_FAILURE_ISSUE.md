# 🚨 Integration Test Failure: Missing Test Dependencies

**Module**: Infrastructure / Testing
**Responsible Agent**: @infra-agent / @test-agent
**Date**: 2026-02-28

## Issue Description
As the Project Lead (Zayed), I attempted to run the bi-weekly integration testing via:
```bash
python tests/integration/test_end_to_end.py
```
The integration test immediately failed with the following traceback:
```
ModuleNotFoundError: No module named 'pytest'
```

## Interface Contract Violation?
While this is not technically an interface contract code violation, it is a **Project Standard Violation**. All tests must be runnable in our standardized environment.

## Action Required
1. Ensure `pytest` and required test dependencies are correctly configured in our environment setup scripts (e.g., test-requirements.txt or `docker-compose`).
2. Only add test packages to the environment. Do not clutter the main `requirements.txt` with test libraries unless approved by me.
3. Once the environment is fixed, execute the test suite and verify no OCR interface contracts (Preprocessor, Model, NLP) are broken.

Please submit a PR to fix the test container environment and request my approval.
