# CompletSetDev v2.80.1 regression evidence

Files outside `derived/` are original ZIP member bytes. `PROVENANCE.json` records their SHA256 and sizes. JSONs in `derived/` retain selected original values and hierarchy but are explicitly derived subsets, not original files. Each selection and original member SHA256 is recorded.

These are historical failures and successful hardware measurements; fixtures must never be rewritten to v2.80.2 success. No model, tensor array, image, or full debug ZIP is bundled. Regeneration is read-only with `verification/v2802/integration/audit_completsetdev.py`.
