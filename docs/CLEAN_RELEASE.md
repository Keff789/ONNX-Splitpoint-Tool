# Clean release policy

The clean delivery ZIP should contain only source code, templates, resources and current documentation needed to run the tool.

Included through a strict path allow-list:

```text
onnx_splitpoint_tool/
scripts/
docs/current docs, calibration references and explicitly retained release docs
profiles/current top-level profiles
tests/
README.md
pyproject.toml
requirements.txt
start_gui.sh
analyse_and_split.py
analyse_and_split_gui.py
```

Excluded from the delivery ZIP:

```text
__pycache__/
*.pyc
.pytest_cache/
EvaluationRuns/
RemoteBenchmarkRuns/
BenchmarkSets/
*.onnx
*.hef
*.har
*.harn
*.npz
*.npy
*.tar.gz
*.zip
historical scratch trackers / old release-note clutter
unlisted historical release guides and build reports
profiles/legacy_v58/
```

Runtime artifacts should be uploaded through EvaluationRun debug packs, not kept in the source package.

Compatibility code and regression tests may stay in the package when they are
needed for old run resume/import paths. The current test guide and build report
are included together with the small, explicit set of recent historical guides
and reports named by `scripts/build_source_manifest.py`; every other historical
release document remains excluded. Empty unused scaffolds, generated package
metadata and stale docs do not enter the clean ZIP.
