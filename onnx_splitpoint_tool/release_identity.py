"""Single authoritative Python identity for the current release.

Shell launchers, packaging metadata, and documentation still carry literal
values because they must remain independently inspectable.  Python modules
must import these constants instead of repeating the current release identity.
"""
from __future__ import annotations


VERSION = "2.82"
RELEASE = VERSION
DEVELOPMENT_LINEAGE = "v2.79"
BUILD_ID = "v2.82-selected-energy-generic-roles-workspace-product-evidence"
BUILD_CONTRACT_VERSION = 2
SOURCE_INTEGRITY_CONTRACT = "installed_release_source_manifest_runtime_binding"
BYTECODE_ISOLATION_CONTRACT = "source_local_bytecode_cache_isolation"
REMOTE_ENERGY_PRIMARY_ADMISSION_CONTRACT = (
    "benchmark_primary_remote_energy_admission_before_transport"
)
