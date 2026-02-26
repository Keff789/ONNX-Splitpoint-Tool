"""Remote execution helpers (SSH/SCP).

The tool intentionally relies on the system ssh/scp clients to avoid adding new
runtime dependencies.
"""

from .ssh_transport import HostConfig, SSHTransport

__all__ = ["HostConfig", "SSHTransport"]
