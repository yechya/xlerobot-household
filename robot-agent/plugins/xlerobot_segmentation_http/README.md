# XLeRobot Local HTTP Segmentation Backend

Reference `xlerobot.segmentation_backends` plugin for a reviewed segmentation service on loopback. The GPU/model server remains a separate process; this package provides a strict, bounded transport adapter and no model dependency.

The endpoint must be `http://127.0.0.1:<port>/v1/segment` (or another loopback spelling). Configure the plugin through the main project's reviewed `segmentation-backend.yaml`; include the server manifest, bearer token, plugin wheel and deployment lockfile in the bound artifacts.
