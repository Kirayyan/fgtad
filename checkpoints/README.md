# Checkpoint Explanation

The final reproducible weights are saved with the package to:

``text repro_assets/models/ repro_assets/temporal_model/

```

Do not treat `checkpoints/` as the main weight directory. The code reads by default:

```text src/project_v2/models/ src/project_v2/temporal_model/

```

Please execute the following in the root directory of the open-source package:

```bash python3 scripts/install_repro_assets.py

```

This script will link the checkpoint and temporal weights in `repro_assets/` to the default code path.