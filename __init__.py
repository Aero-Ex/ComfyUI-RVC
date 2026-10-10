import os

import folder_paths

for _key, _path in (
    ("rvc", os.path.join(folder_paths.models_dir, "rvc")),
):
    folder_paths.add_model_folder_path(_key, _path)

from .nodes_rvc import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
