import os
import shutil


def on_pre_build(config):
    src = "assets"
    dst = os.path.join(config["docs_dir"], "assets")
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
