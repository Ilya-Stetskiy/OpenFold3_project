# Copyright 2025 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from pathlib import Path

from openfold3.core.utils.s3 import download_s3_file

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("~/.openfold3/").expanduser()
CHECKPOINT_ROOT_FILENAME = "ckpt_root"
CHECKPOINT_NAME = "of3-p2-v1.pt"

OPENFOLD_BUCKET = "openfold"
CHECKPOINT_S3_KEY = f"staging/{CHECKPOINT_NAME}"


def download_model_parameters(download_dir: Path) -> None:
    """Download OpenFold3 model parameters from S3 if not already present.

    Args:
        download_dir: Directory to download the checkpoint file into.
            The file will be saved as ``download_dir / CHECKPOINT_NAME``.
    """
    download_dir = Path(download_dir)
    target_path = download_dir / CHECKPOINT_NAME

    if target_path.exists():
        logger.info("Parameters already present at %s", target_path)
        return

    confirm = input(
        f"Download {CHECKPOINT_S3_KEY} from s3://{OPENFOLD_BUCKET} "
        f"to {target_path}? (yes/no): "
    )

    if confirm.lower() in ["yes", "y"]:
        download_s3_file(OPENFOLD_BUCKET, CHECKPOINT_S3_KEY, target_path)
    else:
        logger.warning("Download cancelled")
