import os.path
import subprocess
import sys

import hydra
import dataclasses
from hydra.core.config_store import ConfigStore
from tqdm import tqdm


@dataclasses.dataclass
class Config:
    manifests: dict
    out_dir: str
    is_rnnt: bool
    pretrained_name: str
    total_buffer_in_secs: float
    chunk_len_in_secs: float
    model_stride: int
    batch_size: int = 1
    lcs: bool = False


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def __create_ctc_cmd(cfg: Config, in_manifest: str, out_manifest: str):
    return [
        "python", "NeMo/examples/asr/asr_chunked_inference/ctc/speech_to_text_buffered_infer_ctc.py",
        f'pretrained_name="{cfg.pretrained_name}"',
        f'dataset_manifest="{in_manifest}"',
        f'output_filename="{out_manifest}"',
        f"total_buffer_in_secs={cfg.total_buffer_in_secs}",
        f"chunk_len_in_secs={cfg.chunk_len_in_secs}",
        f"model_stride={cfg.model_stride}",
        f"batch_size={cfg.batch_size}"
    ]


def __create_rnnt_cmd(cfg: Config, in_manifest: str, out_manifest: str):
    return [
        "python", "NeMo/examples/asr/asr_chunked_inference/rnnt/speech_to_text_buffered_infer_rnnt.py",
        f'pretrained_name="{cfg.pretrained_name}"',
        f'dataset_manifest="{in_manifest}"',
        f'output_filename="{out_manifest}"',
        f"total_buffer_in_secs={cfg.total_buffer_in_secs}",
        f"chunk_len_in_secs={cfg.chunk_len_in_secs}",
        f"model_stride={cfg.model_stride}",
        f"batch_size={cfg.batch_size}"
    ]


@hydra.main(version_base=None, config_path="nemo_configs", config_name="buffered_ctc")
def evaluate(cfg: Config):
    if cfg.is_rnnt:
        create_cmd = __create_rnnt_cmd
    else:
        if cfg.lcs:
            raise ValueError("LCS method is supported with only rnnt")
        create_cmd = __create_ctc_cmd

    for tag, manifest_path in tqdm(cfg.manifests.items()):
        out_manifest_path = os.path.join(cfg.out_dir, f"{tag}.json")
        cmd = create_cmd(cfg, manifest_path, out_manifest_path)
        p = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        p.wait()
        if p.returncode != 0:
            raise RuntimeError(f"Not zero return code {p.returncode} after cmd: {' '.join(cmd)}")
    print('Done!')


if __name__ == "__main__":
    evaluate() # noqa
