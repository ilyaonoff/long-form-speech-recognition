import argparse
import fnmatch
import functools
import json
import logging
import multiprocessing
import os
import subprocess
import tarfile
import urllib.request
import shutil

from sox import Transformer
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Noise Data download")
parser.add_argument("--data_root", required=True, default=None, type=str)
args = parser.parse_args()

URLS = {
    "SLR28": ("http://www.openslr.org/resources/28/rirs_noises.zip"),
}


def __retrieve_with_progress(source: str, filename: str):
    """
    Downloads source to destination
    Displays progress bar
    Args:
        source: url of resource
        destination: local filepath
    Returns:
    """
    with open(filename, "wb") as f:
        response = urllib.request.urlopen(source)
        total = response.length

        if total is None:
            f.write(response.content)
        else:
            with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                for data in response:
                    f.write(data)
                    pbar.update(len(data))


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    Returns:
    """
    source = URLS[source]
    if not os.path.exists(destination):
        logging.info("{0} does not exist. Downloading ...".format(destination))

        __retrieve_with_progress(source, filename=destination + ".tmp")

        os.rename(destination + ".tmp", destination)
        logging.info("Downloaded {0}.".format(destination))
    else:
        logging.info("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_file(filepath: str, data_dir: str):
    try:
        tar = tarfile.open(filepath)
        tar.extractall(data_dir)
        tar.close()
    except Exception:
        logging.info("Not extracting. Maybe already there?")


def __process_transcript(file_path: str, dst_folder: str):
    """
    Converts flac files to wav from a given transcript, capturing the metadata.
    Args:
        file_path: path to a source transcript  with flac sources
        dst_folder: path where wav files will be stored
    Returns:
        a list of metadata entries for processed files.
    """
    filename = os.path.basename(file_path)

    n_chans = int(subprocess.check_output("soxi -c {0}".format(file_path), shell=True))
    assert n_chans == 1

    wav_file = os.path.join(dst_folder, filename)
    shutil.copy(file_path, wav_file)

    # check duration
    duration = subprocess.check_output("soxi -D {0}".format(wav_file), shell=True)

    entry = {}
    entry["audio_filepath"] = os.path.abspath(wav_file)
    entry["duration"] = float(duration)
    entry["text"] = "_"
    return entry


def __process_data(data_folder: str, dst_folder: str, manifest_file: str, num_workers: int):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder: source with flac files
        dst_folder: where wav files will be stored
        manifest_file: where to store manifest
        num_workers: number of parallel workers processing files
    Returns:
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    files = []
    entries = []

    for filename in os.listdir(data_folder):
        entries.append(__process_transcript(os.path.join(data_folder, filename), dst_folder))
        files.append(os.path.join(data_folder, filename))

    with open(manifest_file, "w") as fout:
        for m in entries:
            fout.write(json.dumps(m) + "\n")


def main():
    data_root = args.data_root
    data_set = "SLR28"
    num_workers = args.num_workers

    if args.log:
        logging.basicConfig(level=logging.INFO)
    logging.info("\n\nWorking on: {0}".format(data_set))
    filepath = os.path.join(data_root, data_set + ".tar.gz")
    logging.info("Getting {0}".format(data_set))
    __maybe_download_file(filepath, data_set.upper())
    logging.info("Extracting {0}".format(data_set))
    __extract_file(filepath, data_root)
    logging.info("Processing {0}".format(data_set))
    __process_data(
        os.path.join(data_root, "RIRS_NOISES", "pointsource_noises"),
        os.path.join(data_root, "RIRS_NOISES", "pointsource_noises") + "-processed",
        os.path.join(data_root, "noises.json"),
        num_workers=num_workers,
    )
    logging.info("Done!")


if __name__ == "__main__":
    main()