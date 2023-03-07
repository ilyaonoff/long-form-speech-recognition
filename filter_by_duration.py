import argparse
import json
from typing import Optional
from tqdm import tqdm


def filter_audio(infile: str, outfile: str, min_duration: Optional[float], max_duration: Optional[float]):
    with open(infile, "r") as f:
        manifest = f.readlines()
    num_filtered = 0
    with open(outfile, "w") as out:
        for sample in tqdm(manifest):
            sample = json.loads(sample)
            duration = sample["duration"]
            if min_duration is not None and duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                continue
            num_filtered += 1
            out.write(f"{json.dumps(sample)}\n")
    print(f"Done! There are {num_filtered} filtered samples")


if __name__ == "__main":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--filtered_manifest", type=str, required=True)
    parser.add_argument("--min_duration", type=float, default=None)
    parser.add_argument("--max_duration", type=float, default=None)
    args = parser.parse_args()

    filter_audio(
        infile=args.manifest,
        outfile=args.filtered_manifest,
        min_duration=args.min_duration,
        max_duration=args.max_duration
    )
