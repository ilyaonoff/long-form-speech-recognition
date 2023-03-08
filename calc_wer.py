import argparse
import json

from wer import word_error_rate_detail

def normalize(_str: str, num_to_words: bool = True, langid="en") -> str:
    """
    Remove unauthorized characters in a string, lower it and remove unneeded spaces
    """
    replace_with_space = [char for char in '/?*\"\',.:=?_{|}~¨«·»¡¿„…‧‹›≪≫!:;ː→']
    #replace_with_blank = [char for char in '`¨´‘’“”`ʻ‘’“"‘”']
    #replace_with_apos = [char for char in '‘’ʻ‘’‘']
    _str = _str.strip()
    _str = _str.lower()
    # for i in replace_with_blank:
    #     _str = _str.replace(i, "")
    for i in replace_with_space:
        _str = _str.replace(i, " ")
    # for i in replace_with_apos:
    #     _str = _str.replace(i, "'")
    # if num_to_words:
    #     if langid == "en":
    #         _str = convert_num_to_words(_str, langid="en")
    #     else:
    #         logging.info(
    #             "Currently support basic num_to_words in English only. Please use Text Normalization to convert other languages! Skipping!"
    #         )

    ret = " ".join(_str.split())
    return ret

def process(in_filepath: str, out_filepath: str, metrics_filepath: str):
    entries = []
    hypos = []
    refs = []
    with open(in_filepath) as f:
        for line in f:
            sample = json.loads(line)

            ref = normalize(sample["text"])
            hypo = normalize(sample["pred_text"])

            wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(
                hypotheses=[hypo],
                references=[ref],
                use_cer=False
            )

            sample["words"] = words
            sample["wer"] = wer
            sample["ins_rate"] = ins_rate
            sample["del_rate"] = del_rate
            sample["sub_rate"] = sub_rate

            entries.append(sample)
            hypos.append(hypo)
            refs.append(ref)

    total_wer, total_words, total_ins_rate, total_del_rate, total_sub_rate = word_error_rate_detail(
        hypotheses=hypos,
        references=refs,
        use_cer=False
    )

    metrics = {
        "wer": total_wer,
        "words": total_words,
        "ins_rate": total_ins_rate,
        "del_rate": total_del_rate,
        "sub_rate": total_sub_rate
    }

    with open(out_filepath, "w") as f:
        for entry in entries:
            f.write(f"{json.dumps(entry)}\n")

    with open(metrics_filepath, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Total metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", type=str, required=True)
    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--output_metrics", type=str, required=True)
    args = parser.parse_args()

    process(
        in_filepath=args.input_manifest,
        out_filepath=args.output_manifest,
        metrics_filepath=args.output_metrics
    )
