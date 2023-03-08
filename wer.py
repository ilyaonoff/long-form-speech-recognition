from typing import List, Tuple

import editdistance
import jiwer

__all__ = ['word_error_rate', 'word_error_rate_detail']


def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same
    length.
    Args:
      hypotheses: list of hypotheses
      references: list of references
      use_cer: bool, set True to enable cer
    Returns:
      (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        # May deprecate using editdistance in future release for here and rest of codebase
        # once we confirm jiwer is reliable.
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer


def word_error_rate_detail(
        hypotheses: List[str], references: List[str], use_cer=False
) -> Tuple[float, int, float, float, float]:
    """
    Computes Average Word Error Rate with details (insertion rate, deletion rate, substitution rate)
    between two texts represented as corresponding lists of string.

    Hypotheses and references must have same length.
    Args:
      hypotheses (list): list of hypotheses
      references(list) : list of references
      use_cer (bool): set True to enable cer
    Returns:
      wer (float): average word error rate
      words (int):  Total number of words/charactors of given reference texts
      ins_rate (float): average insertion error rate
      del_rate (float): average deletion error rate
      sub_rate (float): average substitution error rate

    """
    scores = 0
    words = 0
    ops_count = {'substitutions': 0, 'insertions': 0, 'deletions': 0}

    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )

    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()

        # To get rid of the issue that jiwer does not allow empty string
        if len(r_list) == 0:
            if len(h_list) != 0:
                errors = len(h_list)
                ops_count['insertions'] += errors
        else:
            if use_cer:
                measures = jiwer.cer(r, h, return_dict=True)
            else:
                measures = jiwer.compute_measures(r, h)

            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']
            ops_count['insertions'] += measures['insertions']
            ops_count['deletions'] += measures['deletions']
            ops_count['substitutions'] += measures['substitutions']

        scores += errors
        words += len(r_list)

    if words != 0:
        wer = 1.0 * scores / words
        ins_rate = 1.0 * ops_count['insertions'] / words
        del_rate = 1.0 * ops_count['deletions'] / words
        sub_rate = 1.0 * ops_count['substitutions'] / words
    else:
        wer, ins_rate, del_rate, sub_rate = float('inf'), float('inf'), float('inf'), float('inf')

    return wer, words, ins_rate, del_rate, sub_rate
