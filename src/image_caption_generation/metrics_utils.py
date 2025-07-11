import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer
from rouge_score import rouge_scorer

def calculate_bleu(reference, hypothesis):
    # reference: list of reference captions (list of strings)
    # hypothesis: generated caption (string)
    smoothie = SmoothingFunction().method4
    ref_tokens = [ref.lower().split() for ref in reference]
    hyp_tokens = hypothesis.lower().split()
    return sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothie)

def calculate_wer(reference, hypothesis):
    # reference: string (single reference, concatenated)
    # hypothesis: string
    return wer(reference.lower(), hypothesis.lower())

def calculate_rouge(reference, hypothesis):
    # reference: string (single reference, concatenated)
    # hypothesis: string
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure

# Batch metrics for a list of (reference, hypothesis) pairs
def compute_metrics(references, hypotheses):
    bleu_scores = []
    wer_scores = []
    rouge_scores = []
    for refs, hyp in zip(references, hypotheses):
        bleu_scores.append(calculate_bleu(refs, hyp))
        wer_scores.append(calculate_wer(' '.join(refs), hyp))
        rouge_scores.append(calculate_rouge(' '.join(refs), hyp))
    return {
        'bleu': np.mean(bleu_scores),
        'wer': np.mean(wer_scores),
        'rouge': np.mean(rouge_scores)
    }
