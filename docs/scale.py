"""Draw docs/scale.png: what the scale does to the frequency matrix.

Optimizes the IRF4 example on the first 1,000 sequences of each file (as -d),
on the CPU (a minute or two), then draws the motif at the fitted scale and at three fixed ones, and the
likelihood that fit_scale maximizes as a function of the scale.

Needs matplotlib, pandas and logomaker besides daimo. From the root of the
repository:

    python docs/scale.py
"""

import os
import sys

import logomaker
import matplotlib
import numpy as np
import pandas as pd
import torch
from Bio import motifs

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import daimo.damo  # noqa: E402
from daimo.damo import (_composition, _golden_max, _one_hot, damo,  # noqa: E402
                        fit_scale, pwm_to_pfm)
from daimo.io import read_sequences  # noqa: E402

POSITIVES = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
NEGATIVES = os.path.join(ROOT, "examples", "negative_seqs.fa.gz")
SEED_MOTIF = os.path.join(ROOT, "examples", "MA1419.1.jaspar")
FIXED_SCALES = (0.1, 0.3, 1.0)
FITTED_COLOR = "#D55E00"


def likelihood_curve(pwm, positive_seqs, negative_seqs):
    """
    The log-likelihood per sequence that fit_scale maximizes.

    Same model as fit_scale, for sequences without N and all of one length.

    Returns:
        Function of the scale.
    """
    width = pwm.shape[1]
    weights = pwm - pwm.mean(axis=0, keepdims=True)
    log_bg = np.log(_composition(negative_seqs))
    kernels = torch.from_numpy(np.stack([
        weights, weights[::-1, ::-1].copy(),
        np.repeat(log_bg[:, None], width, 1)]).astype(np.float32))
    out = torch.nn.functional.conv1d(_one_hot(positive_seqs), kernels)
    out = out.double().numpy()
    raw, bg = out[:, :2], out[:, 2]

    def log_likelihood(scale):
        log_z = np.log(np.exp(scale * weights).sum(axis=0)).sum()
        a = (scale * raw - log_z - bg[:, None, :]).reshape(len(raw), -1)
        top = a.max(axis=1, keepdims=True)
        m = (top + np.log(np.exp(a - top).sum(axis=1, keepdims=True)))[:, 0] \
            - np.log(a.shape[1])

        return _golden_max(
            lambda t: np.logaddexp(-np.logaddexp(0, t),
                                   -np.logaddexp(0, -t) + m).sum(),
            -12.0, 12.0)[1] / len(raw)

    return log_likelihood


def information(pfm):
    """Height of each base in a logo: frequency times the bits of its column."""
    return pfm * (pfm * np.log2(4 * np.clip(pfm, 1e-12, 1))).sum(axis=0)


def main(output_file):
    # On the CPU, so that the figure does not depend on the device: a GPU
    # ends at slightly different weights (and at about the same fitted bits)
    daimo.damo._get_device = lambda: "cpu"
    pwm = damo(POSITIVES, NEGATIVES, SEED_MOTIF, debugging=True)
    positives, negatives = (
        [str(record.seq).upper() for record in read_sequences(file, True)]
        for file in (POSITIVES, NEGATIVES))
    scale = fit_scale(pwm, positives, negatives)
    log_likelihood = likelihood_curve(pwm, positives, negatives)

    with open(SEED_MOTIF) as handle:
        counts = motifs.read(handle, "jaspar").counts
    seed = np.array([counts[base] for base in "ACGT"], float)
    seed /= seed.sum(axis=0)

    rows = [("JASPAR seed MA1419.1 (IRF4), the starting motif", seed),
            (f"daimo, fitted scale {scale:.3f} (--scale auto, the default)",
             pwm_to_pfm(pwm, scale)),
            ("daimo, --scale 0.1 (the default up to 1.0.1)",
             pwm_to_pfm(pwm, 0.1)),
            ("daimo, --scale 0.3", pwm_to_pfm(pwm, 0.3)),
            ("daimo, --scale 1 (DiMO's unscaled conversion)",
             pwm_to_pfm(pwm, 1.0))]

    fig = plt.figure(figsize=(10, 6.4), dpi=200)
    grid = fig.add_gridspec(len(rows), 2, width_ratios=[1.55, 1], hspace=0.75,
                            wspace=0.28, left=0.07, right=0.985, top=0.9,
                            bottom=0.1)

    # a: the logos
    for i, (title, pfm) in enumerate(rows):
        ax = fig.add_subplot(grid[i, 0])
        heights = information(pfm)
        logomaker.Logo(pd.DataFrame(heights.T, columns=list("ACGT")), ax=ax,
                       color_scheme="classic")
        ax.set_ylim(0, 2)
        ax.set_yticks([0, 1, 2])
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)
        fitted = i == 1
        ax.set_title(f"{title}: {heights.sum():.1f} bits", loc="left",
                     fontsize=9.5, pad=3,
                     color=FITTED_COLOR if fitted else "black",
                     fontweight="bold" if fitted else "normal")
        if i == 2:
            ax.set_ylabel("bits")
        print(f"{title}: {heights.sum():.1f} bits")

    # b: the likelihood as a function of the scale
    ax = fig.add_subplot(grid[:, 1])
    scales = np.geomspace(0.004, 2, 300)
    ax.plot(scales, [log_likelihood(s) for s in scales], color="#555555",
            lw=1.6)
    for s in FIXED_SCALES:
        ax.plot(s, log_likelihood(s), "o", color="#999999", ms=7)
        ax.annotate(f"{s:g}", (s, log_likelihood(s)), xytext=(8, 6),
                    textcoords="offset points", color="#444444", fontsize=9.5)
    ax.plot(scale, log_likelihood(scale), "o", color=FITTED_COLOR,
            mec="black", ms=9, zorder=5)
    ax.annotate(f"fitted\n{scale:.3f}", (scale, log_likelihood(scale)),
                xytext=(12, -4), textcoords="offset points",
                color=FITTED_COLOR, fontweight="bold", fontsize=9.5,
                va="center")
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 1])
    ax.set_xticklabels(["0.01", "0.1", "1"])
    top = log_likelihood(scale)
    ax.set_ylim(-0.12 * top, 1.12 * top)
    ax.set_xlabel("scale", fontsize=11)
    ax.set_ylabel("how well the motif explains\n"
                  "the sequences (log-likelihood per sequence)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.012, 0.955, "a", fontsize=15, fontweight="bold")
    fig.text(0.045, 0.957, "The IRF4 example at the fitted scale and at three "
             "fixed scales", fontsize=11)
    fig.text(0.635, 0.955, "b", fontsize=15, fontweight="bold")
    fig.text(0.668, 0.957, "The fitted scale is the top of this curve",
             fontsize=11)
    fig.savefig(output_file, facecolor="white")


if __name__ == "__main__":
    main(os.path.join(ROOT, "docs", "scale.png"))
