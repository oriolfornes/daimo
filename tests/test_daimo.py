"""Checks that daimo follows DAMO.py (https://github.com/sx-ruan/DAMO).

    pixi run test
"""

import gzip
import os
import random

import numpy as np
import torch
from Bio import motifs
from Bio.Seq import Seq
from click.testing import CliRunner

import pytest

from daimo import cli
from daimo.io import read_pwm
from daimo.damo import (_compute_auc, _generate_pwm, _get_pwm_model,
                         _score_sequences, _update_pwm, damo, pwm_to_pfm)
from daimo.damo import _choose_width, _resize, _split, fit_scale, parse_widths
from daimo.damo import NothingToMatch, match_scale
from daimo.io import read_sequences
from daimo.sequence import dna_one_hot, dna_one_hot_many

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JASPAR = os.path.join(ROOT, "examples", "MA1419.1.jaspar")


def test_pseudocounts_follow_damo():
    # DAMO gen_pwm: PFM.Normalize(PFM.Normalize(counts), 1e-4), i.e. counts to
    # frequencies first, then add 1e-4 and renormalize
    pwm = _generate_pwm(["A"] * 500, 1)
    freq = np.array([1., 0., 0., 0.]) + 1e-4
    np.testing.assert_allclose(pwm[:, 0], np.log(freq / freq.sum()))
    assert abs(pwm[1, 0] - np.log(1e-4 / 1.0004)) < 1e-9     # -9.2, not log(1e-4 / 500.0004)


def test_one_hot_encodes_bases_and_zeroes_n():
    encoded = dna_one_hot("ACGTN")
    assert encoded.shape == (4, 5)
    np.testing.assert_array_equal(encoded.argmax(axis=0)[:4], [0, 1, 2, 3])
    assert encoded[:, 4].sum() == 0


def test_iterations_option_reaches_damo(monkeypatch, tmp_path):
    seen = {}

    def fake_damo(*args, **kwargs):
        seen.update(kwargs)
        return np.zeros((4, 2))

    monkeypatch.setattr(cli, "damo", fake_damo)
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    res = CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-i", "3", "-o", str(tmp_path / "out.txt")])
    assert res.exit_code == 0, res.output
    assert seen["iterations"] == 3


def test_pfm_is_boltzmann_distribution():
    pwm = np.log(np.array([[.7, .1], [.1, .2], [.1, .3], [.1, .4]]))
    # exp(W) normalized per column; a log PWM that is already a
    # distribution comes back unchanged
    np.testing.assert_allclose(pwm_to_pfm(pwm), np.exp(pwm))
    np.testing.assert_allclose(pwm_to_pfm(pwm).sum(axis=0), 1.)
    # no overflow on large weights
    assert np.isfinite(pwm_to_pfm(pwm * 1e4)).all()


def test_pfm_scale_sharpens_and_flattens():
    pwm = np.log(np.array([[.7], [.1], [.1], [.1]]))
    assert pwm_to_pfm(pwm, 2.)[0, 0] > pwm_to_pfm(pwm)[0, 0] > pwm_to_pfm(pwm, .5)[0, 0]


def _run(monkeypatch, tmp_path, args, pwm=None):
    pwm = np.log(np.full((4, 3), .25)) if pwm is None else pwm
    monkeypatch.setattr(cli, "damo", lambda *a, **k: pwm)
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    return CliRunner().invoke(cli.main, [fa, fa, JASPAR] + args)


def test_oformat_jaspar_is_readable_and_keeps_the_name(monkeypatch, tmp_path):
    out = tmp_path / "motif.jaspar.gz"
    res = _run(monkeypatch, tmp_path, ["--oformat", "jaspar", "--scale", "2", "-o", str(out)])
    assert res.exit_code == 0, res.output
    with gzip.open(out, "rt") as f:
        m = motifs.read(f, "jaspar")
    np.testing.assert_allclose([m.counts[b] for b in "ACGT"], .25)
    assert (m.matrix_id, m.name) == ("MA1419.1", "IRF4")


def test_oformat_meme_round_trips(monkeypatch, tmp_path):
    pwm = np.log(np.array([[.7, .1], [.1, .2], [.1, .3], [.1, .4]]))
    out = tmp_path / "motif.meme"
    res = _run(monkeypatch, tmp_path, ["--oformat", "meme", "--scale", "1", "-o", str(out)], pwm)
    assert res.exit_code == 0, res.output
    assert "MOTIF MA1419.1 IRF4" in out.read_text()
    back = read_pwm(str(out))
    # read as probabilities: 1e-4 added and renormalized
    expected = np.exp(pwm) + 1e-4
    np.testing.assert_allclose(np.exp(back), expected / expected.sum(axis=0), atol=1e-6)


def test_oformat_defaults_to_the_log_pwm(monkeypatch, tmp_path):
    pwm = np.log(np.array([[.7, .1], [.1, .2], [.1, .3], [.1, .4]]))
    out = tmp_path / "motif.txt"
    assert _run(monkeypatch, tmp_path, ["-o", str(out)], pwm).exit_code == 0
    np.testing.assert_allclose(np.loadtxt(out).T, pwm, atol=1e-8)


def _written(path):
    written = np.exp(read_pwm(str(path)))
    return written / written.sum(axis=0)


def test_frequency_formats_default_to_the_fitted_scale(monkeypatch, tmp_path):
    pwm = read_pwm(JASPAR) * 20
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    sequences = [str(r.seq) for r in read_sequences(fa, True)]
    scale = fit_scale(pwm, sequences, sequences)
    assert not np.isclose(scale, .1, rtol=.2)
    out = tmp_path / "motif.meme"
    res = _run(monkeypatch, tmp_path, ["-d", "--oformat", "meme", "-o", str(out)], pwm)
    assert res.exit_code == 0, res.output
    np.testing.assert_allclose(_written(out), pwm_to_pfm(pwm, scale), atol=5e-4)
    assert f"Fitted scale: {scale:.4g}" in res.output
    # --scale auto is the default, in any case
    for auto in ("auto", "AUTO"):
        out_auto = tmp_path / f"motif.{auto}.meme"
        res = _run(monkeypatch, tmp_path, ["-d", "--oformat", "meme", "--scale", auto,
                                           "-o", str(out_auto)], pwm)
        assert res.exit_code == 0, res.output
        assert out_auto.read_text() == out.read_text()
    # --scale 0.1 gives the output of 1.0.1, and fits nothing
    out01 = tmp_path / "motif01.meme"
    res = _run(monkeypatch, tmp_path, ["--oformat", "meme", "--scale", "0.1", "-o", str(out01)], pwm)
    assert res.exit_code == 0 and "Fitted scale" not in res.output
    np.testing.assert_allclose(_written(out01), pwm_to_pfm(pwm, .1), atol=5e-4)
    # an explicit scale of 1 still gives DiMO's unscaled conversion
    out1 = tmp_path / "motif1.meme"
    assert _run(monkeypatch, tmp_path, ["--oformat", "meme", "--scale", "1", "-o", str(out1)], pwm).exit_code == 0
    np.testing.assert_allclose(_written(out1), pwm_to_pfm(pwm, 1.), atol=5e-4)


def test_fitted_scale_is_reported_after_the_motif(monkeypatch, tmp_path):
    pwm = np.log(np.array([[.7, .1], [.1, .2], [.1, .3], [.1, .4]]))
    res = _run(monkeypatch, tmp_path, ["-d", "--oformat", "meme"], pwm)
    assert res.exit_code == 0, res.output
    assert res.output.startswith("MEME version 4")
    assert res.output.rstrip().splitlines()[-1].startswith("Fitted scale: ")


def test_scale_rejects_what_is_not_a_positive_number_or_auto(monkeypatch, tmp_path):
    for bad in ("banana", "0", "-1", "nan", "inf"):
        res = _run(monkeypatch, tmp_path, ["--oformat", "meme", f"--scale={bad}"])
        assert res.exit_code == 2 and "auto" in res.output and "seed" in res.output, bad


def test_oformat_writes_to_stdout(monkeypatch, tmp_path):
    res = _run(monkeypatch, tmp_path, ["--oformat", "meme"])
    assert res.exit_code == 0 and res.output.startswith("MEME version 4")


def test_scale_needs_a_frequency_format(monkeypatch, tmp_path):
    for scale in ("2", "auto", "seed"):
        res = _run(monkeypatch, tmp_path, ["--scale", scale, "-o", str(tmp_path / "o.txt")])
        assert res.exit_code == 2 and "--oformat" in res.output


def test_meme_input_name_is_kept_in_the_output(monkeypatch, tmp_path):
    meme = tmp_path / "in.meme"
    meme.write_text(MEME)
    monkeypatch.setattr(cli, "damo", lambda *a, **k: np.log(np.full((4, 2), .25)))
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    res = CliRunner().invoke(cli.main, [fa, fa, str(meme), "-m", "MA0002.1", "--oformat", "jaspar"])
    assert res.exit_code == 0, res.output
    assert res.output.startswith(">MA0002.1 second")


MEME = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF MA0001.1 first
letter-probability matrix: alength= 4 w= 2 nsites= 20 E= 0
  0.700000  0.100000  0.100000  0.100000
  0.100000  0.100000  0.100000  0.700000

MOTIF MA0002.1 second
letter-probability matrix: alength= 4 w= 3 nsites= 20 E= 0
  0.250000  0.250000  0.250000  0.250000
  0.250000  0.250000  0.250000  0.250000
  0.250000  0.250000  0.250000  0.250000
"""


def test_read_meme_motif_by_name_or_id(tmp_path):
    meme = tmp_path / "motifs.meme"
    meme.write_text(MEME)
    for name in ("MA0001.1 first", "MA0001.1"):
        pwm = read_pwm(str(meme), name)
        assert pwm.shape == (4, 2)
        # probabilities, with 1e-4 added and renormalized
        expected = np.array([[.7, .1], [.1, .1], [.1, .1], [.1, .7]]) + 1e-4
        np.testing.assert_allclose(pwm, np.log(expected / expected.sum(axis=0)))
    assert read_pwm(str(meme), "MA0002.1").shape == (4, 3)


def test_read_meme_needs_a_name_when_there_are_several(tmp_path):
    meme = tmp_path / "motifs.meme"
    meme.write_text(MEME)
    with pytest.raises(ValueError, match="--motif"):
        read_pwm(str(meme))
    with pytest.raises(ValueError, match="not found"):
        read_pwm(str(meme), "nope")


def test_read_meme_single_motif_gzipped(tmp_path):
    single = MEME.split("MOTIF MA0002.1")[0]
    meme = tmp_path / "motif.meme.gz"
    with gzip.open(meme, "wt") as f:
        f.write(single)
    assert read_pwm(str(meme)).shape == (4, 2)


def test_motif_option_reaches_damo(monkeypatch, tmp_path):
    seen = {}

    def fake_damo(*args, **kwargs):
        seen.update(kwargs)
        return np.zeros((4, 2))

    monkeypatch.setattr(cli, "damo", fake_damo)
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    res = CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-m", "MA0001.1", "-o", str(tmp_path / "o.txt")])
    assert res.exit_code == 0, res.output
    assert seen["motif_name"] == "MA0001.1"


def test_update_stops_on_perfect_separation():
    sites = ["AC", "AG", "CA", "TT"]
    pwm = np.zeros((4, 2))
    assert _update_pwm([1, 1, 0, 0], [3., 2., 1., 0.], sites, 1., pwm) is None
    assert _update_pwm([1, 0, 1, 0], [3., 2., 1., 0.], sites, 1., pwm) is not None


def test_damo_returns_on_separable_data(tmp_path):
    with open(JASPAR) as f:
        site = str(motifs.read(f, "jaspar").consensus)
    rng = random.Random(1714)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    pos.write_text("".join(f">p{i}\n{rand(20)}{site}{rand(20)}\n" for i in range(50)))
    neg.write_text("".join(f">n{i}\n{rand(40 + len(site))}\n" for i in range(50)))
    pwm = damo(str(pos), str(neg), JASPAR, iterations=20)
    assert np.isfinite(pwm).all()


def test_reported_auc_is_the_accepted_one(tmp_path, capsys):
    # Weak signal: the run ends when every learning rate is rejected. The
    # reported final AUC must be that of the returned PWM, not of the last
    # rejected candidate.
    with open(JASPAR) as f:
        site = str(motifs.read(f, "jaspar").consensus)
    rng = random.Random(7)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    pos.write_text("".join(
        f">p{i}\n{rand(30)}{site if i % 3 == 0 else rand(len(site))}{rand(30)}\n"
        for i in range(300)))
    neg.write_text("".join(f">n{i}\n{rand(60 + len(site))}\n" for i in range(300)))
    pwm = damo(str(pos), str(neg), JASPAR, iterations=100)

    # Recompute the AUC of the returned PWM, forward and reverse strands
    def best(path):
        seqs = [str(r.seq).upper() for r in read_sequences(path)]
        both = seqs + [str(Seq(x).reverse_complement()) for x in seqs]
        model = _get_pwm_model(pwm, len(seqs[0]))
        scores, _ = _score_sequences(model, torch.Tensor(dna_one_hot_many(both).tolist()),
                                     both, pwm.shape[1])
        return scores

    expected = _compute_auc(best(str(pos)), best(str(neg)))[0]
    reported = float(capsys.readouterr().out.split(" to ")[-1].rstrip(".\n "))
    assert abs(reported - expected) < 1e-9


def test_resize_pads_with_uniform_columns_half_on_each_side():
    pwm = np.log(np.full((4, 3), .1) + np.eye(4)[:, [0, 1, 2]] * .6)
    big = _resize(pwm, 8)
    assert big.shape == (4, 8)
    np.testing.assert_array_equal(big[:, 2:5], pwm)           # 2 left, 3 right
    np.testing.assert_allclose(big[:, :2], np.log(.25))
    np.testing.assert_allclose(big[:, 5:], np.log(.25))
    assert _resize(pwm, 3) is pwm


def test_resize_keeps_the_most_informative_columns():
    uniform, sharp = [.25] * 4, [.97, .01, .01, .01]
    # columns 3 and 4 are the informative ones
    pfm = np.array([uniform, uniform, uniform, sharp, sharp, uniform, uniform]).T
    pwm = _resize(np.log(pfm), 2)
    np.testing.assert_allclose(pwm, np.log(pfm[:, 3:5]))
    # ties go to the most central run
    flat = np.log(np.full((4, 7), .25))
    np.testing.assert_array_equal(_resize(flat, 3), flat[:, 2:5])


def test_choose_width_prefers_the_smallest_within_tolerance():
    assert _choose_width({8: .700, 10: .7015, 12: .702}) == 8
    assert _choose_width({8: .700, 10: .7030, 12: .704}) == 10
    assert _choose_width({8: .700, 10: .700, 12: .710}) == 12


def test_split_holds_out_a_fraction():
    train, val = _split(list(range(100)), 1714)
    assert len(val) == 20 and len(train) == 80
    assert sorted(train + val) == list(range(100))
    assert _split(list(range(100)), 1714) == (train, val)


def test_widths_search_returns_a_valid_pwm(tmp_path, capsys):
    with open(JASPAR) as f:
        site = str(motifs.read(f, "jaspar").consensus)[3:12]   # 9 of 15 columns
    rng = random.Random(1714)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    pos.write_text("".join(f">p{i}\n{rand(40)}{site}{rand(40)}\n" for i in range(200)))
    neg.write_text("".join(f">n{i}\n{rand(89)}\n" for i in range(200)))
    pwm = damo(str(pos), str(neg), JASPAR, iterations=30, widths=[8, 12, 200])
    out = capsys.readouterr().out
    assert pwm.shape[0] == 4 and pwm.shape[1] in (8, 12, 15)   # 200 is too long
    assert np.isfinite(pwm).all()
    assert "width   8" in out and "width  15" in out and "width 200" not in out
    assert out.count("<- chosen") == 1


def test_width_options_reach_damo(monkeypatch, tmp_path):
    seen = {}

    def fake_damo(*args, **kwargs):
        seen.update(kwargs)
        return np.zeros((4, 2))

    monkeypatch.setattr(cli, "damo", fake_damo)
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    out = str(tmp_path / "o.txt")
    CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-o", out])
    assert seen["widths"] is None
    CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-w", "6,9", "-o", out])
    assert seen["widths"] == "6,9"
    CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-w", "5-19", "-o", out])
    assert seen["widths"] == "5-19"


def test_widths_syntax_errors_are_usage_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "damo", lambda *a, **k: np.zeros((4, 2)))
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    for bad in ("19-5", "x", "0", "-", "8,,10"):
        res = CliRunner().invoke(cli.main, [fa, fa, JASPAR, "-w", bad])
        assert res.exit_code == 2 and "Invalid width" in res.output, bad


def test_parse_widths_like_cut():
    assert parse_widths("5-19") == list(range(5, 20))
    assert parse_widths("8,10,12") == [8, 10, 12]
    assert parse_widths("12,8,10,8") == [8, 10, 12]
    assert parse_widths(" 8 , 10-12 ") == [8, 10, 11, 12]
    assert parse_widths("-4") == [1, 2, 3, 4]
    assert parse_widths("5-8,12,15-", maximum=17) == [5, 6, 7, 8, 12, 15, 16, 17]
    assert parse_widths("5-8,6-10") == [5, 6, 7, 8, 9, 10]
    assert parse_widths("7") == [7]


def test_parse_widths_rejects_bad_input():
    for bad in ("", "a", "19-5", "0", "-", "5--7", "8,,10", "1.5", "0-3"):
        with pytest.raises(ValueError, match="Invalid width"):
            parse_widths(bad)
    with pytest.raises(ValueError, match="open-ended"):
        parse_widths("5-")


def test_widths_string_reaches_the_search(tmp_path, capsys):
    with open(JASPAR) as f:
        site = str(motifs.read(f, "jaspar").consensus)[3:12]
    rng = random.Random(1714)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    pos, neg = tmp_path / "pos.fa", tmp_path / "neg.fa"
    pos.write_text("".join(f">p{i}\n{rand(40)}{site}{rand(40)}\n" for i in range(100)))
    neg.write_text("".join(f">n{i}\n{rand(89)}\n" for i in range(100)))
    # open-ended: 14 up to the sequence length; here 14 and the seed's 15, up to 89,
    # so cap it with a range to keep the run short
    pwm = damo(str(pos), str(neg), JASPAR, iterations=5, widths="5-7,15")
    out = capsys.readouterr().out
    for k in (5, 6, 7, 15):
        assert f"width {k:>3}:" in out
    assert "width   8" not in out and pwm.shape[1] in (5, 6, 7, 15)


def _planted(seed=1714, width=10, n=3000, length=100, with_site=.7, conc=.25):
    """Random sequences; a fraction of the positives has a site drawn from a PFM."""
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    rand = lambda: "".join(prng.choice("ACGT") for _ in range(length))  # noqa: E731
    pfm = rng.dirichlet(np.full(4, conc), width).T             # columns of mixed sharpness
    pfm = (pfm + .01) / (pfm + .01).sum(0)
    pos, neg = [], []
    for _ in range(n):
        s = rand()
        neg.append(rand())
        if prng.random() < with_site:
            site = "".join("ACGT"[rng.choice(4, p=pfm[:, j])] for j in range(width))
            if prng.random() < .5:
                site = str(Seq(site).reverse_complement())
            i = prng.randrange(0, length - width + 1)
            s = s[:i] + site + s[i + width:]
        pos.append(s)
    return pfm, pos, neg


@pytest.fixture(scope="module")
def planted():
    return _planted()


def test_fit_scale_undoes_the_size_of_the_weights(planted):
    pfm, pos, neg = planted
    rng = np.random.default_rng(1714)
    products = []
    for size in (.1, 1., 10., 37.):
        # arbitrary size, arbitrary constant per column
        pwm = size * np.log(pfm) + rng.normal(0, 3, (1, pfm.shape[1]))
        scale, details = fit_scale(pwm, pos, neg, return_details=True)
        products.append(scale * size)
        # the fitted matrix is the matrix the sites were drawn from
        assert np.abs(pwm_to_pfm(pwm, scale) - pfm).max() < .05
        assert abs(details["fraction_with_site"] - .7) < .08
        assert not details["at_edge"]
    products = np.array(products)
    assert np.all(np.abs(products - 1) < .15)
    # same answer whatever the size of the weights
    assert products.max() / products.min() < 1.02


def test_fit_scale_with_fewer_sites():
    pfm, pos, neg = _planted(seed=7, with_site=.3)
    scale, details = fit_scale(np.log(pfm) * 5, pos, neg, return_details=True)
    assert abs(scale * 5 - 1) < .2
    assert abs(details["fraction_with_site"] - .3) < .08


def test_fit_scale_skips_n_and_short_sequences(planted):
    pfm, pos, _ = planted
    # variable lengths, N, a sequence shorter than the motif, no negatives
    mixed = pos[:500] + ["ACGT", "N" * 50, pos[0][:40] + "NNN" + pos[0][40:]]
    scale, details = fit_scale(np.log(pfm), mixed, return_details=True)
    assert np.isfinite(scale) and scale > 0
    assert details["sequences"] == 501                         # not ACGT, not the Ns
    with pytest.raises(ValueError, match="as long as the motif"):
        fit_scale(np.log(pfm), ["ACGT"])
    with pytest.raises(ValueError, match="real bases"):
        fit_scale(np.log(pfm), ["N" * 50], ["ACGT"])


def test_fit_scale_of_a_flat_pwm_is_1(planted):
    _, pos, neg = planted
    # uniform at every scale: no fit, no error
    assert fit_scale(np.zeros((4, 6)), pos[:10]) == 1.
    assert fit_scale(np.full((4, 6), -1.386), pos[:10], neg[:10]) == 1.


def test_fit_scale_samples_beyond_max_sequences(planted):
    pfm, pos, neg = planted
    capped = fit_scale(np.log(pfm), pos, neg, max_sequences=1000)
    everything = fit_scale(np.log(pfm), pos, neg, max_sequences=None)
    assert abs(capped / everything - 1) < .1
    assert capped == fit_scale(np.log(pfm), pos, neg, max_sequences=1000)   # seeded


def _bits(pfm):
    """Information content of a PFM, in bits."""
    pfm = np.asarray(pfm, float)
    return float((pfm * np.log2(np.clip(pfm, 1e-300, None) / .25)).sum())


def _seed_19():
    """Log PWM of 19 columns: 9 informative ones (5 to 13) between weak flanks."""
    rng = np.random.default_rng(1714)
    flank = lambda n: rng.dirichlet(np.full(4, 40.), n).T        # noqa: E731, nearly flat
    core = rng.dirichlet(np.full(4, .3), 9).T + .01
    return np.log(np.hstack([flank(5), core / core.sum(0), flank(5)]))


def test_match_scale_gives_the_bits_of_the_seed_whatever_the_size_of_the_weights():
    rng = np.random.default_rng(1714)
    seed = np.log(rng.dirichlet(np.full(4, .5), 12).T + .01)
    optimized = np.log(rng.dirichlet(np.full(4, .5), 12).T + .01)   # another motif
    target = _bits(pwm_to_pfm(seed))
    products = []
    for size in (.1, 1., 10., 37.):
        # arbitrary size, arbitrary constant per column
        pwm = size * optimized + rng.normal(0, 3, (1, 12))
        scale, details = match_scale(pwm, seed, return_details=True)
        assert abs(_bits(pwm_to_pfm(pwm, scale)) - target) < 1e-6
        assert abs(details["seed_bits"] - target) < 1e-9 and abs(details["bits"] - target) < 1e-6
        assert not details["at_edge"] and not details["flat"] and details["width"] == 12
        # the preferred base of each position is the optimized motif's, not the seed's
        assert np.array_equal(pwm_to_pfm(pwm, scale).argmax(0), optimized.argmax(0))
        products.append(scale * size)
    assert max(products) / min(products) < 1 + 1e-6
    # a seed matched to itself keeps its scale
    assert abs(match_scale(seed, seed) - 1) < 1e-6
    assert abs(match_scale(5 * seed, seed) - .2) < 1e-6


def test_match_scale_counts_only_the_seed_columns_that_were_optimized():
    seed = _seed_19()
    whole = _bits(pwm_to_pfm(seed))
    centre = _bits(pwm_to_pfm(seed[:, 5:14]))
    assert np.array_equal(_resize(seed, 9), seed[:, 5:14]) and whole - centre > .05
    rng = np.random.default_rng(7)
    pwm = 30 * np.log(rng.dirichlet(np.full(4, .5), 9).T + .01)     # 9 columns, large weights
    scale, details = match_scale(pwm, seed, return_details=True)
    # the bits of the 9 columns kept, not those of the 19
    assert abs(details["seed_bits"] - centre) < 1e-9
    assert abs(_bits(pwm_to_pfm(pwm, scale)) - centre) < 1e-6
    # every width follows the columns damo starts from
    for width in (5, 7, 12, 19):
        pwm = np.log(rng.dirichlet(np.full(4, .5), width).T + .01)
        _, details = match_scale(pwm, seed, return_details=True)
        assert abs(details["seed_bits"] - _bits(pwm_to_pfm(_resize(seed, width)))) < 1e-9
        assert abs(details["bits"] - details["seed_bits"]) < 1e-6


def test_match_scale_of_a_wider_motif_uses_the_whole_seed():
    seed = _seed_19()[:, 5:14]
    rng = np.random.default_rng(3)
    pwm = np.log(rng.dirichlet(np.full(4, .5), 15).T + .01)         # 6 columns added to the seed
    scale, details = match_scale(pwm, seed, return_details=True)
    # padded columns are uniform: no bits
    assert abs(details["seed_bits"] - _bits(pwm_to_pfm(seed))) < 1e-9
    assert abs(_bits(pwm_to_pfm(pwm, scale)) - details["seed_bits"]) < 1e-6


def test_match_scale_edge_cases():
    seed = _seed_19()
    # a flat PWM is uniform at every scale: 1, and no error
    assert match_scale(np.zeros((4, 9)), seed) == 1.
    scale, details = match_scale(np.full((4, 9), -1.386), seed, return_details=True)
    assert scale == 1. and details["flat"] and not details["at_edge"]
    # a seed without information in the columns kept has nothing to match
    pwm = np.log(np.random.default_rng(1).dirichlet(np.full(4, .5), 6).T + .01)
    with pytest.raises(NothingToMatch, match="no information in the 6 columns"):
        match_scale(pwm, np.log(np.full((4, 6), .25)))
    assert issubclass(NothingToMatch, ValueError)
    # two tied best bases: at most 1 bit per column, the seed has more
    tied = np.log(np.array([[.45, .45, .05, .05]] * 6).T)
    sharp = np.log(np.array([[.97, .01, .01, .01]] * 6).T)
    scale, details = match_scale(tied, sharp, return_details=True)
    assert details["at_edge"] and np.isfinite(scale) and scale > 0
    assert details["bits"] < 6.001 < details["seed_bits"]
    assert np.all(np.isfinite(pwm_to_pfm(tied, scale)))
    with pytest.raises(ValueError, match="not finite"):
        match_scale(np.full((4, 6), np.nan), sharp)


def test_scale_seed_matches_the_seed(monkeypatch, tmp_path):
    seed = read_pwm(JASPAR)
    pwm = seed * 20
    out = tmp_path / "motif.meme"
    res = _run(monkeypatch, tmp_path, ["--oformat", "meme", "--scale", "seed", "-o", str(out)], pwm)
    assert res.exit_code == 0, res.output
    # the seed, 20 times larger, comes back as the seed
    np.testing.assert_allclose(_written(out), pwm_to_pfm(seed), atol=5e-4)
    assert "Scale matched to the seed: 0.05 (" in res.output and "Fitted scale" not in res.output
    assert f"{_bits(pwm_to_pfm(seed)):.2f} bits in the 15 seed columns" in res.output
    # any case, and to stdout: the report comes after the motif
    res = _run(monkeypatch, tmp_path, ["--oformat", "meme", "--scale", "SEED"], pwm)
    assert res.exit_code == 0 and res.output.startswith("MEME version 4")
    assert res.output.rstrip().splitlines()[-1].startswith("Scale matched to the seed: ")
    # a narrower motif gets the bits of the seed columns it was cut to
    rng = np.random.default_rng(1714)
    narrow = 50 * np.log(rng.dirichlet(np.full(4, .5), 7).T + .01)
    out7 = tmp_path / "motif7.meme"
    res = _run(monkeypatch, tmp_path, ["--oformat", "meme", "--scale", "seed", "-o", str(out7)], narrow)
    assert res.exit_code == 0, res.output
    kept = _bits(pwm_to_pfm(_resize(seed, 7)))
    assert kept < _bits(pwm_to_pfm(seed)) - 1
    np.testing.assert_allclose(_written(out7), pwm_to_pfm(narrow, match_scale(narrow, seed)), atol=5e-4)
    assert abs(_bits(_written(out7)) - kept) < .05
    assert "in the 7 seed columns" in res.output
    # the default is still the fitted scale
    res = _run(monkeypatch, tmp_path, ["-d", "--oformat", "meme"], pwm)
    assert "Fitted scale: " in res.output and "matched" not in res.output


def test_scale_seed_falls_back_to_the_fit_for_a_flat_seed(monkeypatch, tmp_path):
    flat = tmp_path / "flat.jaspar"
    flat.write_text(">FLAT\tflat\n" + "".join(f"{b}  [ 100 100 100 100 ]\n" for b in "ACGT"))
    pwm = np.log(np.array([[.7, .1, .1, .1], [.1, .2, .7, .1], [.1, .3, .1, .1], [.1, .4, .1, .7]]))
    monkeypatch.setattr(cli, "damo", lambda *a, **k: pwm)
    fa = os.path.join(ROOT, "examples", "positive_seqs.fa.gz")
    out = tmp_path / "motif.meme"
    res = CliRunner().invoke(cli.main, [fa, fa, str(flat), "-d", "--oformat", "meme",
                                        "--scale", "seed", "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert "no information in the 4 columns" in res.output and "Fitted scale: " in res.output
    sequences = [str(r.seq) for r in read_sequences(fa, True)]
    np.testing.assert_allclose(_written(out), pwm_to_pfm(pwm, fit_scale(pwm, sequences, sequences)), atol=5e-4)
    # a flat PWM has nothing to match either, and is written as it is
    monkeypatch.setattr(cli, "damo", lambda *a, **k: np.log(np.full((4, 3), .25)))
    res = CliRunner().invoke(cli.main, [fa, fa, JASPAR, "--oformat", "meme", "--scale", "seed"])
    assert res.exit_code == 0 and "nothing to match" in res.output


def test_scale_seed_after_a_width_search(tmp_path):
    with open(JASPAR) as f:
        site = str(motifs.read(f, "jaspar").consensus)[3:12]
    rng = random.Random(1714)
    rand = lambda n: "".join(rng.choice("ACGT") for _ in range(n))  # noqa: E731
    pos, neg, out = tmp_path / "pos.fa", tmp_path / "neg.fa", tmp_path / "motif.meme"
    pos.write_text("".join(f">p{i}\n{rand(40)}{site}{rand(40)}\n" for i in range(100)))
    neg.write_text("".join(f">n{i}\n{rand(89)}\n" for i in range(100)))
    # the real optimization: the motif written has the bits of the seed columns of its width
    res = CliRunner().invoke(cli.main, [str(pos), str(neg), JASPAR, "-i", "5", "-w", "5-7",
                                        "--oformat", "meme", "--scale", "seed", "-o", str(out)])
    assert res.exit_code == 0, res.output
    written = _written(out)
    width = written.shape[1]
    assert width in (5, 6, 7, 15)
    kept = _bits(pwm_to_pfm(_resize(read_pwm(JASPAR), width)))
    assert abs(_bits(written) - kept) < .05
    assert f"in the {width} seed columns" in res.output
