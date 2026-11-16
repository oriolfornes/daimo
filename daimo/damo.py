"""DAMO (Discriminative Additive Model Optimization) in PyTorch.

Level 1 of DAMO.py (https://github.com/sx-ruan/DAMO), which is DiMO
(Patel and Stormo, 2014): a single PWM is refined so that the best site in
each sequence separates positives from negatives. The motif can also be
optimized at several widths, keeping the one that does best on held-out
sequences.
"""

import copy
from collections import Counter

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

from daimo.io import read_pwm, read_sequences
from daimo.sequence import dna_one_hot_many

bar_format = "{desc}{percentage:3.0f}%|{bar:20}{r_bar}"


# Motif width search: how much held-out AUC
# a wider motif may lose against the best one before it is passed over, and
# the fraction of sequences held out
WIDTH_TOLERANCE = 0.002
VALIDATION_FRACTION = 0.2

# Scale applied to the PWM before it is written as frequencies. The optimized
# PWM only has to rank positives above negatives, and a ranking does not change
# when every weight is multiplied by the same number: the size of the weights is
# arbitrary, and it keeps growing while daimo runs. So no fixed scale is right.
# "auto" fits it to the sequences (see fit_scale). A number is used as it is:
# 0.1 was the default up to 1.0.1, and 1 is DiMO's unscaled conversion.
DEFAULT_SCALE = "auto"

# fit_scale. The search interval is in units of "nats between the best and the
# worst base of an average column": 0.05 is flat, 60 is one-hot. Dividing by the
# average column range of the PWM makes the search independent of the size of
# the weights.
BASES = "ACGT"
SPREAD_RANGE = (0.05, 60.0)
GRID_POINTS = 60
MAX_SEQUENCES = 10000      # the fit does not need more: a seeded sample is used beyond this
SEED = 1714
GOLDEN = (np.sqrt(5) - 1) / 2

LEARNING_RATES = [1.0, 0.55, 0.1]


def _parse_width_ranges(spec):
    """
    Parse a width list in the style of `cut -f`.

    Items are separated by commas: N, N-M, -M (1 to M) and N- (N up to the
    longest width the sequences allow).

    Returns:
        List of (first, last) tuples; last is None for an open-ended range.
    """
    ranges = []
    for item in str(spec).split(","):
        item = item.strip()
        try:
            if "-" in item:
                first, _, last = item.partition("-")
                if not first.strip() and not last.strip():
                    raise ValueError
                first = int(first) if first.strip() else 1
                last = int(last) if last.strip() else None
            else:
                first = last = int(item)
        except ValueError:
            raise ValueError(
                f"Invalid width {item!r} in {spec!r}: use N, N-M, -M or N-, "
                "separated by commas") from None
        if first < 1 or (last is not None and last < first):
            raise ValueError(f"Invalid width range {item!r} in {spec!r}")
        ranges.append((first, last))

    return ranges


def parse_widths(spec, maximum=None):
    """
    Expand a width list in the style of `cut -f`, e.g. "5-19", "8,10,12" or
    "5-8,12,15-".

    Args:
        spec: The list, as a string.
        maximum: Where an open-ended range (N-) stops.

    Returns:
        Sorted list of distinct widths.
    """
    widths = set()
    for first, last in _parse_width_ranges(spec):
        if last is None:
            if maximum is None:
                raise ValueError(f"{spec!r} has an open-ended range; "
                                 "it needs a maximum width")
            last = maximum
        widths.update(range(first, last + 1))

    return sorted(widths)


def damo(positive_seqs_file, negative_seqs_file, motif_file,
         debugging=False, iterations=500, motif_name=None,
         widths=None, seed=1714):
    """
    Optimize a PWM to separate positive from negative sequences.

    Every sequence is scanned on both strands. Each iteration moves the
    PWM towards the sites of positives in the overlap zone and away from
    the sites of negatives. It stops early when the AUC stops improving,
    or when the classes are perfectly separated.

    With widths, the motif is optimized at each width and the best one is
    kept. 20% of the sequences of each class are held out. For each width
    the seed is cut to its most informative columns, or padded with
    uniform ones, optimized on the other 80%, and scored on the held-out
    sequences. The seed's own width is always tried. The smallest width
    within 0.002 of the best held-out AUC wins.

    Args:
        positive_seqs_file: FASTA file of positive sequences.
        negative_seqs_file: FASTA file of negative sequences.
        motif_file: JASPAR or MEME motif used as the seed PWM.
        debugging: If True, use only the first 1,000 sequences per file.
        iterations: Maximum number of optimization iterations.
        motif_name: Motif to use when a MEME file has several.
        widths: Motif widths to try, as a list of ints or a string like
            `cut -f` ("5-19", "8,10,12", "5-8,12,15-"; see parse_widths).
            None: only the seed's width, with no held-out set. Widths longer
            than the sequences are skipped.
        seed: Random seed of the held-out split (widths only).

    Returns:
        Log PWM of shape (4, motif length), rows A, C, G and T. The
        motif length is the seed's unless widths are given.
    """

    device = _get_device()

    # Get sequences, and the seed PWM
    positive_records = read_sequences(positive_seqs_file, debugging)
    negative_records = read_sequences(negative_seqs_file, debugging)
    pwm = read_pwm(motif_file, motif_name)

    if widths is None:
        positives = _encode(positive_records, device)
        negatives = _encode(negative_records, device)
        pwm, original_auc, auc = _optimize(pwm, positives, negatives,
            iterations, device, desc=f"Optimizing width {pwm.shape[1]}: ")
        print("\n" + f"AUC optimized from {original_auc} to {auc}." + "\n")

        return pwm

    # Hold out sequences to compare the widths
    positive_records, positive_val = _split(positive_records, seed)
    negative_records, negative_val = _split(negative_records, seed)
    positives = _encode(positive_records, device)
    negatives = _encode(negative_records, device)
    positives_val = _encode(positive_val, device)
    negatives_val = _encode(negative_val, device)

    # Widths to try, as far as the sequences allow
    sequence_length = positives[1].shape[-1]
    if isinstance(widths, str):
        widths = parse_widths(widths, sequence_length)
    candidates = sorted({int(k) for k in widths} | {pwm.shape[1]})
    candidates = [k for k in candidates if 1 <= k <= sequence_length]
    if not candidates:
        raise ValueError(f"No width fits sequences of length {sequence_length}")

    # Optimize at each width
    pwms, aucs = {}, {}
    for k in candidates:
        pwms[k], _, _ = _optimize(_resize(pwm, k), positives, negatives,
            iterations, device, desc=f"Optimizing width {k}: ")
        aucs[k] = _auc(pwms[k], positives_val, negatives_val, device)
    best = _choose_width(aucs)

    seed_auc = _auc(pwm, positives_val, negatives_val, device)
    print("\n" + f"AUC on held-out sequences: seed {seed_auc}")
    for k in candidates:
        print(f"  width {k:>3}: {aucs[k]}" + ("  <- chosen" if k == best else ""))
    print()

    return pwms[best]


def _get_device():
    """The best available torch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode(records, device):
    """
    Both strands of the records, one-hot encoded.

    Returns:
        Tuple (sequences, tensor): the forward strands first and the
        reverse complements after, as strings and as a one-hot tensor.
    """
    sequences = [str(s.seq).upper() for s in records] + \
        [str(s.reverse_complement().seq).upper() for s in records]
    one_hot = dna_one_hot_many(sequences)

    return sequences, torch.Tensor(one_hot.tolist()).to(device)


def _optimize(pwm, positives, negatives, iterations, device, desc=None):
    """
    Run the optimization loop on a seed PWM.

    Args:
        pwm: Seed log PWM of shape (4, motif length).
        positives: Tuple (sequences, tensor) from _encode.
        negatives: Tuple (sequences, tensor) from _encode.
        iterations: Maximum number of iterations.
        device: Torch device.
        desc: Label for the progress bar.

    Returns:
        Tuple (pwm, original_auc, auc): the optimized PWM, and the AUC
        of the seed and of the PWM returned.
    """
    positive_seqs, positive_seqs_tensor = positives
    negative_seqs, negative_seqs_tensor = negatives
    sequence_length = positive_seqs_tensor.shape[-1]

    # Get a PyTorch model of the PWM
    model = _get_pwm_model(pwm, sequence_length, device)

    # Score sequences
    positive_scores, positive_sites = _score_sequences(
        model, positive_seqs_tensor, positive_seqs, pwm.shape[1]
    )
    negative_scores, negative_sites = _score_sequences(
        model, negative_seqs_tensor, negative_seqs, pwm.shape[1],
    )

    # Compute AUC
    original_auc, y_true, y_score = _compute_auc(positive_scores,
        negative_scores)
    auc = copy.copy(original_auc)

    # For each iteration...
    for _ in tqdm(range(iterations), total=iterations, desc=desc,
                  bar_format=bar_format):

        # Initialize
        pwm_0 = copy.copy(pwm)
        auc_0 = copy.copy(auc)
        y_true_0 = copy.copy(y_true)
        y_score_0 = copy.copy(y_score)
        sites_0 = copy.copy(positive_sites) + copy.copy(negative_sites)

        # For each learning rate...
        for lr in LEARNING_RATES:

            # Update PWM
            pwm_1 = _update_pwm(y_true_0, y_score_0, sites_0, lr, pwm_0)

            # Perfect separation (i.e. no overlap zone): nothing to learn
            if pwm_1 is None:
                break

            # Get a PyTorch model of the updated PWM
            model = _get_pwm_model(pwm_1, sequence_length, device)

            # Score sequences
            positive_scores, positive_sites = _score_sequences(
                model, positive_seqs_tensor, positive_seqs, pwm_1.shape[1]
            )
            negative_scores, negative_sites = _score_sequences(
                model, negative_seqs_tensor, negative_seqs, pwm_1.shape[1],
            )

            # Compute AUC of the candidate
            auc_1, y_true_1, y_score_1 = _compute_auc(positive_scores,
                negative_scores)

            # Accept the candidate only if it improves the AUC. A rejected
            # candidate must not leak into auc, which is reported at the end.
            if auc_1 > auc_0:
                pwm = copy.copy(pwm_1)
                auc, y_true, y_score = auc_1, y_true_1, y_score_1
                break

        # Early stop
        if np.array_equal(pwm_0, pwm):
            break

    return pwm, original_auc, auc


def _auc(pwm, positives, negatives, device):
    """AUC of a PWM on encoded sequences (best site on either strand)."""
    sequence_length = positives[1].shape[-1]
    model = _get_pwm_model(pwm, sequence_length, device)
    positive_scores, _ = _score_sequences(model, positives[1], positives[0],
        pwm.shape[1])
    negative_scores, _ = _score_sequences(model, negatives[1], negatives[0],
        pwm.shape[1])

    return _compute_auc(positive_scores, negative_scores)[0]


def _split(records, seed, fraction=VALIDATION_FRACTION):
    """Randomly hold out a fraction of the records; returns (train, held out)."""
    n_val = max(1, int(round(len(records) * fraction)))
    if len(records) - n_val < 1:
        raise ValueError("Too few sequences to hold out a validation set")
    held_out = set(np.random.RandomState(seed).permutation(len(records))[:n_val].tolist())

    return ([r for i, r in enumerate(records) if i not in held_out],
            [records[i] for i in sorted(held_out)])


def _resize(pwm, width):
    """
    Cut a PWM down to its most informative columns, or pad it.

    A narrower width keeps the run of columns with the most information
    (ties go to the most central run). A wider one adds uniform columns,
    half on each side (the odd one on the right).

    Args:
        pwm: Log PWM of shape (4, motif length).
        width: Wanted number of columns.

    Returns:
        Log PWM of shape (4, width).
    """
    motif_length = pwm.shape[1]
    if width == motif_length:
        return pwm

    if width > motif_length:
        left = (width - motif_length) // 2
        right = width - motif_length - left
        return np.hstack([np.full((4, left), np.log(0.25)), pwm,
                          np.full((4, right), np.log(0.25))])

    # Information content of each column, in bits
    pfm = pwm_to_pfm(pwm)
    information = (pfm * np.log2(pfm / 0.25)).sum(axis=0)
    totals = np.convolve(information, np.ones(width), mode="valid")
    centre = (motif_length - width) / 2
    start = min(range(len(totals)), key=lambda i: (-round(totals[i], 9),
                                                    abs(i - centre)))

    return pwm[:, start:start + width]


def _choose_width(aucs, tolerance=WIDTH_TOLERANCE):
    """
    Pick a width from the held-out AUC of each.

    The smallest width within the tolerance of the best AUC wins, so extra
    columns have to pay for themselves.

    Args:
        aucs: Dict mapping width to held-out AUC.
        tolerance: Held-out AUC a narrower motif may lose against the best.

    Returns:
        The chosen width.
    """
    best = max(aucs.values())

    return min(k for k, auc in aucs.items() if auc >= best - tolerance - 1e-12)


def pwm_to_pfm(pwm, scale=1.0):
    """
    Convert a log PWM to a PFM with a Boltzmann distribution.

    As in DiMO (Patel and Stormo, 2014), the weights are read as energies
    in units of kT: each position is a softmax over A, C, G and T. A scale
    above 1 sharpens the columns and one below 1 flattens them.

    Args:
        pwm: Log PWM of shape (4, motif length), rows A, C, G and T.
        scale: Factor applied to the weights before the softmax.

    Returns:
        Frequencies of shape (4, motif length); each column sums to 1.
    """
    scaled = scale * np.asarray(pwm, float)
    z = np.exp(scaled - scaled.max(axis=0, keepdims=True))

    return z / z.sum(axis=0, keepdims=True)


def _one_hot(sequences):
    """
    One-hot encode sequences of any length for fit_scale.

    Returns:
        Tensor of shape (n, 4, longest), float32; N and padding are all-zero
        columns.
    """
    lut = np.full(256, 4, dtype=np.int64)
    for i, b in enumerate(BASES):
        lut[ord(b)] = i
    width = max(len(s) for s in sequences)
    idx = np.full((len(sequences), width), 4, dtype=np.int64)
    for i, s in enumerate(sequences):
        idx[i, :len(s)] = lut[np.frombuffer(s.upper().encode(), dtype=np.uint8)]

    return torch.from_numpy(
        np.eye(5, dtype=np.float32)[idx][:, :, :4].transpose(0, 2, 1).copy())


def _composition(sequences):
    """Strand-symmetric base frequencies (A, C, G, T), with a count of 1 added."""
    counts = _one_hot(sequences).sum((0, 2)).numpy().astype(float) + 1.0
    counts = counts + counts[::-1]

    return counts / counts.sum()


def _golden_max(fun, lo, hi, iterations=40):
    """Maximum of a unimodal function on [lo, hi]; returns (argmax, max)."""
    a, b = lo, hi
    c, d = b - GOLDEN * (b - a), a + GOLDEN * (b - a)
    fc, fd = fun(c), fun(d)
    for _ in range(iterations):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - GOLDEN * (b - a)
            fc = fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + GOLDEN * (b - a)
            fd = fun(d)

    return (a + b) / 2, max(fc, fd)


def fit_scale(pwm, positive_seqs, negative_seqs=None,
              max_sequences=MAX_SEQUENCES, seed=SEED, batch_size=4096,
              return_details=False):
    """
    Scale at which softmax(scale * pwm) best describes the positive sequences.

    The scaled PWM is read as a probability model of the sites: each positive
    sequence has no site or one, at any position of either strand, and the
    rest of it follows the background. The scale and the fraction of
    sequences with a site are fitted together by maximum likelihood. A
    constant added to a column changes nothing, and neither does the size of
    the weights: the search interval is set from their spread.

    Only windows made of real bases count (N and padding are skipped), and
    sequences shorter than the motif are dropped.

    Args:
        pwm: Log PWM of shape (4, motif length), rows A, C, G and T (what
            `damo` returns).
        positive_seqs: Sequences (strings) that the motif should explain.
            Scanned on both strands.
        negative_seqs: Sequences whose strand-symmetric base composition is
            the background. None: the positives' own composition.
        max_sequences: Fit on at most this many positives (a random sample
            with `seed`); None for all.
        seed: Random seed of that sample.
        batch_size: Sequences scanned at a time.
        return_details: Also return a dict: fraction_with_site,
            log_likelihood, at_edge (the fit ended at the edge of its search
            interval), sequences (the number used) and spread.

    Returns:
        The scale (float), 1.0 for a flat PWM, which is uniform at every
        scale. Use it as `pwm_to_pfm(pwm, scale)`.
    """
    pwm = np.asarray(pwm, dtype=np.float64)
    width = pwm.shape[1]

    # A constant per column changes nothing
    weights = pwm - pwm.mean(axis=0, keepdims=True)
    spread = float((weights.max(axis=0) - weights.min(axis=0)).mean())
    if not np.all(np.isfinite(weights)):
        raise ValueError("the PWM has values that are not finite")

    # A flat PWM is uniform at every scale
    if spread <= 0:
        if return_details:
            return 1.0, dict(fraction_with_site=float("nan"),
                             log_likelihood=0.0, at_edge=False, sequences=0,
                             spread=0.0)
        return 1.0

    log_bg = np.log(_composition(
        negative_seqs if negative_seqs is not None else positive_seqs))

    usable = [s for s in positive_seqs if len(s) >= width]
    if max_sequences is not None and len(usable) > max_sequences:
        picks = np.sort(np.random.default_rng(seed).choice(
            len(usable), max_sequences, replace=False))
        usable = [usable[i] for i in picks]
    if not usable:
        raise ValueError("no positive sequence is as long as the motif")

    # Scan with four filters: the motif, its reverse complement, the
    # background, and the number of real bases in the window
    x = _one_hot(usable)
    kernels = torch.from_numpy(np.stack([
        weights, weights[::-1, ::-1].copy(),
        np.repeat(log_bg[:, None], width, 1),
        np.ones((4, width))]).astype(np.float32))
    out = torch.cat([torch.nn.functional.conv1d(x[i:i + batch_size], kernels)
                     for i in range(0, len(x), batch_size)]).double().numpy()

    # Windows without N or padding
    valid = out[:, 3] > width - 0.5
    keep = valid.any(axis=1)
    raw, bg, valid = out[keep, :2], out[keep, 2], valid[keep]   # raw: (n, 2, positions)
    n_windows = 2 * valid.sum(axis=1)
    if len(raw) == 0:
        raise ValueError("no positive sequence has a window of real bases as "
                         "long as the motif")

    def site_log_odds(scale):
        """Per sequence: log of the mean, over its windows and strands, of
        P(window | motif) / P(window | background)."""
        log_z = np.log(np.exp(scale * weights).sum(axis=0)).sum()
        a = np.where(valid[:, None, :], scale * raw - log_z - bg[:, None, :],
                     -np.inf).reshape(len(raw), -1)
        top = a.max(axis=1, keepdims=True)

        return (top + np.log(np.exp(a - top).sum(axis=1, keepdims=True)))[:, 0] \
            - np.log(n_windows)

    def log_likelihood(scale):
        """Best (logit of the fraction of sequences with a site, log-likelihood)."""
        m = site_log_odds(scale)

        return _golden_max(
            lambda t: np.logaddexp(-np.logaddexp(0, t),
                                   -np.logaddexp(0, -t) + m).sum(),
            -12.0, 12.0)

    # A grid first, then a finer search around its best point
    grid = np.geomspace(*SPREAD_RANGE, GRID_POINTS) / spread
    values = np.array([log_likelihood(s)[1] for s in grid])
    best = int(values.argmax())
    at_edge = best in (0, len(grid) - 1)
    lo = np.log(grid[max(best - 1, 0)])
    hi = np.log(grid[min(best + 1, len(grid) - 1)])
    log_scale, ll = _golden_max(
        lambda t: log_likelihood(float(np.exp(t)))[1], lo, hi, iterations=20)
    scale = float(np.exp(log_scale))
    if not return_details:
        return scale
    logit, _ = log_likelihood(scale)

    return scale, dict(fraction_with_site=float(1 / (1 + np.exp(-logit))),
                       log_likelihood=float(ll), at_edge=bool(at_edge),
                       sequences=int(len(raw)), spread=spread)


def _get_pwm_model(pwm, sequence_length, device="cpu"):
    """
    Build a fixed PyTorch model that scores a PWM.

    A convolution with the PWM as its only filter, followed by a max pool
    over the whole sequence. The max pool returns the position of the best
    site too.

    Args:
        pwm: Log PWM of shape (4, motif length).
        sequence_length: Length of the (one-hot) sequences to score.
        device: Torch device to place the model on.

    Returns:
        torch.nn.Sequential returning (scores, site positions).
    """
    conv1d = torch.nn.Conv1d(4, 1, kernel_size=pwm.shape[1])
    conv1d.bias.data = torch.Tensor([0.])
    conv1d.weight.data = torch.tensor(np.array([pwm]), dtype=torch.float32)
    for p in conv1d.parameters():
        p.requires_grad = False
    maxpool1d = torch.nn.MaxPool1d(sequence_length - pwm.shape[1] + 1, 1,
        return_indices=True)

    return torch.nn.Sequential(conv1d, maxpool1d).to(device)


def _score_sequences(model, sequences_tensor, sequences, motif_length):
    """
    Score each sequence by its best site on either strand.

    Args:
        model: Model from _get_pwm_model.
        sequences_tensor: One-hot sequences, forward strands first and
            reverse complements after (as built in damo).
        sequences: Sequence strings in the same order as the tensor.
        motif_length: Length of the PWM.

    Returns:
        Tuple (scores, sites): best score per sequence, and the matching
        site string.
    """
    # Get predictions
    with torch.no_grad():
        scores_tmp, sites_tmp = model(sequences_tensor)
    scores_tmp = scores_tmp.flatten()
    sites_tmp = sites_tmp.flatten()

    # Forward strands come first, reverse complements after. Ties go to the
    # reverse strand, as in DAMO.
    half = len(sequences) // 2
    use_fwd = scores_tmp[:half] > scores_tmp[half:]
    scores = torch.where(use_fwd, scores_tmp[:half], scores_tmp[half:]).tolist()
    positions = torch.where(use_fwd, sites_tmp[:half], sites_tmp[half:]).tolist()
    use_fwd = use_fwd.tolist()

    # Get the site strings
    sites = []
    for i in range(half):
        seq = sequences[i] if use_fwd[i] else sequences[half + i]
        sites.append(seq[positions[i]:positions[i] + motif_length])

    return scores, sites


def _compute_auc(positive_scores, negative_scores):
    """
    ROC AUC of the scores, with positives labelled 1.

    Returns:
        Tuple (auc, labels, scores), where labels and scores are the
        concatenated inputs.
    """
    y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
    y_score = positive_scores + negative_scores

    return metrics.roc_auc_score(y_true, y_score), y_true, y_score


def _update_pwm(y_true, y_score, sites, lr, pwm):
    """
    One update step, as DAMO's makeAMove.

    Sites that rank between the first negative and the last positive (the
    overlap zone) are turned into a positive and a negative PFM. The PWM
    moves by lr times their log-odds difference.

    Args:
        y_true: Labels, 1 for positive and 0 for negative.
        y_score: Scores, same order as y_true.
        sites: Best site string per sequence, same order.
        lr: Learning rate.
        pwm: Current log PWM of shape (4, motif length).

    Returns:
        Updated log PWM, or None if the classes are perfectly separated.
    """
    positive_mix = []
    negative_mix = []

    # Sort and zip sequences and labels
    idxs = np.argsort(-np.asarray(y_score), kind="mergesort")
    sequences = [sites[ix] for ix in idxs]
    labels = [y_true[ix] for ix in idxs]
    z = list(zip(sequences, labels))

    # Get index of first negative and last positive
    ix_first_negative = labels.index(0)
    ix_last_positive = len(labels) - labels[::-1].index(1)

    # Perfect separation: no positive ranks below a negative
    if ix_first_negative >= ix_last_positive:
        return None

    # For each label, sequence...
    for s, l in z[ix_first_negative:ix_last_positive]:
        positive_mix.append(s) if l == 1 else negative_mix.append(s)

    positive_pwm = _generate_pwm(positive_mix, pwm.shape[1])
    negative_pwm = _generate_pwm(negative_mix, pwm.shape[1])

    return pwm + lr * (positive_pwm - negative_pwm)


def _generate_pwm(sequences, motif_length):
    """
    Log PWM from a set of aligned sites, as DAMO's gen_pwm.

    Counts are turned into frequencies, then 1e-4 is added and the result
    is renormalized. Without that, an absent base would get
    log(1e-4 / n) and make the steps too large.

    Args:
        sequences: Sites of equal length.
        motif_length: Length of the sites.

    Returns:
        Array of shape (4, motif length), rows A, C, G and T.
    """
    pfm = np.zeros((motif_length, 4))
    for i in range(motif_length):
        c = Counter(s[i:i+1] for s in sequences)
        pfm[i, :] = [c[key] for key in "ACGT"]
    pfm = _normalize_pfm(_normalize_pfm(pfm), 1e-4)

    return np.log(pfm).T


def _normalize_pfm(pfm, pseudocounts=0.):
    """Add pseudocounts to each row and divide it by its sum."""
    pfm = np.array(pfm, float) + pseudocounts

    return pfm / np.sum(pfm, axis=1, keepdims=True)
