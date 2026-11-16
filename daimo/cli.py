"""Command line interface."""

import sys

import click

from daimo.damo import (DEFAULT_SCALE, _parse_width_ranges,
                        damo, fit_scale, pwm_to_pfm)
from daimo.io import (open_file, read_motif, read_sequences, write_jaspar,
                       write_meme, write_pwm)

CONTEXT_SETTINGS = {
    "help_option_names": ["-h", "--help"],
}


def _check_widths(value):
    """Validate the syntax of --widths; open-ended ranges are expanded later."""
    if value is not None:
        try:
            _parse_width_ranges(value)
        except ValueError as e:
            raise click.BadParameter(str(e)) from None

    return value


class Scale(click.ParamType):
    """A positive number, or "auto" (any case)."""

    name = "number|auto"

    def convert(self, value, param, ctx):
        if isinstance(value, str) and value.strip().lower() == "auto":
            return "auto"
        try:
            scale = float(value)
        except (TypeError, ValueError):
            scale = None
        if scale is None or not 0 < scale < float("inf"):
            self.fail(f"{value!r} is not a positive number or 'auto'",
                      param, ctx)

        return scale


def _fit_scale(pwm, positive_seqs_file, negative_seqs_file, debugging):
    """
    Fit the scale of the PWM to the sequences that `damo` was run on.

    Returns:
        Tuple (scale, report): the fitted scale and a line that reports it.
    """
    positives, negatives = (
        [str(record.seq) for record in read_sequences(file, debugging)]
        for file in (positive_seqs_file, negative_seqs_file))
    scale, details = fit_scale(pwm, positives, negatives, return_details=True)
    if details["sequences"] == 0:
        return scale, (f"Scale: {scale:g} (the PWM is flat, so there is "
                       "nothing to fit).")
    report = (f"Fitted scale: {scale:.4g} (fraction of sequences with a "
              f"site: {details['fraction_with_site']:.2f}).")
    if details["at_edge"]:
        report += (" Warning: the fit ended at the edge of its search "
                   "interval, so this scale may not be the best one.")

    return scale, report


@click.command(no_args_is_help=True, context_settings=CONTEXT_SETTINGS)
@click.argument(
    "positive_seqs",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "negative_seqs",
    type=click.Path(exists=True, resolve_path=True),
)
@click.argument(
    "seed_motif",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-d", "--debugging",
    help="Use only the first 1,000 sequences of each file.",
    is_flag=True,
)
@click.option(
    "-i", "--iterations",
    help="Number of optimization iterations.",
    type=int,
    default=500,
    show_default=True,
)
@click.option(
    "-o", "--output-file",
    help="Output file (.gz to compress). [default: stdout]",
    type=click.Path(resolve_path=True),
)
@click.option(
    "-w", "--widths",
    help="Optimize the motif width: try several widths, given as a list in "
         "the style of cut -f (8,10,12 or 5-19 or 5-8,12,15-; -M is 1 to M, "
         "N- is N up to the sequence length), and keep the one that does "
         "best on held-out sequences. "
         "[default: the width of the seed motif]",
    callback=lambda ctx, param, value: _check_widths(value),
)
@click.option(
    "-m", "--motif",
    help="Motif to use when the MEME file has several (name, or its ID).",
)
@click.option(
    "--oformat",
    help="Output format: the log PWM (one line per position, A C G T), or "
         "the motif as frequencies in JASPAR or MEME format.",
    type=click.Choice(["pwm", "jaspar", "meme"]),
    default="pwm",
    show_default=True,
)
@click.option(
    "--scale",
    help="With --oformat jaspar or meme: factor applied to the PWM before "
         "converting it to frequencies; above 1 sharpens the columns, "
         "below 1 flattens them. auto fits it to the positive sequences, "
         "so the sharpness of the motif does not depend on the size of the "
         "weights; 0.1 gives the output of daimo 1.0.1; 1 is DiMO's "
         "unscaled conversion and gives nearly one-hot columns.",
    type=Scale(),
    default=DEFAULT_SCALE,
    show_default=True,
)
def main(**args):
    """Optimize a JASPAR or MEME motif to separate positive from negative sequences.

    The input format is detected from the file. The output is the log PWM
    by default, with one line per position and four columns (A, C, G, T).
    With --oformat jaspar or meme it is a motif of frequencies instead: the
    PWM weights are read as energies (kT) and converted with a Boltzmann
    distribution, as in DiMO, after scaling them. By default the scale is
    fitted to the positive sequences (see --scale).
    """
    ctx = click.get_current_context()
    scale_given = (ctx.get_parameter_source("scale")
                   != click.core.ParameterSource.DEFAULT)
    if scale_given and args["oformat"] == "pwm":
        raise click.UsageError(
            "--scale only applies with --oformat jaspar or meme")

    pwm = damo(args["positive_seqs"], args["negative_seqs"],
        args["seed_motif"], args["debugging"],
        iterations=args["iterations"], motif_name=args["motif"],
        widths=args["widths"])

    fit_report = None
    if args["oformat"] == "pwm":
        def write(handle):
            write_pwm(pwm, handle)
    else:
        name, _ = read_motif(args["seed_motif"], args["motif"])
        scale = args["scale"]
        if scale == "auto":
            scale, fit_report = _fit_scale(pwm, args["positive_seqs"],
                args["negative_seqs"], args["debugging"])
        pfm = pwm_to_pfm(pwm, scale)
        writer = write_jaspar if args["oformat"] == "jaspar" else write_meme

        def write(handle):
            writer(pfm, handle, name)

    if args["output_file"] is None:
        write(sys.stdout)
        sys.stdout.flush()
    else:
        with open_file(args["output_file"], "w") as handle:
            write(handle)

    # After the motif, and on stderr, so that it stays out of the motif
    if fit_report is not None:
        click.echo(fit_report, err=True)
