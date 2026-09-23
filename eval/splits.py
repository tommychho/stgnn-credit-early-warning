"""Train/validation/test split construction: ONE definition, shared by every notebook.

Why this module exists
----------------------
The published split has no embargo. Validation ends 2019-12-27 and the test window opens
2019-01-03, while ``company.y`` is ``has_default_in_window(t, 52w)``. A firm defaulting in
early 2020 therefore carries ``y == 1`` on validation snapshots, and those labels were
visible to early stopping and to threshold calibration.

Measured on this panel (``nb08 [Cell V10]``): **1,030 of 3,059 positive validation
firm-snapshots, 33.7%, belong to firms defaulting on or after the test boundary**, across
35 distinct firms. 35 of the 96 panel firms defaulting in the test period were already
carrying positive labels while the model was being selected.

Retraining showed the leak was *visible but not exploited* (``ARCHITECTURE_EXPERIMENT.md``
§15.2a): at matched validation strength the leak-free arm scores no worse. The primary
split should nonetheless not carry a known defect, which is what ``shifted_split`` fixes.

Truncate or shift?
------------------
Two remedies remove the leak and they are not equivalent.

``truncate``  cut the last ``horizon`` snapshots off validation. Removes the leak, but pays
              for it out of the validation set: 156 snapshots to 104, and 3,059 positives to
              1,351. Test AP falls by roughly 0.11, and a size control shows **most of that
              fall is the shrinkage, not the leak**.

``shift``     slide the whole validation window back one ``horizon`` and pay out of TRAINING
              data instead: 835 training snapshots to 783, a 6.2% reduction, with validation
              kept at full length. This is the remedy to use.

    published : train 0..834 (835) | val 835..990 (156) | test 991..   val ends where test starts
    shifted   : train 0..782 (783) | val 783..938 (156) | test 991..   52w gap, validation intact

THE LANDMINE
------------
``run_split`` in the notebooks **ignores its ``graphs`` argument except for
``len(graphs)``**. It iterates ``all_graphs[start_idx : start_idx + len(graphs)]``. So:

* a non-contiguous subsample is silently replaced by a contiguous block, and the cell
  reports a different experiment than the one it describes;
* rebinding ``val_graphs`` without also rebinding ``val_start`` scores the wrong snapshots.

This cost one rewrite already: the first version of the size control in
``nb03 [Cell 12embC]`` would have re-run the embargoed arm and labelled it the control.
Every function here therefore returns ``start_idx`` alongside the list, and callers must
move both together. :func:`assert_contiguous` checks it.

Usage
-----
    import src.eval.splits as splits
    spec = splits.shifted_split(all_graphs, len(train_graphs), len(val_graphs))
    splits.assert_contiguous(all_graphs, spec)
    n = splits.audit_leak(all_graphs, spec, gvkey_to_defdate, cid_to_gvkey)   # must be 0
    # gvkey_to_defdate must be the FULL history, not a map filtered to the test period.

    train_graphs, train_start = spec.train, spec.train_start
    val_graphs,   val_start   = spec.val,   spec.val_start
"""

from typing import Dict, List, NamedTuple, Optional, Sequence

__all__ = [
    "SplitSpec", "published_split", "shifted_split", "truncated_split",
    "assert_contiguous", "audit_leak", "count_positives",
]

DEFAULT_HORIZON = 52


class SplitSpec(NamedTuple):
    """A split, with the start index each list maps to in ``all_graphs``.

    ``start`` is not decoration: ``run_split`` indexes ``all_graphs`` directly and uses
    the list only for its length, so the pair must always travel together.
    """

    train: List
    val: List
    test: List
    train_start: int
    val_start: int
    test_start: int
    name: str = "unnamed"

    def describe(self) -> str:
        return (
            f"{self.name}: train {self.train_start}..{self.train_start + len(self.train) - 1} "
            f"({len(self.train)}) | val {self.val_start}..{self.val_start + len(self.val) - 1} "
            f"({len(self.val)}) | test {self.test_start}.. ({len(self.test)})"
        )


def published_split(all_graphs: Sequence, n_train: int, n_val: int) -> SplitSpec:
    """The split as originally cut. Carries the validation-to-test leak; kept so that
    published numbers remain regenerable, exactly as ``detection.py`` keeps
    ``mode='label_onset'``."""
    return SplitSpec(
        train=list(all_graphs[:n_train]),
        val=list(all_graphs[n_train:n_train + n_val]),
        test=list(all_graphs[n_train + n_val:]),
        train_start=0, val_start=n_train, test_start=n_train + n_val,
        name="published",
    )


def shifted_split(all_graphs: Sequence, n_train: int, n_val: int,
                  horizon: int = DEFAULT_HORIZON) -> SplitSpec:
    """Validation slid back one label horizon; the embargo is paid out of training data.

    A snapshot at index ``i`` carries a label covering ``(i, i + horizon]``. The last safe
    validation index is therefore ``test_start - horizon - 1``, whose label stops one step
    short of the test window. Getting this off by one leaves a single leaking snapshot, so
    it is asserted rather than trusted.
    """
    if horizon < 1:
        raise ValueError(
            f"horizon={horizon} would return the published split under a 'shifted' label. "
            f"horizon must equal the label horizon used to build company.y (here {DEFAULT_HORIZON})."
        )

    test_start = n_train + n_val
    last_safe = test_start - horizon - 1
    new_val_start = last_safe - n_val + 1

    if last_safe + horizon >= test_start:
        raise ValueError(
            f"shifted validation still reaches the test window: last safe index {last_safe} "
            f"+ horizon {horizon} >= test_start {test_start}"
        )
    if new_val_start <= 0:
        raise ValueError(
            f"not enough history to shift validation back a full horizon: would start at "
            f"{new_val_start}. Panel has {len(all_graphs)} snapshots."
        )

    return SplitSpec(
        train=list(all_graphs[:new_val_start]),
        val=list(all_graphs[new_val_start:last_safe + 1]),
        test=list(all_graphs[test_start:]),
        train_start=0, val_start=new_val_start, test_start=test_start,
        name="shifted",
    )


def truncated_split(all_graphs: Sequence, n_train: int, n_val: int,
                    horizon: int = DEFAULT_HORIZON) -> SplitSpec:
    """Validation cut short by one horizon. Removes the leak but costs a third of the
    validation set and more than half its positives; retained only to reproduce the
    diagnostic arm of §15.2a. **Do not use as a primary split.**"""
    keep = n_val - horizon
    if keep <= 0:
        raise ValueError(f"horizon {horizon} would leave {keep} validation snapshots")
    return SplitSpec(
        train=list(all_graphs[:n_train]),
        val=list(all_graphs[n_train:n_train + keep]),
        test=list(all_graphs[n_train + n_val:]),
        train_start=0, val_start=n_train, test_start=n_train + n_val,
        name="truncated",
    )


def assert_contiguous(all_graphs: Sequence, spec: SplitSpec) -> None:
    """Every list must map to ``all_graphs`` at its declared start index.

    Guards the landmine in the module docstring: ``run_split`` will happily score a
    contiguous block that has nothing to do with the list it was handed.
    """
    for name, graphs, start in (("train", spec.train, spec.train_start),
                                ("val", spec.val, spec.val_start),
                                ("test", spec.test, spec.test_start)):
        for i, g in enumerate(graphs):
            if all_graphs[start + i] is not g:
                raise AssertionError(
                    f"{spec.name}: {name}[{i}] is not all_graphs[{start + i}]. The split is "
                    f"not contiguous at its start index, so run_split would score the wrong "
                    f"snapshots. See the LANDMINE note in src/eval/splits.py."
                )


def count_positives(graphs: Sequence) -> int:
    """Positive firm-snapshots in a split. The quantity that drives validation-AP noise,
    and the one to match on when comparing split regimes."""
    return int(sum(int(g["company"].y.sum()) for g in graphs))


def audit_leak(all_graphs: Sequence, spec: SplitSpec,
               gvkey_to_defdate: Dict[str, object],
               cid_to_gvkey: Dict[int, str],
               verbose: bool = True) -> int:
    """Count validation positives caused by a default at or after the test boundary.

    Attribution is from the default DATES rather than from ``y``, so the audit does not
    depend on trusting the label construction it is auditing. Returns the count; the
    caller should assert it is zero for any split used as primary.

    ``gvkey_to_defdate`` MUST BE THE FULL DEFAULT HISTORY, unfiltered, as built by
    ``_defs.groupby('gvkey')['default_date'].min()``. A map pre-filtered to the test
    period, which is what ``build_event_anchors`` wants and what several cells build for
    that purpose, makes every firm this function finds satisfy ``dd >= test_ts`` by
    construction, so the audit reports 100% leak on a split that has none. That is checked
    below rather than documented and hoped for.
    """
    test_ts = all_graphs[spec.test_start]["company"].t

    if gvkey_to_defdate:
        earliest = min(gvkey_to_defdate.values())
        if earliest >= test_ts:
            raise ValueError(
                f"gvkey_to_defdate contains no default before the test boundary "
                f"({test_ts}); its earliest is {earliest}. This is a map pre-filtered to "
                f"the test period, which is what build_event_anchors needs but NOT what "
                f"this audit needs: every firm found in it would satisfy dd >= test_ts by "
                f"construction and the leak count would be meaningless. Pass the full "
                f"history, _defs.groupby('gvkey')['default_date'].min()."
            )
    leaked, attributable, total = 0, 0, 0
    firms = set()
    for g in spec.val:
        cids = g["company"].company_ids.cpu().numpy()
        ys = g["company"].y.cpu().numpy()
        for cid, y in zip(cids, ys):
            if int(y) != 1:
                continue
            total += 1
            dd = gvkey_to_defdate.get(cid_to_gvkey.get(int(cid), ""))
            if dd is None:
                continue
            attributable += 1
            if dd >= test_ts:
                leaked += 1
                firms.add(int(cid))
    if verbose:
        print(f"  leak audit [{spec.name}]: {leaked:,} of {attributable:,} attributable "
              f"validation positives reference a test-period default "
              f"({len(firms)} firms; {total:,} positives total)")
        if total > attributable:
            print(f"    note {total - attributable:,} positives have no default date on "
                  f"record and are excluded, so this is a LOWER bound")
    return leaked
