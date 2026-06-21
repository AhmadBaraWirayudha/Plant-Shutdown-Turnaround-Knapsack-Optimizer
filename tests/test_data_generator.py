"""
test_data_generator.py — Regression tests for the synthetic CMMS data
generator, specifically targeting a class of bug found during development:

A module-level `rng = np.random.default_rng(DGEN_CFG.random_seed)` and
function-signature defaults like `n: int = DGEN_CFG.num_work_orders` are
both evaluated exactly ONCE — at import time — regardless of when a CLI
flag or other code later mutates DGEN_CFG. This was empirically confirmed
to silently break `--num-work-orders` entirely and `--seed` partially
(asset master and failure history generation ignored it; only work-order
sampling happened to read DGEN_CFG.random_seed live). The fix threads an
explicit `rng: np.random.Generator` parameter through every generator
function instead of relying on a frozen module global, and resolves
`n`/`mandatory_frac`/`pred_frac` from the LIVE config via `None` sentinels
resolved inside the function body rather than baked into the signature.

Every test below calls the generator with two DIFFERENT explicit
parameters in the SAME process and asserts the output actually differs —
that's the exact pattern that would have caught the original bug
immediately, instead of "it doesn't error" which is what a less careful
test would have checked.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from src.utils.config import DGEN_CFG
from src.utils.data_generator import (
    generate_asset_master,
    generate_failure_history,
    generate_work_orders,
    generate_all,
)


class TestExplicitRngThreading:
    """generate_asset_master / generate_failure_history must use the rng
    object passed to THEM, not any frozen global — same seed in, same
    output out; different seed in, different output out."""

    def test_asset_master_is_deterministic_given_same_seed(self):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        df1 = generate_asset_master(rng1)
        df2 = generate_asset_master(rng2)
        pd.testing.assert_frame_equal(df1, df2)

    def test_asset_master_differs_given_different_seeds(self):
        df1 = generate_asset_master(np.random.default_rng(1))
        df2 = generate_asset_master(np.random.default_rng(2))
        # Same assets, same classes, same counts — but different random
        # install dates / areas. The `area` column is the simplest robust
        # signal since it's a small categorical drawn directly from rng.
        assert not df1["area"].equals(df2["area"])

    def test_failure_history_differs_given_different_seeds(self):
        assets = generate_asset_master(np.random.default_rng(1))
        f1 = generate_failure_history(assets, np.random.default_rng(10))
        f2 = generate_failure_history(assets, np.random.default_rng(20))
        # Row counts (num_failures per asset) and/or values should differ
        assert len(f1) != len(f2) or not f1["time_to_failure_d"].equals(f2["time_to_failure_d"])

    def test_work_orders_differ_given_different_rng_state(self):
        assets = generate_asset_master(np.random.default_rng(1))
        wos1 = generate_work_orders(assets, np.random.default_rng(10), n=50)
        wos2 = generate_work_orders(assets, np.random.default_rng(20), n=50)
        assert not wos1["asset_tag"].equals(wos2["asset_tag"])


class TestNoFrozenModuleGlobal:
    """Direct regression test for the original bug: there must be no
    module-level `rng` left over that any function could fall back to."""

    def test_module_has_no_frozen_rng_global(self):
        import src.utils.data_generator as dg

        assert not hasattr(dg, "rng"), (
            "A module-level 'rng' global was reintroduced — this is exactly "
            "the pattern that caused --seed to silently only partially work "
            "(asset master / failure history ignored CLI overrides). Thread "
            "rng through as an explicit parameter instead."
        )


class TestNoneSentinelResolvesLiveConfig:
    """
    Regression tests for the --num-work-orders bug: generate_work_orders'
    n/mandatory_frac/pred_frac must resolve DGEN_CFG at CALL time, not
    whatever DGEN_CFG happened to be when this module was first imported.
    Simulated here by mutating DGEN_CFG mid-test (exactly what the CLI
    does) and confirming the very next call picks up the new value.
    """

    def test_n_resolves_live_dgen_cfg_when_omitted(self):
        assets = generate_asset_master(np.random.default_rng(1))
        original = DGEN_CFG.num_work_orders
        try:
            DGEN_CFG.num_work_orders = 37
            wos = generate_work_orders(assets, np.random.default_rng(5))  # n omitted
            assert len(wos) == 37
        finally:
            DGEN_CFG.num_work_orders = original

    def test_n_changes_between_two_calls_in_same_process(self):
        """The exact scenario that was broken: two sequential calls in one
        process with DGEN_CFG mutated in between must produce DIFFERENT
        row counts, not both silently using whatever was frozen at import."""
        assets = generate_asset_master(np.random.default_rng(1))
        original = DGEN_CFG.num_work_orders
        try:
            DGEN_CFG.num_work_orders = 20
            wos_a = generate_work_orders(assets, np.random.default_rng(5))
            DGEN_CFG.num_work_orders = 80
            wos_b = generate_work_orders(assets, np.random.default_rng(5))
            assert len(wos_a) == 20
            assert len(wos_b) == 80
            assert len(wos_a) != len(wos_b)
        finally:
            DGEN_CFG.num_work_orders = original

    def test_explicit_n_overrides_dgen_cfg(self):
        """An explicit n= argument must take precedence over DGEN_CFG
        regardless of what DGEN_CFG currently holds."""
        assets = generate_asset_master(np.random.default_rng(1))
        original = DGEN_CFG.num_work_orders
        try:
            DGEN_CFG.num_work_orders = 999  # should be ignored below
            wos = generate_work_orders(assets, np.random.default_rng(5), n=15)
            assert len(wos) == 15
        finally:
            DGEN_CFG.num_work_orders = original

    def test_mandatory_frac_resolves_live_config(self):
        assets = generate_asset_master(np.random.default_rng(1))
        original = DGEN_CFG.mandatory_fraction
        try:
            DGEN_CFG.mandatory_fraction = 0.5
            wos = generate_work_orders(assets, np.random.default_rng(5), n=40)
            # mandatory_count = max(1, int(n * mandatory_frac)) = max(1, int(40*0.5)) = 20
            assert wos["mandatory"].sum() == 20
        finally:
            DGEN_CFG.mandatory_fraction = original


class TestZeroOrNegativeNRaisesCleanError:
    """
    Regression tests for a real bug: --num-work-orders 0 used to produce an
    empty, columnless DataFrame (pd.DataFrame([]) from a zero-iteration
    loop) that crashed much later, deep inside pandas, with a confusing
    `KeyError: 'priority'` traceback exposed directly to the CLI user.
    generate_work_orders now validates n up front and raises a clear,
    actionable ValueError instead.
    """

    def test_n_zero_raises_value_error(self):
        assets = generate_asset_master(np.random.default_rng(1))
        with pytest.raises(ValueError, match="positive integer"):
            generate_work_orders(assets, np.random.default_rng(5), n=0)

    def test_negative_n_raises_value_error(self):
        assets = generate_asset_master(np.random.default_rng(1))
        with pytest.raises(ValueError, match="positive integer"):
            generate_work_orders(assets, np.random.default_rng(5), n=-5)

    def test_n_zero_via_live_dgen_cfg_also_raises_value_error(self):
        """The same validation must apply whether n arrives via an explicit
        argument or via the live-resolved DGEN_CFG.num_work_orders sentinel
        path — both go through the same code path, but worth confirming
        directly since this is exactly the path the CLI actually uses."""
        assets = generate_asset_master(np.random.default_rng(1))
        original = DGEN_CFG.num_work_orders
        try:
            DGEN_CFG.num_work_orders = 0
            with pytest.raises(ValueError, match="positive integer"):
                generate_work_orders(assets, np.random.default_rng(5))  # n omitted
        finally:
            DGEN_CFG.num_work_orders = original

    def test_positive_n_does_not_raise(self):
        assets = generate_asset_master(np.random.default_rng(1))
        wos = generate_work_orders(assets, np.random.default_rng(5), n=1)
        assert len(wos) == 1


class TestGenerateAllSeedResolution:
    """generate_all() must create its rng from the LIVE DGEN_CFG.random_seed
    at call time (or an explicit override), and writes must reflect that
    seed deterministically — not whatever was frozen at import."""

    def test_explicit_seed_overrides_dgen_cfg_and_is_deterministic(self, monkeypatch, tmp_path):
        import src.utils.data_generator as dg

        monkeypatch.setattr(dg, "DATA_RAW", tmp_path)
        assets1, failures1, wos1 = generate_all(seed=555, n_work_orders=25)
        assets2, failures2, wos2 = generate_all(seed=555, n_work_orders=25)
        pd.testing.assert_frame_equal(assets1, assets2)
        pd.testing.assert_frame_equal(wos1, wos2)

    def test_different_explicit_seeds_produce_different_output(self, monkeypatch, tmp_path):
        import src.utils.data_generator as dg

        monkeypatch.setattr(dg, "DATA_RAW", tmp_path)
        assets1, _, wos1 = generate_all(seed=1, n_work_orders=25)
        assets2, _, wos2 = generate_all(seed=2, n_work_orders=25)
        assert not assets1["area"].equals(assets2["area"])
        assert not wos1["asset_tag"].equals(wos2["asset_tag"])

    def test_two_sequential_calls_with_different_dgen_cfg_seed_differ(self, monkeypatch, tmp_path):
        """Replicates the exact original bug scenario end-to-end: mutate
        DGEN_CFG.random_seed between two calls (what --seed does via the
        CLI) and confirm the second call's output actually changes."""
        import src.utils.data_generator as dg

        monkeypatch.setattr(dg, "DATA_RAW", tmp_path)
        original_seed = DGEN_CFG.random_seed
        try:
            DGEN_CFG.random_seed = 111
            assets_a, _, _ = generate_all(n_work_orders=20)  # seed omitted -> reads live DGEN_CFG
            DGEN_CFG.random_seed = 222
            assets_b, _, _ = generate_all(n_work_orders=20)
            assert not assets_a["area"].equals(assets_b["area"])
        finally:
            DGEN_CFG.random_seed = original_seed

    def test_n_work_orders_override_controls_row_count(self, monkeypatch, tmp_path):
        import src.utils.data_generator as dg

        monkeypatch.setattr(dg, "DATA_RAW", tmp_path)
        _, _, wos = generate_all(seed=1, n_work_orders=33)
        assert len(wos) == 33

    def test_writes_csvs_to_data_raw(self, monkeypatch, tmp_path):
        import src.utils.data_generator as dg

        monkeypatch.setattr(dg, "DATA_RAW", tmp_path)
        generate_all(seed=1, n_work_orders=10)
        assert (tmp_path / "asset_master.csv").exists()
        assert (tmp_path / "failure_history.csv").exists()
        assert (tmp_path / "work_orders.csv").exists()


class TestNegativeSeedRaisesCleanError:
    """
    Regression test for a real bug: a negative --seed value used to crash
    with numpy's raw internal error ("expected non-negative integer"),
    exposed directly to the CLI user with no context about which argument
    caused it or why. generate_all() now validates the resolved seed up
    front and raises a clear, application-level ValueError instead.
    """

    def test_negative_explicit_seed_raises_value_error(self, monkeypatch, tmp_path):
        import src.utils.data_generator as dg

        monkeypatch.setattr(dg, "DATA_RAW", tmp_path)
        with pytest.raises(ValueError, match="non-negative"):
            generate_all(seed=-5, n_work_orders=10)

    def test_negative_seed_via_live_dgen_cfg_also_raises(self, monkeypatch, tmp_path):
        import src.utils.data_generator as dg

        monkeypatch.setattr(dg, "DATA_RAW", tmp_path)
        original = DGEN_CFG.random_seed
        try:
            DGEN_CFG.random_seed = -1
            with pytest.raises(ValueError, match="non-negative"):
                generate_all(n_work_orders=10)  # seed omitted -> resolves live DGEN_CFG
        finally:
            DGEN_CFG.random_seed = original

    def test_zero_seed_is_valid(self, monkeypatch, tmp_path):
        """Zero is a perfectly valid seed — only negative values are rejected."""
        import src.utils.data_generator as dg

        monkeypatch.setattr(dg, "DATA_RAW", tmp_path)
        assets, _, wos = generate_all(seed=0, n_work_orders=10)
        assert len(wos) == 10
