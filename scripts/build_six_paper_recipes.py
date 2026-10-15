"""Materialize the reviewed mappings; this is not arbitrary prose-to-code inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from paperquant.papers import MethodSpec, Recipe, verify_recipe
from scripts.build_six_paper_inputs import CLASSES, package, save

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/six-paper"
TEST = "tests/test_six_paper_methods.py::"

MAPPINGS = [
    (
        "graph_matching",
        "2403.07998v1",
        [
            (
                13,
                "single-lag Augmented Dickey-",
                "data",
                "adf_statistic",
                "test_single_lag_adf_matches_numpy_ols_and_standard_error",
                "Section 4.1, Eq18: log-price OLS residuals and single-lag ADF edge weights",
                "Exact intercept-bearing lag-one ADF t-statistic; use its negative as graph weight",
            ),
            (
                14,
                "maximum weight matching",
                "signal",
                "maximum_matching",
                "test_matching_is_global_optimum_and_has_no_shared_assets",
                "Section 4.1: jointly maximize selected pair weights with no repeated assets",
                "Exact subset dynamic programming on a bounded four-asset universe",
            ),
            (
                15,
                "winsorized",
                "inference",
                "decide",
                "test_graph_trading_signal_matches_independent_log_regression",
                "Section 4.2: daily trailing regression, winsorized z-score and beta-dollar hedge",
                (
                    "504 prior sessions, calendar-month pair refresh and "
                    "next-open non-atomic hedge fills"
                ),
            ),
        ],
        [
            "Four generated assets replace the S&P500 universe and Yahoo 2017-2023 sample.",
            (
                "Choose the z-score variant only; q-score variant, original"
                " costs and performance experiments excluded."
            ),
            (
                "Page6 Eq3 and following prose have opposite sign "
                "descriptions; execution follows Section4.2 economic "
                "direction."
            ),
            (
                "Page15 calls the statistic a z-score but says variance; "
                "divide by residual standard deviation."
            ),
            (
                "Subset-DP has the exact maximum-weight objective but is "
                "not the scalable blossom implementation."
            ),
            "Hedge legs fill separately at subsequent symbol opens; no atomic paired fill claim.",
            "Target quantities are explicitly rounded to eight decimal places.",
        ],
    ),
    (
        "ou_percentile",
        "2412.12458v1",
        [
            (
                3,
                "autoregressive model",
                "training",
                "calibrate",
                "test_ou_calibration_matches_independent_ar1_equations",
                (
                    "OU Model Integration, Eq8-10: fit AR1 and convert "
                    "a,b,residual SD to OU parameters"
                ),
                (
                    "Fit training-only spreads, reject a outside (0,1), "
                    "preserve lambda/mu/sigma equations"
                ),
            ),
            (
                4,
                "90 day window",
                "inference",
                "decide",
                "tests/test_six_audit_refinement.py::test_ou_complete_ro"
                "lling_entry_hold_and_median_exit",
                (
                    "Signal Generation: PIT 25th/75th entry and crossing 50th "
                    "exit, executed following day"
                ),
                (
                    "Use prior 90 observed z-scores and frozen training OU fit;"
                    " submit both next-open hedge targets"
                ),
            ),
        ],
        [
            (
                "Preselected generated pair replaces WRDS/Datastream, "
                "distance ranking and manual industry selection."
            ),
            (
                "Eq7 contains a printed exponent inconsistent with Eq9-10; "
                "calibration follows internally consistent Eq10."
            ),
            (
                "Training 504 controlled sessions and evaluation 296 "
                "sessions replace the original historical universe."
            ),
            (
                "No original return, Sharpe, factor neutrality or empirical"
                " pair-selection reproduction claim."
            ),
        ],
    ),
    (
        "grouped_cnn_lstm",
        "2205.08382v1",
        [
            (
                11,
                "three separate lines of convolutional layers",
                "training",
                "forward",
                "test_cnn_training_updates_all_model_stages_and_confidence_equations",
                (
                    "Section2.2: three feature-group Conv1d branches, pooling, "
                    "many-to-one LSTM and classification head"
                ),
                (
                    "Two reduced convolutional autoencoders; "
                    "24-to-20-to-10-to-8 branches; configurable LSTM and MLP width"
                ),
            ),
            (
                14,
                "many-to-one LSTM",
                "training",
                "forward",
                "test_convolution_and_lstm_match_separate_numpy_equations",
                "Eq4-9: candidate, input/forget/output gates, cell and hidden-state recursion",
                (
                    "Shared LSTM parameters trained by reverse-mode "
                    "differentiation through all eight sequence states"
                ),
            ),
            (
                15,
                "binary cross-entropy loss function",
                "training",
                "train",
                "test_autodiff_gradients_match_independent_math_and_finite_differences",
                "Eq11-13: supervised BCE and loss-based training quality proxy",
                (
                    "Natural-log BCE; separately report printed 10^-loss proxy "
                    "and whether paper quality threshold is met"
                ),
            ),
            (
                20,
                "two thresholds",
                "execution",
                "decide",
                "test_confidence_and_bet_size_mapping_uses_real_probability_boundaries",
                (
                    "Section3, Eq15-16: confidence thresholds from risk/reward "
                    "and margin fraction abs(p-.5)"
                ),
                (
                    "Declared theta=.5 thresholds; signed target fraction of "
                    "current equity; neutral closes inventory"
                ),
            ),
        ],
        [
            "Public BTC/USDT Jan-Jun2024 training, July-September development validation, "
            "independent January-March2025 test; prior August-"
            "September and October-December negative results "
            "retained.",
            (
                "Wavelet denoising, XGBoost feature selection, large "
                "indicator grid and dropout are excluded."
            ),
            (
                "Convolutional autoencoders omit the original "
                "encoder/decoder pool-upsampling stacks; latent channels "
                "are reduced."
            ),
            (
                "Configurable bounded LSTM/FC widths replace original widths20/10; "
                "Conv outputs reduced."
            ),
            (
                "Training uses minibatch Adam, training-only standard or"
                " MinMax scaler, 12-epoch AE pretraining, "
                "frozen encoder/CNN stages and validation checkpoint selection. ReLU replaces "
                "the earlier main tanh activation; BatchNorm remains excluded. Bounded stages "
                "do not reproduce the paper's convergence loop."
            ),
            (
                "Training quality threshold is reported honestly; runtime "
                "pass is not a well-trained or profitable-model assertion."
            ),
            (
                "Printed 10^-BCE proxy uses a different log base from "
                "conventional BCE; preserve it as a named paper proxy."
            ),
            (
                "Theta=.5, next-open fills and per-fill costs are declared "
                "execution adaptations; no SL/TP bracket engine."
            ),
            (
                "OHLC/price-like features use causal window-relative price normalization; "
                "volume uses window mean."
            ),
            (
                "Analytic convolution/LSTM backpropagation matches the independent "
                "Scalar autodiff path and finite differences."
            ),
            (
                "Validation influences checkpoint selection only, never gradients "
                "or scaler fitting; test is excluded from training request."
            ),
            "Causal 64-bar EMA/Williams and SMA/volatility alternatives were evaluated on "
            "development validation but were not selected; this is explicit model selection.",
            "Selected four-unit configuration uses cached frozen-stage features and numerical "
            "dot products. Training has no default fixed wall-clock deadline; callers may choose "
            "a finite budget. Load and inference retain 20-second deadlines; memory, CPU, import, "
            "network and filesystem restrictions are unchanged.",
        ],
    ),
    (
        "gmm_knn",
        "2409.03762v1",
        [
            (
                6,
                "first 59 minutes",
                "data",
                "features",
                "test_59_minute_features_match_independent_peak_regression_and_have_no_future_slot",
                (
                    "Methods: training 4th-percentile log-return labels and "
                    "causal 59-minute proposed features"
                ),
                (
                    "Prior minute changes, peak curvature/height regression and"
                    " last-six-to-all-window change ratio"
                ),
            ),
            (
                7,
                "Gaussian Mixture Model",
                "training",
                "fit_gmm",
                "test_gmm_density_bic_and_training_cluster_predictions_are_real",
                (
                    "Methods: training-only standardization, mixture estimation"
                    " and BIC model selection"
                ),
                (
                    "EM diagonal covariance with declared .001 ridge; select "
                    "K=1..4 by exact diagonal-model BIC"
                ),
            ),
            (
                8,
                "K-nearest neighbours",
                "inference",
                "predict",
                "test_knn_is_not_a_constant_hold_classifier",
                "Methods: classify buy/sell/hold within similarly clustered market trends",
                (
                    "One-nearest-neighbor bank per selected mixture cluster; "
                    "trade predicted minute60 then flatten"
                ),
            ),
        ],
        [
            "Single public Binance BTC market replaces OKX BTC/PEPE and Nasdaq sources.",
            (
                "Jan-Feb2024 training, March diagnostic validation and fixed April-May test "
                "replace original six-month three-market experiments."
            ),
            (
                "Diagonal covariance and K=1..4 grid are declared; paper "
                "does not give all estimator settings."
            ),
            (
                "Feature text says six tail values but indices53..59 "
                "contain seven; choose exactly six, with a dedicated "
                "oracle."
            ),
            (
                "Degenerate/no-peak regression uses an explicit zero slope "
                "and local mean feature convention."
            ),
            (
                "KNN only; RF, DNN, SVM, XGBoost, random benchmark and "
                "original APC profitability reproduction excluded."
            ),
            (
                "Validation reports per-class support, recall and an "
                "always-predict-class-0 baseline; "
                "it cannot alter labels, scaler, GMM or neighbors."
            ),
        ],
    ),
    (
        "sac_passive_quotes",
        "2410.14504v2",
        [
            (
                8,
                "Soft Actor-Critic (SAC) algorithm",
                "training",
                "train",
                "test_sac_soft_bellman_terminal_and_entropy_properties",
                (
                    "Section2.2 and Section3.2: Gaussian SAC actor, "
                    "entropy-regularized policy and twin critics"
                ),
                (
                    "SAC twin-Q soft target, reparameterized tanh actor, target"
                    " EMA and bounded (8,8) hidden layers"
                ),
            ),
            (
                13,
                "non-adverse fill probability",
                "execution",
                "decide",
                (
                    "tests/test_six_audit_refinement.py::"
                    "test_sac_execution_uses_trained_policy_and_inventory_bounded_quote_ledger"
                ),
                (
                    "Eq25-32: adverse quote crosses or nonadverse flow fills; "
                    "cash and inventory ledger"
                ),
                (
                    "Unit passive quotes, p=.2 flow draws, cancellation between"
                    " control intervals and exact marked inventory"
                ),
            ),
            (
                14,
                "maximum position size",
                "reward",
                "train",
                "tests/test_six_audit_refinement.py::test_sac_collected_"
                "rewards_match_separate_marked_wealth_and_penalty",
                (
                    "Eq33-35: price/inventory state, q=5 feasible controls, "
                    "marked wealth minus inventory penalty"
                ),
                (
                    "Replay actual collected quote rewards; penalty "
                    ".001*abs(next inventory)*dt; terminal target has no "
                    "bootstrap"
                ),
            ),
        ],
        [
            "Retain Table1 dt=.001,T=1,spread=.01,q=5,p=.2 and total diffusion variance=.03.",
            (
                "Separate parameter-extraction network omitted; actor/critics "
                "take normalized state directly."
            ),
            (
                "Four random-behavior episodes and 128 updates replace 1000"
                " training episodes and 200 test episodes."
            ),
            "Hidden layers8 replace256; hidden activation is tanh "
            "instead of the source ReLU. The Gaussian action also "
            "has a separate tanh squash. Fixed entropy coefficient "
            ".2 and gamma .99 are disclosed.",
            (
                "Modern twin-Q target follows the implemented "
                "stable-baselines-style SAC rather than the earlier "
                "separate-V equations."
            ),
            "Gaussian latent score is explicitly quantized into sell/hold/buy posting controls.",
            (
                "The critic learns the collected continuous latent, whose quantized posting "
                "control produced the reward; it does not treat an integer control as that latent."
            ),
            (
                "Simulated diffusion approximation and independent "
                "order-flow draws do not reconstruct a calibrated "
                "non-Markov LOB."
            ),
            (
                "Interval probability fill approximation may fill an "
                "off-touch resting quote; it does not enforce exchange "
                "price priority. No queue priority, partial fills, "
                "market impact or "
                "original experimental reward distribution claim."
            ),
        ],
    ),
    (
        "ppo_twap_aware",
        "2510.27334v1",
        [
            (
                4,
                "Proximal Policy Optimisation",
                "training",
                "train",
                "test_ppo_clipping_changes_gradient_by_advantage_sign_and_masks_infeasible_actions",
                "RL MM Agent: decision and action networks trained by PPO",
                (
                    "Two categorical policies, explicit joint behavior "
                    "log-probability binding, clipped objective and value "
                    "regression"
                ),
            ),
            (
                4,
                "self-imitation learning",
                "training",
                "train",
                "test_self_imitation_uses_only_positive_return_advantage",
                (
                    "RL MM Agent: imitate previous trajectories with return "
                    "above current value estimate"
                ),
                (
                    "Positive-advantage actor and value losses; no imitation "
                    "from nonpositive-return advantage"
                ),
            ),
            (
                5,
                "the TW AP agent’s behaviours",
                "inference",
                "decide",
                "test_ppo_clipping_changes_gradient_by_advantage_sign_and_masks_infeasible_actions",
                "Training RL Agent, Table4: rho=0,+1,-1 denotes absent,buy,sell metaorder",
                (
                    "Price/inventory/Hawkes-intensity/rho observations; post or"
                    " cancel top-of-book quotes through contract1.1"
                ),
            ),
        ],
        [
            (
                "Five observations and width8 networks replace the full "
                "source high-dimensional LOB networks."
            ),
            (
                "Training uses a late-start rPOV2-style prefix instead of the source "
                "rPOV1 training metaorder; evaluation observes eight unit children."
            ),
            (
                "Three initial on-policy rollouts and four rounds sampling 32 transitions each, "
                "with one SGD update per transition (128 steps), replace 140+ training episodes."
            ),
            (
                "Interval probability fill approximation may fill an "
                "off-touch resting quote; it does not enforce exchange "
                "price priority. Quote stream uses bounded exponential "
                "Hawkes excitation "
                "with metaorder flow; not the original compound full LOB."
            ),
            (
                "Queue, deep-book, partial fills, endogenous market-maker "
                "impact and original urgency-based TWAP agent are excluded."
            ),
            (
                "100-second warmup, .213-second action step and 40-second "
                "metaorder interval are retained in the bounded setup."
            ),
            (
                "One-basis-point per-fill fee differs from the source "
                "liquidation fee; the exact configured ledger is reported."
            ),
            (
                "No original Sharpe, slippage, square-root impact or "
                "full-system pressure-test reproduction claim."
            ),
        ],
    ),
]


def main():
    lock = json.loads((BASE / "source-lock.json").read_text())
    sources = {s["id"]: s for s in [*lock["sources"], *lock["supporting_sources"]]}
    catalog = []
    for name, identifier, methods, differences in MAPPINGS:
        cls = package(name)
        declaration = cls.declaration
        folder = BASE / "cases" / name
        source = sources[identifier]
        support = {
            "sac_passive_quotes": [
                ("1812.05905v2", 6, "soft Bellman residual", 0),
                ("1812.05905v2", 8, "two soft Q-functions", 0),
            ],
            "ppo_twap_aware": [
                ("1707.06347v2", 3, "probability ratio", 0),
                ("1806.05635v1", 2, "off-policy", 1),
            ],
        }.get(name, [])
        additional_ids = list(dict.fromkeys(item[0] for item in support))
        claim = {
            "strategy_id": declaration.strategy_id,
            "source": {"url": source["url"], "title": source["title"]},
            "reproduction_scope": "bounded local method adaptation; not full paper experiment",
            "differences": differences,
            "claims": [
                {"locator": f"PDF page {page}", "paper_rule": paper, "runtime_rule": runtime}
                for page, anchor, role, implementation, test, paper, runtime in methods
            ],
        }
        spec = {
            "strategy_id": declaration.strategy_id,
            "source_sha256": source["sha256"],
            "fields": sorted(declaration.data.fields),
            "excluded_paper_components": differences,
            "steps": [
                {
                    "role": role,
                    "claim_index": i,
                    "paper_rule": paper,
                    "runtime_rule": runtime,
                    "implementation": CLASSES[name] + "." + implementation,
                    "oracle_node": test if test.startswith("tests/") else TEST + test,
                    "adaptation": "Bounded setting and listed differences apply: " + differences[0],
                }
                for i, (page, anchor, role, implementation, test, paper, runtime) in enumerate(
                    methods
                )
            ],
        }
        extra_oracles = {
            "grouped_cnn_lstm": [
                (
                    0,
                    "fast_prediction",
                    "test_analytic_cnn_all_gradients_match_autodiff_and_sampled_finite_differences",
                ),
                (
                    2,
                    "train",
                    "test_cnn_learns_unseen_controlled_patterns_and_reloads_without_test_branches",
                ),
                (2, "train", "test_cnn_validation_does_not_fit_scaler_or_change_training_updates"),
                (
                    0,
                    "features",
                    "test_cnn_window_normalization_is_price_and_volume_scale_invariant",
                ),
            ],
            "gmm_knn": [
                (
                    1,
                    "train",
                    "test_gmm_validation_cannot_change_quantile_scaler_mixture_or_neighbor_bank",
                )
            ],
        }.get(name, [])
        for claim_index, implementation, oracle in extra_oracles:
            step = dict(spec["steps"][claim_index])
            step.update(
                implementation=CLASSES[name] + "." + implementation,
                oracle_node="tests/test_two_paper_adaptation.py::" + oracle,
                adaptation=(
                    "Additional adaptation check; controlled learnability "
                    "is not paper performance reproduction."
                ),
            )
            spec["steps"].append(step)
        if name == "ppo_twap_aware":
            for node in (
                "test_ppo_multistep_gradient_reset_matches_independent_finite_difference",
                "test_ppo_real_training_never_updates_parameters_outside_current_loss",
            ):
                spec["steps"].append(
                    {
                        **spec["steps"][0],
                        "implementation": "PPOTWAPAware.update_parameters",
                        "oracle_node": "tests/test_six_audit_refinement.py::" + node,
                        "runtime_rule": "Clear all optimizer gradients before every "
                        "skip/mask/clip-dependent backward pass; no stale "
                        "updates",
                    }
                )
        if name == "grouped_cnn_lstm":
            for claim_index, implementation, oracle in (
                (
                    2,
                    "train",
                    "test_author_minmax_fit_uses_only_training_rows_without_validation_clipping",
                ),
                (
                    0,
                    "causal_features",
                    "test_causal_indicator_features_match_independent_prefix_equations",
                ),
                (
                    0,
                    "causal_features",
                    "test_causal_features_use_completed_prefix_and_reject_insufficient_history",
                ),
                (
                    2,
                    "train",
                    "test_actual_hidden4_training_freezes_encoder_and_exclud"
                    "es_validation_from_fitting",
                ),
                (
                    0,
                    "fast_prediction",
                    "test_frozen_encoder_cache_matches_uncached_prediction_and_training_gradients",
                ),
                (
                    2,
                    "head_prediction",
                    "test_head_refinement_gradients_match_independent_scalar_autodiff",
                ),
                (
                    0,
                    "convolution_sequence",
                    "test_cached_convolution_recurrent_stage_preserves_forwa"
                    "rd_and_recurrent_gradients",
                ),
                (
                    2,
                    "train",
                    "test_deployed_hidden4_relu_stages_learn_unseen_patterns"
                    "_without_threshold_changes",
                ),
            ):
                step = dict(spec["steps"][claim_index])
                step.update(
                    implementation=CLASSES[name] + "." + implementation,
                    oracle_node="tests/test_cnn_refinement.py::" + oracle,
                    adaptation="Actual hidden4 staged adaptation or unselected causal alternative; "
                    "controlled checks do not establish market prediction quality.",
                )
                spec["steps"].append(step)
        if additional_ids:
            claim["algorithm_sources"] = [
                {"url": sources[item]["url"], "title": sources[item]["title"]}
                for item in additional_ids
            ]
            spec["additional_source_sha256"] = [sources[item]["sha256"] for item in additional_ids]
        MethodSpec.model_validate(spec)
        save(folder / "claims.json", claim)
        save(folder / "method.json", spec)
        code = ROOT / "strategies" / declaration.strategy_id / "strategy.py"
        expected = {
            "contract_version": declaration.contract_version,
            "strategy_id": declaration.strategy_id,
            "kind": declaration.kind,
            "granularity": declaration.data.granularity,
            "fields": sorted(declaration.data.fields),
            "training_required": declaration.training_required,
            "symbols": sorted(declaration.data.symbols),
            "actions": sorted(declaration.actions),
            "market_kind": declaration.data.market_kind,
            "bar_seconds": declaration.data.bar_seconds,
        }
        recipe = {
            "source": {
                "title": source["title"],
                "url": source["url"],
                "format": "pdf",
                "file": "../../../source-cache/six/" + identifier + ".pdf",
                "sha256": source["sha256"],
                "license": "Redistribution not asserted; exact cached public source",
            },
            "package": "../../../../strategies/" + declaration.strategy_id,
            "package_source_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
            "claim_file": "claims.json",
            "spec_file": "method.json",
            "spec_sha256": hashlib.sha256((folder / "method.json").read_bytes()).hexdigest(),
            "expected": expected,
            "anchors": [
                {"page": page, "text": anchor, "claim_index": i}
                for i, (page, anchor, *_) in enumerate(methods)
            ],
            "review_note": (
                "Upstream paper interpretation and code were authored after"
                " the original sealed submission. This is reviewed "
                "directional intake, not autonomous generation from "
                "arbitrary prose. "
            )
            + differences[0],
        }
        if additional_ids:
            recipe["additional_sources"] = [
                {
                    "title": sources[item]["title"],
                    "url": sources[item]["url"],
                    "format": "pdf",
                    "file": "../../../source-cache/six/" + item + ".pdf",
                    "sha256": sources[item]["sha256"],
                    "license": "Redistribution not asserted; exact cached public source",
                }
                for item in additional_ids
            ]
            recipe["anchors"].extend(
                {
                    "source_index": additional_ids.index(item) + 1,
                    "page": page,
                    "text": text,
                    "claim_index": claim_index,
                }
                for item, page, text, claim_index in support
            )
        Recipe.model_validate(recipe)
        save(folder / "recipe.json", recipe)
        checked = verify_recipe(folder / "recipe.json")
        catalog.append(
            {
                "strategy_id": declaration.strategy_id,
                "paper_id": identifier,
                "recipe": (folder / "recipe.json").relative_to(ROOT).as_posix(),
                "fixture": "examples/six-paper/" + name,
                "training_required": declaration.training_required,
                "engine": "passive"
                if name in {"sac_passive_quotes", "ppo_twap_aware"}
                else "reference",
                "fee_rate": "0.0001"
                if name in {"grouped_cnn_lstm", "gmm_knn", "ppo_twap_aware"}
                else "0",
                "oracle_nodes": sorted(set(step["oracle_node"] for step in spec["steps"])),
                "source_mapping": checked,
            }
        )
    save(BASE / "catalog.json", {"cases": catalog})
    print(json.dumps({"status": "mapped", "cases": len(catalog)}))


if __name__ == "__main__":
    main()
