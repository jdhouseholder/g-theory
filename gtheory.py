from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.distributions.transforms import OrderedTransform
from numpyro.infer import MCMC, NUTS
from numpyro.infer.reparam import LocScaleReparam


SEED = 1

NUTS_WARMUP = 1000
NUTS_SAMPLES = 1000
NUTS_CHAINS = 4
TARGET_ACCEPT = 0.95

# Vectorized chains required for GPU parallelism in NumPyro NUTS.
_JAX_BACKEND = jax.default_backend()
NUTS_CHAIN_METHOD = "vectorized" if _JAX_BACKEND == "gpu" else "sequential"
print(f"[gtheory] JAX backend = {_JAX_BACKEND}, chain_method = {NUTS_CHAIN_METHOD}")

DSTUDY_NI = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
DSTUDY_NJ = [1, 2, 3, 5, 8]
TARGET_PHI = 0.9

LOGIT_RESIDUAL_VAR = math.pi**2 / 3.0


@dataclass
class Encoded:
    model_idx: jnp.ndarray
    item_idx: jnp.ndarray
    judge_idx: jnp.ndarray
    score_ord: jnp.ndarray
    score_raw: jnp.ndarray
    n_models: int
    n_items: int
    n_judges: int
    models: list[str]
    items: list[str]
    judges: list[str]


def encode(df: pd.DataFrame) -> Encoded:
    models = sorted(df["model"].unique())
    items = sorted(df["item"].unique())
    judges = sorted(df["judge"].unique())
    m_to_i = {x: i for i, x in enumerate(models)}
    i_to_i = {x: i for i, x in enumerate(items)}
    j_to_i = {x: i for i, x in enumerate(judges)}
    return Encoded(
        model_idx=jnp.array(df["model"].map(m_to_i).to_numpy(), dtype=jnp.int32),
        item_idx=jnp.array(df["item"].map(i_to_i).to_numpy(), dtype=jnp.int32),
        judge_idx=jnp.array(df["judge"].map(j_to_i).to_numpy(), dtype=jnp.int32),
        # OrderedLogistic expects 0..K-1
        score_ord=jnp.array(df["score_ord"].to_numpy() - 1, dtype=jnp.int32),
        score_raw=jnp.array(df["score_raw"].to_numpy(), dtype=jnp.float32),
        n_models=len(models),
        n_items=len(items),
        n_judges=len(judges),
        models=models,
        items=items,
        judges=judges,
    )


VAR_NAMES = ("M", "I", "J", "MI", "MJ", "IJ")


# Per-effect partial-centering: 0.0 = non-centered (good for small sigma),
# 1.0 = centered (good for large sigma), in-between = partial.
_DEFAULT_REPARAM = {
    "alpha": 1.0,  # centered: sigma_M is moderate; NC induces inverted funnel
    "beta": 0.0,
    "gamma": 0.0,
    "ab": 1.0,  # centered: sigma_MI runs large (4-7 on logit scale)
    "ag": 0.0,
    "bg": 0.0,
}

# sigma_MI is the only component that needs a wider prior than HalfNormal(1).
_DEFAULT_PRIOR_SIGMA = {
    "M": 1.0,
    "I": 1.0,
    "J": 1.0,
    "MI": 2.0,
    "MJ": 1.0,
    "IJ": 1.0,
}


@dataclass(frozen=True)
class ModelConfig:
    reparam: dict[str, float]
    prior_sigma: dict[str, float]
    n_categories: int = 5

    @property
    def n_cutpoints(self) -> int:
        return self.n_categories - 1

    def with_overrides(
        self,
        *,
        reparam: dict[str, float] | None = None,
        prior_sigma: dict[str, float] | None = None,
        n_categories: int | None = None,
    ) -> "ModelConfig":
        return ModelConfig(
            reparam={**self.reparam, **(reparam or {})},
            prior_sigma={**self.prior_sigma, **(prior_sigma or {})},
            n_categories=self.n_categories if n_categories is None else n_categories,
        )


def default_config() -> ModelConfig:
    return ModelConfig(
        reparam=dict(_DEFAULT_REPARAM),
        prior_sigma=dict(_DEFAULT_PRIOR_SIGMA),
        n_categories=5,
    )


def _sample_random_effects(n_models, n_items, n_judges, *, config: ModelConfig):
    sigmas = {}
    for name in VAR_NAMES:
        sigmas[name] = numpyro.sample(
            f"sigma_{name}", dist.HalfNormal(config.prior_sigma.get(name, 1.0))
        )

    reparam_dict = {}
    for site, c in config.reparam.items():
        if c < 1.0 - 1e-9:
            reparam_dict[site] = LocScaleReparam(centered=float(c))

    with numpyro.handlers.reparam(config=reparam_dict):
        with numpyro.plate("models", n_models):
            alpha = numpyro.sample("alpha", dist.Normal(0.0, sigmas["M"]))
        with numpyro.plate("items", n_items):
            beta = numpyro.sample("beta", dist.Normal(0.0, sigmas["I"]))
        with numpyro.plate("judges", n_judges):
            gamma = numpyro.sample("gamma", dist.Normal(0.0, sigmas["J"]))
        with numpyro.plate("MI_plate", n_models * n_items):
            ab_flat = numpyro.sample("ab", dist.Normal(0.0, sigmas["MI"]))
        with numpyro.plate("MJ_plate", n_models * n_judges):
            ag_flat = numpyro.sample("ag", dist.Normal(0.0, sigmas["MJ"]))
        with numpyro.plate("IJ_plate", n_items * n_judges):
            bg_flat = numpyro.sample("bg", dist.Normal(0.0, sigmas["IJ"]))
    ab = ab_flat.reshape(n_models, n_items)
    ag = ag_flat.reshape(n_models, n_judges)
    bg = bg_flat.reshape(n_items, n_judges)
    return sigmas, alpha, beta, gamma, ab, ag, bg


def lmm_model(
    model_idx,
    item_idx,
    judge_idx,
    score,
    n_models,
    n_items,
    n_judges,
    config: ModelConfig,
):
    sigmas, alpha, beta, gamma, ab, ag, bg = _sample_random_effects(
        n_models, n_items, n_judges, config=config
    )
    mu = numpyro.sample("mu", dist.Normal(3.0, 2.0))  # data lives on 1..5
    sigma_E = numpyro.sample("sigma_E", dist.HalfNormal(1.0))

    eta = (
        mu
        + alpha[model_idx]
        + beta[item_idx]
        + gamma[judge_idx]
        + ab[model_idx, item_idx]
        + ag[model_idx, judge_idx]
        + bg[item_idx, judge_idx]
    )
    numpyro.sample("obs", dist.Normal(eta, sigma_E), obs=score)


def glmm_ordinal(
    model_idx,
    item_idx,
    judge_idx,
    score,
    n_models,
    n_items,
    n_judges,
    config: ModelConfig,
):
    """mu fixed to 0; cutpoints absorb the location (else jointly unidentified)."""
    sigmas, alpha, beta, gamma, ab, ag, bg = _sample_random_effects(
        n_models, n_items, n_judges, config=config
    )
    cut_raw = numpyro.sample(
        "cutpoints_raw",
        dist.Normal(jnp.zeros(config.n_cutpoints), 2.0).to_event(1),
    )
    cutpoints = numpyro.deterministic("cutpoints", OrderedTransform()(cut_raw))

    eta = (
        alpha[model_idx]
        + beta[item_idx]
        + gamma[judge_idx]
        + ab[model_idx, item_idx]
        + ag[model_idx, judge_idx]
        + bg[item_idx, judge_idx]
    )
    numpyro.sample("obs", dist.OrderedLogistic(eta, cutpoints), obs=score)


def glmm_ordinal_het(
    model_idx,
    item_idx,
    judge_idx,
    score,
    n_models,
    n_items,
    n_judges,
    config: ModelConfig,
):
    """Per-judge logistic scale s_j; vanilla GLMM is the s_j=1 special case."""
    sigmas, alpha, beta, gamma, ab, ag, bg = _sample_random_effects(
        n_models, n_items, n_judges, config=config
    )
    cut_raw = numpyro.sample(
        "cutpoints_raw",
        dist.Normal(jnp.zeros(config.n_cutpoints), 2.0).to_event(1),
    )
    cutpoints = numpyro.deterministic("cutpoints", OrderedTransform()(cut_raw))
    log_s_j = numpyro.sample(
        "log_s_j", dist.Normal(jnp.zeros(n_judges), 0.5).to_event(1)
    )
    s_j = numpyro.deterministic("s_j", jnp.exp(log_s_j))

    eta = (
        alpha[model_idx]
        + beta[item_idx]
        + gamma[judge_idx]
        + ab[model_idx, item_idx]
        + ag[model_idx, judge_idx]
        + bg[item_idx, judge_idx]
    )
    # Compute log-probs from the cumulative sigmoid directly because NumPyro's
    # OrderedLogistic doesn't accept per-observation cutpoints.
    s_per_obs = s_j[judge_idx]
    eta_scaled = eta / s_per_obs
    cuts_per_obs = cutpoints[None, :] / s_per_obs[:, None]
    cum_logits = cuts_per_obs - eta_scaled[:, None]
    cum_probs = jax.nn.sigmoid(cum_logits)
    p_low = cum_probs[..., :1]
    p_mid = cum_probs[..., 1:] - cum_probs[..., :-1]
    p_high = 1.0 - cum_probs[..., -1:]
    probs = jnp.concatenate([p_low, p_mid, p_high], axis=-1)
    probs = jnp.clip(probs, 1e-12, 1.0)  # keep log_prob finite under extreme cutpoints
    numpyro.sample("obs", dist.Categorical(probs=probs), obs=score)


def run_nuts(
    model_fn,
    key,
    *,
    model_kwargs,
    label: str,
    warmup: int = NUTS_WARMUP,
    samples: int = NUTS_SAMPLES,
    chains: int = NUTS_CHAINS,
    target_accept: float = TARGET_ACCEPT,
    dense_mass: bool = False,
    init_strategy=None,
    max_tree_depth: int = 10,
):
    kernel_kwargs = dict(
        target_accept_prob=target_accept,
        dense_mass=dense_mass,
        max_tree_depth=max_tree_depth,
    )
    if init_strategy is not None:
        kernel_kwargs["init_strategy"] = init_strategy
    kernel = NUTS(model_fn, **kernel_kwargs)
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress_bar=sys.stdout.isatty(),
        chain_method=NUTS_CHAIN_METHOD,
    )
    t0 = time.time()
    mcmc.run(key, **model_kwargs)
    elapsed = time.time() - t0
    print(f"  [{label}] NUTS finished in {elapsed:.1f}s")

    sigma_sites = [f"sigma_{k}" for k in VAR_NAMES] + (
        ["sigma_E"] if "sigma_E" in mcmc.get_samples() else []
    )
    summary = numpyro.diagnostics.summary(
        {
            k: v
            for k, v in mcmc.get_samples(group_by_chain=True).items()
            if k in sigma_sites
        },
        prob=0.95,
    )
    divergences = int(mcmc.get_extra_fields().get("diverging", np.array([0])).sum())
    return mcmc.get_samples(), summary, divergences, elapsed


def variance_table(
    samples: dict, label: str, residual_var: float | None
) -> pd.DataFrame:
    rows = []
    for k in VAR_NAMES:
        s = np.asarray(samples[f"sigma_{k}"]).reshape(-1)
        v = s**2
        rows.append(
            dict(
                model=label,
                component=f"sigma2_{k}",
                median=float(np.median(v)),
                mean=float(v.mean()),
                ci_low=float(np.quantile(v, 0.025)),
                ci_high=float(np.quantile(v, 0.975)),
            )
        )
    if residual_var is None:
        s = np.asarray(samples["sigma_E"]).reshape(-1)
        v = s**2
        rows.append(
            dict(
                model=label,
                component="sigma2_E",
                median=float(np.median(v)),
                mean=float(v.mean()),
                ci_low=float(np.quantile(v, 0.025)),
                ci_high=float(np.quantile(v, 0.975)),
            )
        )
    else:
        rows.append(
            dict(
                model=label,
                component="sigma2_E",
                median=residual_var,
                mean=residual_var,
                ci_low=residual_var,
                ci_high=residual_var,
            )
        )
    return pd.DataFrame(rows)


def save_dstudy_samples(
    samples: dict, out_path: Path | str, *, extras: dict | None = None
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in samples.items()}
    if extras:
        for k, v in extras.items():
            arrays[k] = np.asarray(v)
    np.savez_compressed(out_path, **arrays)
    return out_path


def load_dstudy_samples(in_path: Path | str) -> dict:
    z = np.load(in_path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def latent_dstudy(
    samples: dict,
    residual_var: float | np.ndarray | None,
    ni_grid=DSTUDY_NI,
    nj_grid=DSTUDY_NJ,
) -> pd.DataFrame:
    """residual_var: None reads samples['sigma_E'] (LMM); scalar = homoscedastic
    GLMM; ndarray = per-draw effective residual (het GLMM)."""
    s2 = {k: (np.asarray(samples[f"sigma_{k}"]).reshape(-1)) ** 2 for k in VAR_NAMES}
    if residual_var is None:
        s2_E = (np.asarray(samples["sigma_E"]).reshape(-1)) ** 2
    elif np.ndim(residual_var) == 0:
        s2_E = np.full_like(s2["M"], float(residual_var))
    else:
        s2_E = np.asarray(residual_var).reshape(-1)

    rows = []
    for n_I in ni_grid:
        for n_J in nj_grid:
            sigma_delta = s2["MI"] / n_I + s2["MJ"] / n_J + s2_E / (n_I * n_J)
            sigma_Delta = (
                sigma_delta + s2["I"] / n_I + s2["J"] / n_J + s2["IJ"] / (n_I * n_J)
            )
            e_rho2 = s2["M"] / (s2["M"] + sigma_delta)
            phi = s2["M"] / (s2["M"] + sigma_Delta)
            rows.append(
                dict(
                    n_I=n_I,
                    n_J=n_J,
                    scale="latent",
                    Phi_median=float(np.median(phi)),
                    Phi_low=float(np.quantile(phi, 0.025)),
                    Phi_high=float(np.quantile(phi, 0.975)),
                    Erho2_median=float(np.median(e_rho2)),
                    Erho2_low=float(np.quantile(e_rho2, 0.025)),
                    Erho2_high=float(np.quantile(e_rho2, 0.975)),
                )
            )
    return pd.DataFrame(rows)


def _draw_eta_from_re(eta_keys, s2, n_models, n_I, n_J):
    alpha = jax.random.normal(eta_keys[0], (n_models,)) * jnp.sqrt(s2["M"])
    beta = jax.random.normal(eta_keys[1], (n_I,)) * jnp.sqrt(s2["I"])
    gamma = jax.random.normal(eta_keys[2], (n_J,)) * jnp.sqrt(s2["J"])
    ab = jax.random.normal(eta_keys[3], (n_models, n_I)) * jnp.sqrt(s2["MI"])
    ag = jax.random.normal(eta_keys[4], (n_models, n_J)) * jnp.sqrt(s2["MJ"])
    bg = jax.random.normal(eta_keys[5], (n_I, n_J)) * jnp.sqrt(s2["IJ"])
    return (
        alpha[:, None, None]
        + beta[None, :, None]
        + gamma[None, None, :]
        + ab[:, :, None]
        + ag[:, None, :]
        + bg[None, :, :]
    )


def _simulate_observed_phi_ordinal(
    rng_key, s2, cutpoints, n_models, n_I, n_J, n_reps=20
):
    """Reps are scanned (not vmapped) to cap peak memory at one rep."""

    def one_rep(key):
        keys = jax.random.split(key, 7)
        eta = _draw_eta_from_re(keys[:6], s2, n_models, n_I, n_J)
        latent = eta + jax.random.logistic(keys[6], eta.shape)
        y = jnp.sum(latent[..., None] > cutpoints, axis=-1) + 1
        return _anova_phi(y, n_models, n_I, n_J)

    rep_keys = jax.random.split(rng_key, n_reps)

    def step(acc, k):
        phi, e = one_rep(k)
        return (acc[0] + phi / n_reps, acc[1] + e / n_reps), None

    (phi_mean, e_mean), _ = jax.lax.scan(
        step, (jnp.float32(0.0), jnp.float32(0.0)), rep_keys
    )
    return phi_mean, e_mean


def _anova_phi(y, n_models, n_I, n_J):
    """Three-way ANOVA MoM -> Phi on the observed scale.

    Standard balanced-design formulas (Brennan ch. 4 / AIMS Reliability sec. 7.5).
    """
    y = y.astype(jnp.float32)
    grand = y.mean()
    Mbar = y.mean(axis=(1, 2))
    Ibar = y.mean(axis=(0, 2))
    Jbar = y.mean(axis=(0, 1))
    MIbar = y.mean(axis=2)
    MJbar = y.mean(axis=1)
    IJbar = y.mean(axis=0)

    ss_M = n_I * n_J * jnp.sum((Mbar - grand) ** 2)
    ss_I = n_models * n_J * jnp.sum((Ibar - grand) ** 2)
    ss_J = n_models * n_I * jnp.sum((Jbar - grand) ** 2)
    ss_MI = n_J * jnp.sum((MIbar - Mbar[:, None] - Ibar[None, :] + grand) ** 2)
    ss_MJ = n_I * jnp.sum((MJbar - Mbar[:, None] - Jbar[None, :] + grand) ** 2)
    ss_IJ = n_models * jnp.sum((IJbar - Ibar[:, None] - Jbar[None, :] + grand) ** 2)
    ss_total = jnp.sum((y - grand) ** 2)
    ss_res = ss_total - (ss_M + ss_I + ss_J + ss_MI + ss_MJ + ss_IJ)

    df_M = n_models - 1
    df_I = n_I - 1
    df_J = n_J - 1
    df_MI = df_M * df_I
    df_MJ = df_M * df_J
    df_IJ = df_I * df_J
    df_res = df_M * df_I * df_J

    ms_M = ss_M / df_M
    ms_I = ss_I / df_I
    ms_J = ss_J / df_J
    ms_MI = ss_MI / df_MI
    ms_MJ = ss_MJ / df_MJ
    ms_IJ = ss_IJ / df_IJ
    ms_res = ss_res / jnp.maximum(df_res, 1)

    # No replication, so residual collapses three-way + error (standard).
    s2_E = ms_res
    s2_MI = jnp.maximum((ms_MI - ms_res) / n_J, 0.0)
    s2_MJ = jnp.maximum((ms_MJ - ms_res) / n_I, 0.0)
    s2_IJ = jnp.maximum((ms_IJ - ms_res) / n_models, 0.0)
    s2_M = jnp.maximum((ms_M - ms_MI - ms_MJ + ms_res) / (n_I * n_J), 0.0)
    s2_I = jnp.maximum((ms_I - ms_MI - ms_IJ + ms_res) / (n_models * n_J), 0.0)
    s2_J = jnp.maximum((ms_J - ms_MJ - ms_IJ + ms_res) / (n_models * n_I), 0.0)

    sigma_delta = s2_MI / n_I + s2_MJ / n_J + s2_E / (n_I * n_J)
    sigma_Delta = sigma_delta + s2_I / n_I + s2_J / n_J + s2_IJ / (n_I * n_J)
    phi = s2_M / (s2_M + sigma_Delta + 1e-12)
    e_rho2 = s2_M / (s2_M + sigma_delta + 1e-12)
    return phi, e_rho2


# Batched + JIT'd versions of the per-draw simulators (one JIT per unique
# (n_models, n_I, n_J) shape). Single-draw versions above kept for clarity.


@partial(jax.jit, static_argnames=("n_models", "n_I", "n_J"))
def _phi_ordinal_batch(rng_keys, s2_stack, cuts_stack, n_models, n_I, n_J):
    def one(key, s2_row, cuts_row):
        s2 = {
            "M": s2_row[0],
            "I": s2_row[1],
            "J": s2_row[2],
            "MI": s2_row[3],
            "MJ": s2_row[4],
            "IJ": s2_row[5],
        }
        phi, _ = _simulate_observed_phi_ordinal(key, s2, cuts_row, n_models, n_I, n_J)
        return phi

    return jax.vmap(one)(rng_keys, s2_stack, cuts_stack)


@partial(jax.jit, static_argnames=("n_models", "n_I", "n_J"))
def _phi_gaussian_batch(rng_keys, s2_stack, s2_E_stack, n_models, n_I, n_J):
    def one(key, s2_row, s2_E):
        s2 = {
            "M": s2_row[0],
            "I": s2_row[1],
            "J": s2_row[2],
            "MI": s2_row[3],
            "MJ": s2_row[4],
            "IJ": s2_row[5],
        }
        return _simulate_observed_phi_gaussian(key, s2, s2_E, n_models, n_I, n_J)

    return jax.vmap(one)(rng_keys, s2_stack, s2_E_stack)


def probability_dstudy(
    samples: dict,
    rng_key,
    *,
    label: str,
    cutpoints_samples: np.ndarray | None,
    ni_grid=DSTUDY_NI,
    nj_grid=DSTUDY_NJ,
    n_draws: int = 200,
    n_models_sim: int = 25,
) -> pd.DataFrame:
    """LMM path (cutpoints_samples is None) uses Gaussian noise; ANOVA is still
    used so finite-sample bias matches the GLMM path. Refuses het-GLMM samples
    (use latent_dstudy with a per-draw effective residual instead).
    """
    if cutpoints_samples is not None and "s_j" in samples:
        raise ValueError(
            f"[probability_dstudy] {label}: 's_j' present in samples (het GLMM) "
            "but the simulator is homoscedastic. Use latent_dstudy with a "
            "per-draw effective residual variance instead."
        )
    s2_all = {
        k: (np.asarray(samples[f"sigma_{k}"]).reshape(-1)) ** 2 for k in VAR_NAMES
    }
    is_gaussian = cutpoints_samples is None
    if is_gaussian:
        s2_E_all = (np.asarray(samples["sigma_E"]).reshape(-1)) ** 2
    n_total = len(s2_all["M"])
    idx = np.random.default_rng(SEED).choice(
        n_total, size=min(n_draws, n_total), replace=False
    )
    s2_stack = jnp.stack(
        [jnp.asarray(s2_all[k][idx], dtype=jnp.float32) for k in VAR_NAMES],
        axis=1,
    )
    if is_gaussian:
        s2_E_stack = jnp.asarray(s2_E_all[idx], dtype=jnp.float32)
    else:
        cuts_stack = jnp.asarray(cutpoints_samples[idx], dtype=jnp.float32)

    rows = []
    for n_I in ni_grid:
        for n_J in nj_grid:
            # ANOVA needs >= 2 levels per facet; latent_dstudy handles singletons.
            if n_I < 2 or n_J < 2:
                rows.append(
                    dict(
                        n_I=n_I,
                        n_J=n_J,
                        scale="probability",
                        Phi_median=np.nan,
                        Phi_low=np.nan,
                        Phi_high=np.nan,
                        Erho2_median=np.nan,
                        Erho2_low=np.nan,
                        Erho2_high=np.nan,
                    )
                )
                continue
            cell_key = jax.random.fold_in(rng_key, n_I * 100000 + n_J)
            keys_cell = jax.random.split(cell_key, len(idx))
            if is_gaussian:
                phis = _phi_gaussian_batch(
                    keys_cell,
                    s2_stack,
                    s2_E_stack,
                    n_models_sim,
                    n_I,
                    n_J,
                )
            else:
                phis = _phi_ordinal_batch(
                    keys_cell,
                    s2_stack,
                    cuts_stack,
                    n_models_sim,
                    n_I,
                    n_J,
                )
            phis = np.asarray(phis)
            rows.append(
                dict(
                    n_I=n_I,
                    n_J=n_J,
                    scale="probability",
                    Phi_median=float(np.median(phis)),
                    Phi_low=float(np.quantile(phis, 0.025)),
                    Phi_high=float(np.quantile(phis, 0.975)),
                    Erho2_median=np.nan,  # MC noise dominates Erho2/Phi gap; skip
                    Erho2_low=np.nan,
                    Erho2_high=np.nan,
                )
            )
    return pd.DataFrame(rows)


def _simulate_observed_phi_gaussian(rng_key, s2, s2_E, n_models, n_I, n_J, n_reps=20):
    def one_rep(key):
        keys = jax.random.split(key, 7)
        eta = _draw_eta_from_re(keys[:6], s2, n_models, n_I, n_J)
        y = eta + jax.random.normal(keys[6], eta.shape) * jnp.sqrt(s2_E)
        phi, _ = _anova_phi(y, n_models, n_I, n_J)
        return phi

    rep_keys = jax.random.split(rng_key, n_reps)

    def step(acc, k):
        return acc + one_rep(k) / n_reps, None

    phi_mean, _ = jax.lax.scan(step, jnp.float32(0.0), rep_keys)
    return phi_mean


def cheapest_design(
    dstudy: pd.DataFrame, target: float = TARGET_PHI, use_lower_ci: bool = False
) -> dict:
    """use_lower_ci=True is conservative: even the 2.5% posterior tail must clear target."""
    col = "Phi_low" if use_lower_ci else "Phi_median"
    ok = dstudy[dstudy[col] >= target].copy()
    if ok.empty:
        return dict(
            found=False,
            n_I=None,
            n_J=None,
            cost=None,
            Phi_median=None,
            Phi_low=None,
            Phi_high=None,
            col=col,
            target=target,
        )
    ok["cost"] = ok["n_I"] * ok["n_J"]
    best = ok.sort_values(["cost", "n_J", "n_I"]).iloc[0]
    return dict(
        found=True,
        n_I=int(best.n_I),
        n_J=int(best.n_J),
        cost=int(best.cost),
        Phi_median=float(best.Phi_median),
        Phi_low=float(best.Phi_low),
        Phi_high=float(best.Phi_high),
        col=col,
        target=target,
    )
