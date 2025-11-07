# Toy Mirror Descent Tilt in 1D

This is a minimal, reproducible proof-of-concept demonstrating inference-time mirror descent on a 1D action and 1D next-state toy model.
We use a known linear-Gaussian joint model and a quadratic reward/value surrogate so that the mirror descent (MD) target can be computed in closed form.

- **Base policy:** q(a|s) = N(μ_a, σ_a^2)
- **Dynamics:** q(s'|a,s) = N(α a + d s + μ_ε, σ_{s'}^2)
- **Value surrogate:** Q(a) = -0.5 · qA · (a - μ_Q)^2
- **MD target:** π_β(a|s) ∝ q(a|s) · exp{β Q(a)} is Gaussian with
  - precision τ_tilt = 1/σ_a^2 + β·qA
  - mean μ_tilt = (μ_a/σ_a^2 + β·qA·μ_Q) / τ_tilt

Because the tilt depends only on a, the conditional over next-state is invariant:
- **Conditional invariance:** π_β(s'|a,s) = q(s'|a,s)

## What this script does

- Samples from the base joint q(a,s') and from the exact tilted joint using the closed-form π_β(a|s).
- Constructs an importance-weighted approximation to π_β(a|s) from base samples to verify MD numerically.
  - Verifies conditional invariance both analytically and empirically via:
  - KS tests of q(s'|a≈a₀) vs π_β(s'|a≈a₀) against the analytic N(α a₀ + d s + μ_ε, σ_{s'}^2).
- Produces clear plots and a metrics.json summary.

## Requirements

- Python ≥ 3.9 (tested on Linux)
- Install with:
  
  ```bash
  pip install -r requirements.txt
  ```

The script writes all artifacts to `toy-md/outputs/`.

## Environment setup

- venv (recommended):
  
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt
  ```
- conda (optional):
  
  ```bash
  conda create -y -n toy-md python=3.10
  conda activate toy-md
  pip install -U pip
  pip install -r requirements.txt
  ```

## Quickstart (smoke test)

Runs fast and produces a subset of outputs:
  
```bash
python run_toy_md.py
```

Check `outputs/` for `metrics.json`, `action_match_pc_ula.png`, `state_match_pc_ula.png`, and `action_kl_vs_refresh.png` / `state_kl_vs_refresh.png`.

## Full run

Default settings generate animations and diagnostics for selected samplers:
  
```bash
# Run both ULA and MALA panels and diagnostics
python run_toy_md.py --samplers pc_ula,pc_mala
  
# Include joint MALA as well
python run_toy_md.py --samplers pc_ula,pc_mala,pc_joint_mala
```

Key flags:
- `--beta-schedule {linear,late_constant,late_ramp,tfg_plus_final_mala,sigma_power}`
- `--lambda-scale FLOAT` (overall MD strength at clean)
- `--K INT` (levels), `--refresh-L INT` (state refreshes per level)
- `--action-steps-per-level INT` (per-level action steps)
- `--kl-runs INT`, `--kl-bins INT`, `--sweep-max-refresh INT`, `--sweep-runs INT`
- See `run_toy_md.py --help` for the full list.

## Outputs

- `metrics.json` — summary of KLs, evals per action, mixing metrics, diagnostics.
- `action_match_{pc_ula|pc_mala|pc_joint_mala}.png`
- `state_match_{pc_ula|pc_mala|pc_joint_mala}.png` and corresponding `_gt` variants
- `action_kl_vs_refresh.png`, `state_kl_vs_refresh.png`
- `anim_pc_ula.gif`, `anim_pc_mala.gif`, `anim_pc_joint_mala.gif` (depending on `--samplers`)

## Notes

- GIF creation requires `pillow` (included in requirements).
- Results are stochastic; use `--seed` to control the RNG.

## Tuning parameters
  
Use CLI flags in `run_toy_md.py` to explore regimes (see `--help`):
- `--sigma-a`, `--mu-a` — base action prior.
- `--qA`, `--mu-Q` — curvature and optimum of Q(a).
- `--alpha`, `--sigma-sprime`, `--d`, `--mu-eps` — linear-Gaussian dynamics.
- `--beta-schedule`, `--lambda-scale` — MD strength schedule.
- `--K`, `--refresh-L`, `--action-steps-per-level` — PC sampler controls.

## Additional samplers (per proposal)

- **Importance reweighting (discrete MD proxy):** w ∝ exp{β Q(a)} applied to base samples.
- **Independence-MH on actions:** proposal q(a|s), target π_β(a|s); acceptance α = min{1, exp[β(Q(a')−Q(a))]}.
- **ULA on actions (PoE action score):** Langevin on a with ∇ log q(a) + β ∇Q(a); s' is refreshed exactly from q(s'|a,s).
- **MALA on actions:** preconditioned MH correction of ULA (exact for fixed step in the limit); s' refresh remains exact.
- **Gibbs for tilted joint:** alternate s'~q(s'|a,s) and a~N(μ_post(s'),σ_post) where a|s' ∝ π_β(a|s)·q(s'|a,s) is Gaussian.

All methods empirically confirm MD action marginal and conditional invariance in this toy model.
