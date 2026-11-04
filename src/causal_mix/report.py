from __future__ import annotations

import html
import json

from .common import DOMAINS, POLICIES, progress, read_json


def plot_results(store, fitted, comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    x = np.arange(len(POLICIES))
    labels = ["Uniform", "Best proxy", "RegMix-style", "CausalMix-A", "CausalMix-S"]
    bottom = np.zeros(len(POLICIES))
    for i, domain in enumerate(DOMAINS):
        values = [fitted["policies"][p][i] for p in POLICIES]
        axes[0].bar(x, values, bottom=bottom, label=domain)
        bottom += values
    axes[0].set(title="Selected mixtures", ylabel="Example sampling proportion", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    for axis, role in zip(axes[1:], ("proxy", "transfer")):
        scores = {r["policy"]: r["test_improvement"] for r in comparisons if r["role"] == role}
        for i, policy in enumerate(POLICIES):
            if policy in scores:
                axis.bar(i, scores[policy], color="#247663")
        axis.axhline(0, color="#333", linewidth=.7)
        axis.set(title=f"{role.title()}: held-out improvement", ylabel="Response NLL reduction vs base model")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Toy demonstration · one seed · compare policies within each model", fontsize=12)
    fig.savefig(store.path("results.png"), dpi=160)
    plt.close(fig)


def render_table(headers, rows):
    escape = lambda v: html.escape(str(v))
    return "<table><thead><tr>" + "".join(f"<th>{escape(v)}</th>" for v in headers) + \
           "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in row)
           + "</tr>" for row in rows) + "</tbody></table>"


def report(store):
    status = store.get("status.json") if store.exists("status.json") else {"state": "incomplete"}
    parts = ["<p class='eyebrow'>LOCAL RESEARCH DEMONSTRATION</p><h1>CausalMix, at toy scale</h1>",
             f"<p>Status: <strong>{html.escape(status['state'])}</strong>. "
             "SmolLM2-135M proxy experiments → Qwen2.5-0.5B mixture transfer.</p>",
             "<p>Positive improvement means lower response negative log-likelihood relative to the same model's "
             "untrained baseline. Compare policies within each model; raw token losses are not comparable across tokenizers.</p>"]
    if "message" in status:
        parts.append(f"<p>{html.escape(status['message'])}</p>")
    if store.exists("profile.json"):
        profile = store.get("profile.json")
        parts += ["<h2>Budget and selected profile</h2>", f"<pre>{html.escape(json.dumps(profile, indent=2))}</pre>"]
    if store.exists("clock.json"):
        parts.append(f"<p>Recorded active experiment time: {store.get('clock.json')['elapsed']/60:.1f} minutes.</p>")
    if store.exists("synthetic.json"):
        s = store.get("synthetic.json")
        parts += ["<h2>Synthetic validation</h2>", f"<p>Passed: {s['passed']}. Effect RMSE: {s['effect_rmse']:.4f}. "
                  "This checks known synthetic effects, not language-model results.</p>"]
    if store.exists("policies.json"):
        fitted = store.get("policies.json")
        parts += ["<h2>Frozen mixture policies</h2>", render_table(["Policy", *DOMAINS],
                  [[p, *[f"{v:.3f}" for v in fitted["policies"][p]]] for p in POLICIES]),
                  "<h2>Estimated target-state effects</h2>",
                  render_table(["Domain", "Effect per log-mixture unit"],
                               [[d, f"{v:+.5f}"] for d, v in zip(DOMAINS, fitted["theta"])]),
                  "<h2>Diagnostics</h2>",
                  "<ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in fitted["warnings"]) + "</ul>",
                  f"<pre>{html.escape(json.dumps(fitted.get('diagnostics', {}), indent=2))}</pre>"]
        diagnostics = fitted.get("diagnostics", {})
        if diagnostics.get("outer_grouped_r_loss", 0) >= diagnostics.get("constant_effect_r_loss", float("inf")):
            parts.append("<p>The heterogeneous forest did not outperform the constant-effect baseline on held-out residual "
                         "prediction. These diagnostics do not establish an advantage from state-dependent effects.</p>")
    comparisons = [read_json(p) for p in sorted(store.path("comparisons").glob("*.json"))]
    complete = [r for r in comparisons if r["status"] == "complete"]
    if complete:
        plot_results(store, store.get("policies.json"), complete)
        parts.append('<img src="results.png" alt="Selected mixtures and held-out improvements within each model" style="width:100%;margin-top:24px">')
        parts.append("<h2>Held-out results</h2>")
        for role in ("proxy", "transfer"):
            rows = [r for r in complete if r["role"] == role]
            if not rows:
                continue
            uniform = next((r["test_improvement"] for r in rows if r["policy"] == "uniform"), None)
            parts += [f"<h3>{role.title()} model</h3>", render_table(
                ["Policy", "Dev improvement", "Test improvement", "Test Δ vs uniform", *[f"{d} test improvement" for d in DOMAINS]],
                [[r["policy"], f"{r['dev_improvement']:+.5f}", f"{r['test_improvement']:+.5f}",
                  f"{r['test_improvement']-uniform:+.5f}" if uniform is not None else "pending",
                  *[f"{r['baseline_test_nll'][d]-r['test_nll'][d]:+.5f}" for d in DOMAINS]] for r in rows])]
        parts += ["<h2>Truncation</h2>", render_table(["Model", "Policy", "Encoded examples", "Prompt truncation rate", "Response truncation rate"],
                    [[r["role"], r["policy"], r["truncation"]["examples"],
                      f"{r['truncation']['prompt_truncated']/max(1,r['truncation']['examples']):.1%}",
                      f"{r['truncation']['response_truncated']/max(1,r['truncation']['examples']):.1%}"] for r in complete])]
        by_policy = {(r["role"], r["policy"]): r for r in complete}
        ties = [role for role in ("proxy", "transfer")
                if (role, "causalmix_a") in by_policy and (role, "causalmix_s") in by_policy
                and by_policy[role, "causalmix_a"]["test_example_nll"] == by_policy[role, "causalmix_s"]["test_example_nll"]]
        if ties:
            parts.append("<p>CausalMix-A and CausalMix-S have identical per-example test losses for: "
                         + ", ".join(ties) + ". Nearby weights can produce identical sampled examples at this short budget "
                         "and fixed seed. Do not treat the tie as independent supporting evidence.</p>")
    proxy_runs = [read_json(p) for p in sorted(store.path("runs").glob("*.json"))]
    parts.append(f"<p>Completed proxy runs: {sum(r['status'] == 'complete' for r in proxy_runs)}. "
                 f"Completed final comparisons: {len(complete)}/10.</p>")
    parts += ["<h2>Interpretation and departures from the paper</h2><ul>",
              "<li>Three small task datasets, rather than five Tulu 3 domains and full benchmark suites.</li>",
              "<li>SmolLM2 proxies and cross-family transfer to Qwen; small LoRA updates rather than the published full training recipe.</li>",
              "<li>Pool loss, length, and character diversity replace OpenDataArena quality metrics.</li>",
              "<li>Response likelihood replaces capability accuracy; coding results are not pass@1.</li>",
              "<li>Short sequences can truncate both prompts and answers. Counts above are cumulative unique examples encoded by each model instance.</li>",
              "<li>5% domain floors, 10,000 candidates, shallow forests, and a RegMix-style baseline are toy adaptations.</li>",
              "<li>The analytic positive-effect rule is a heuristic with negative coefficients. It is not claimed to solve an unrestricted global optimization.</li>",
              "<li>One seed and few independent pools do not establish statistical significance. A regression or tie is a valid result.</li>",
              "<li>Public examples may have appeared in base-model pretraining. Split checks prevent leakage introduced by this experiment only.</li></ul>",
              '<p>Source: <a href="https://arxiv.org/pdf/2607.01104">CausalMix paper</a>. '
              'Implementation settings and provenance are preserved in neighboring JSON files.</p>']
    if store.exists("manifest.json"):
        manifest = store.get("manifest.json")
        parts += ["<h2>Provenance</h2>", f"<pre>{html.escape(json.dumps({k:manifest[k] for k in ('models','datasets')}, indent=2))}</pre>"]
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CausalMix local demonstration</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#f6f5f1;color:#20262c;max-width:1120px;margin:48px auto;padding:0 24px}
h1{font-size:42px;line-height:1.15}h2{margin-top:40px}a{color:#146456}.eyebrow{letter-spacing:.12em;color:#146456;font-size:12px}
table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}
th{background:#e4ede8}pre{background:#eaece7;padding:18px;overflow:auto;font-size:12px}li{margin:8px 0}</style><main>""" + "\n".join(parts) + "</main></html>"
    path = store.path("report.html")
    path.write_text(document)
    progress(f"Report: {path}")
    return path
