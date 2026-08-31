# Review Response Map — Brutal verdictLLM.docx → Fixes Applied

Every point from the brutal review, with its resolution status and location.
"Resolved in paper" = the rebuilt manuscript (31_build_paper.py output) addresses it;
"Resolved in data/code" = the underlying experiment/analysis was actually changed.

| # | Review point | Status | Resolution |
|---|---|---|---|
| 1 | Mathematically impossible percentages (Table V: 78%, 75%, 72% at N=2) | **Resolved in data+paper** | All tables regenerated from raw logs by `31_build_paper.py`; per-class table reports raw counts k/n only (no percentages); sample composition disclosed (missing E0/E8). |
| 2 | Contradictory denominators (20 vs 80 vs 600 vs 2400) | **Resolved in paper** | New experimental-accounting table (Table II) with exact denominators; every result cell shows k/n; oracle N=600 labeled as integration test. |
| 3 | Synthetic WLS admitted unfinished while claiming contribution | **Resolved in paper** | Contribution list no longer claims WLS estimation; corpus section states closed-form noise-aware approximation, replacement with `pandapower.estimation.estimate` listed as future work. |
| 4 | Evaluation circular (labels from same rules as prompts) | **Disclosed in paper; expert study future work** | New "Circularity" limitation: evaluation measures recovery of the synthetic labeling policy; leakage audit referenced; required remedies enumerated (expert labels κ≥0.8, multiple valid sequences, blind scoring, independent evaluator). |
| 5 | Headline LLM claims statistically unsupported | **Resolved in paper** | All superiority language ("dominates", "confirming the margin", "consistently leads") retracted; abstract framed as demonstration protocol with exact Ns; Wilson CIs on every estimate. |
| 6 | Mock cannot support LLM claims | **Resolved in paper** | Mock reclassified as harness oracle for plumbing validation; its McNemar isolated in a clearly-labeled paragraph and excluded from all model claims. |
| 7 | Generalization not demonstrated | **Resolved in paper+figure** | Fig. generalization marked PROJECTED (Stage-26 preset ladder); text states no agent ran on 39/118; held-out transfer protocol listed as future work. |
| 8 | Weak physical realism / incomparable limits | **Disclosed in paper; deeper fix future work** | Limits disclosed per system with explicit non-comparability statement ("26% overload under 6% limit carries constructed meaning"); static-corpus limitation stated. |
| 9 | "21/21 validated" misleading | **Resolved in data+paper** | Per-system reporting: 21/21 (14), 20/21 (39, islanding NaNs unresolved), 21/21 (118); corpus table and text carry the breakdown; no global certificate. |
| 10 | Trace placeholder CASE118_SCN_000xxx | **Resolved in paper** | Trace table quotes the actual logged response for a real scenario ID (IEEE14_SCN_002952) from the Gemini run log; invented tool-call narrative removed; states pilot does not execute tools. |
| 11 | References appear fabricated | **Resolved in data+paper** | All unverifiable citations removed; new `references.bib` contains only entries verified against publisher/arXiv records (classics + Majumder 2024 Joule; Cheng 2025 Sci Rep; Grid-Agent; X-GridAgent; ElecBench; L2RPN; Grid2Op). PDF previously had NO rendered bibliography at all — now compiles with bibtex. |
| 12 | Model identity/reproducibility inadequate; "boosted simulation" unclear | **Resolved in paper** | Per-model API strings, decoding settings, throttle/retry policy, access dates, parse policy documented; "boosted simulation" removed; Muse Spark 1.2 explicitly declared not evaluated (simulated preset rows excluded from all numbers). |
| 13 | Hallucination scoring subjective/underdefined | **Disclosed in paper; annotation future work** | Metrics section defines response-level any-of-6-tag scoring by a single automated rule-based judge; explicit statement that rates = judge agreement, not ground truth; expert double-annotation listed as required remedy. |
| 14 | RAG evaluation too weak (Recall@1 on 8 docs) | **Disclosed in paper** | Recall@1 characterized as plumbing check; retrieval config stated (384-d, k=3); hard-negative evaluation listed as future work. |
| 15 | Missing baselines | **Resolved in data+paper** | Deterministic baselines computed on the identical scenario sample (`data/results/baselines.json`): majority-class 20%, random-uniform 12.5±7.4%, random tool choice 53%, degenerate always-power-flow 100%; reported in Results. |
| 16 | Safety framing inappropriate | **Resolved in paper** | Reframed as offline advisory research prototype; no switching/protection/authorization claims; closed-loop out of scope. |
| 17 | Reproducibility vs "on request" | **Partially resolved** | Paper no longer states "on request" as the mechanism; deterministic regeneration from seeds documented; archival DOI release planned before submission (action item outside the manuscript). |
| 18 | Presentation problems | **Resolved in paper** | Shorter title; source filenames removed from prose; LaTeX artifacts (`\textasciitilde`, `\{,\}`, placeholder `\textless`) removed; free-tier project notes removed; figures carry evidentiary-status annotations. |

## Additional fixes beyond the review (found during rework)

- **Severity formula mismatch (old issue #5)**: Eq. (sev) now matches the code (`S = 0.6·min(1, max(0,|V−1|−0.02)/0.06) + 0.4·min(1, max(0, ℓ−L)/10)`); Stage 08 rerun on the full 15k corpus.
- **Severity ρ disclosure (old issue #4)**: per-case ρ = 0.764/0.968/0.823 now computed from WLS state-estimate voltages (not injected noise); quantified label-noise bound: 0/30,000 accepted-set flips on IEEE-14, 31/150,000 (0.021%) corpus-wide (`08b_severity_label_noise.py`, `data/results/severity_label_noise_bound.json`).
- **Tool-metric degeneracy (new)**: permissive accepted-set tool metric shown to be attainable at 100% by always answering power_flow; dual-metric reporting introduced (permissive upper bound + strict-specific main metric), all computed post-hoc from logged raw responses.
- **Duplicate-append checkpoint bug** in the Gemini runner fixed (file had 5,690 rows for 410 unique cells; duplicates verified byte-identical, then deduplicated); quota-exhaustion now stops the run cleanly instead of writing poisoned error rows.
- **Local-model harness (`30_local_pilot_resilient.py`)**: crash-resilient (per-row durable writes, health-poll resume, indefinite patience through GPU overheat crashes); fixed thinking-model contract (max_tokens 1024, content→reasoning_content fallback, last-E-digit parse; 79/80 clean JSON).
- **Figure provenance**: all figures regenerated with sample sizes, Wilson CIs, PROJECTED/simulated annotations; Muse simulation and mock removed from result figures.

## Post-review reframe (v2, texflow build)

The paper was subsequently reframed from "pipeline + pilot" to its strongest defensible story, per author decision:

- **New title**: "Do Small Local LLMs Match API Models for Power-Grid Situation Awareness? A Reproducible Benchmark on IEEE 14/39/118"
- **New hook (F1)**: local Gemma-4-E4B Q4_0 diagnoses 105-108/140 vs API 109/140 — McNemar n.s. in all configs, **non-inferior at a declared 10pp margin in all four configurations** (bootstrap CIs), fails only the stricter 5pp margin in E3 — reported with the margin declared up front.
- **New finding (F2)**: under the strict-specific tool metric the LOCAL model outperforms the API model (63-64/140 vs 35-53/140); the permissive metric's 100% degenerate strategy disclosed.
- **New finding (F3)**: latency is the trade-off (~40x), against $0 API cost, no rate-limit dependence, and on-premises data.
- **Muse Spark removed entirely** (author decision); mock retained only as harness oracle.
- **Quantization limitation added** with verified citations (Dettmers & Zettlemoyer ICML 2023; GPTQ ICLR 2023): Q4_0 results are a conservative lower bound.
- **Scope limitation added**: one model pair, one corpus, single runs — no class-level "local models are sufficient" claim.
- Final build produced through the **texflow MCP** (IEEE conference class) in `texflowmcp/workspace-gridpower`; visual acceptance: 6/6 pages pass.

## Reframe v3 — back to the proposal (final)

A second review correctly argued the benchmark framing oversold an underpowered pilot and buried its own insights. The paper was re-anchored on Research Proposal 1:

- **Title**: "GridPowerAgent: A Grid-Aware LLM Agent for Power System Event Understanding and Tool-Orchestrated Decision Support" (agent paper, not benchmark paper).
- **Research questions restored**: RQ1 diagnosis, RQ2 tool selection, RQ4 RAG effect, RQ5 hallucination are now the organizing structure of the Results; RQ7 generalization explicitly stated as planned-not-performed.
- **Power honesty**: minimum-detectable-difference analysis added (2.8/2.0/3.9/2.8pp per config at the observed discordance rates); NI analysis downgraded to exploratory with post-hoc margins disclosed; null results stated as "no detectable difference above the MDE".
- **E6/E8 failure mode solved and promoted**: 0% on E6/E8 traced to a mixed-axis label taxonomy (E6 undervoltage injected via line outages → models answer E3 in 128/144 failures; E8 thermal overload injected as compounds → models answer E9 in 104/104). Logged model reasons quoted verbatim. Framed as a benchmark-design finding, with both-axes reporting queued for the powered sweep.
- **Tool metric reframed**: strict-metric difference attributed to default-answer bias (API model answers power_flow in 52% of responses), not capability.
- **Generalized conditions**: free-tier quota narrative, thermal-pacing details, consumer-GPU/llama.cpp specifics removed from the body; models described by tier and decoding contract only; explicit scope sentence that the API tier is lightweight and results do not speak to frontier API models.
- **IEEE-39 NaNs**: count corrected to 41 (24 from line_16_19 islanding + 17 from line_23_36; 31 E3 + 10 E9); identifier list shipped as data/case39_nan_scenarios.csv and referenced from the paper.
- **Severity circularity layer added**: boundary grid-search performed on the same corpus, disclosed in Limitations.
- **Projected generalization figure removed** entirely; transfer stated as planned-not-performed.
- **Double-blind**: Acknowledgment removed; AI disclosure softened; author "Anonymous Authors" throughout; PDF metadata carries no author.
- Final build: texflow MCP (IEEE conference class), 6 pages, visual acceptance re-verified after fixes.

## v3.1 precision pass (second review round)

- **Table IV de-vestigialized**: NI 10pp/5pp pass-fail columns removed (margins were post-hoc); the table is now a descriptive paired summary (Cfg / Pairs / Diff (pp) / bootstrap 95% CI / exact McNemar p) with an explicit "no pass/fail is claimed" caption. Fig 4(b) margin lines removed; the MDE paragraph in the Protocol is the single power statement.
- **"Sec. ??" fixed**: raw-table blocks now pass through the reference-substitution pass; Table III caption reworded to avoid the ref.
- **"2.8pppp" unit-doubling fixed** (mde_txt unit removed; prose carries the unit once).
- **RQ3 explicitly deferred** (pilot does not execute tools); RQ4 promoted into the subsection title ("RQ1/RQ4: Event Diagnosis and the Effect of Retrieval"); RQ5 subsection labeled.
- **RQ2 default-bias claim strengthened with measured conditioning**: when the API model states power flow it is actually required in only a minority of those cases (data/results/tool_bias_analysis.json); the local model's contingency picks are predominantly required (contingency is required exactly on outage/overload scenarios).
- **Abstract rewritten**: label-axis finding promoted to the lead result; strict-metric qualifier attached to the tool percentages; awkward fragment fixed.
- **Citation numbering fixed**: in-text [n] now derived from the compiled .bbl order (was offset +2 by a hardcoded order); verified in render ([3,4] agentic, [22,23] quantization, [24] Gemma 4, [25] llama.cpp).
- Visual gate: 7/7 pages pass.
