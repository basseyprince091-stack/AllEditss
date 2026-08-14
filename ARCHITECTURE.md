# ALLEDITS — Architecture

## The core bet

ALLEDITS is not a video generator. It is an **editorial reasoning engine** that
operates on the user's own footage. The product promise is *professional editing
without knowing how to edit*, so the system must make defensible editorial
decisions and be able to explain them.

Everything follows from one rule:

> **The model decides. It never renders.**

An LLM (or the rule-based planner standing in for it) emits a **Timeline** — a
typed, versioned, validated data structure. A renderer turns that structure into
pixels. No model output ever becomes a shell command.

```
media ──► MEDIA BRAIN ──► shots + analysis + embeddings ──► INDEX
                                                              │
reference ──► REFERENCE ANALYSIS ──► StyleGrammar ────────┐   │
                                                          ▼   ▼
music ──► AUDIO ANALYSIS ──► beat grid ──────────────► PLANNER
                                                          │
                                                   slot plan (beat-locked)
                                                          │
                                                      SELECTOR  ◄── decision ledger
                                                          │
                                                       BUILDER
                                                          │
                                                    Timeline DSL v1
                                                          │
                                                     VALIDATOR  ── hard gate
                                                          │
                                                      RENDERER (segment cache)
                                                          │
                                                   preview ──► CRITIC
                                                          │        │
                                                          └── revise (bounded)
                                                          │
                                                    final render
```

## Layers

**`core/`** — storage namespaces (raw/proxy/analysis/render/output), job queue
abstraction, the decision ledger, and the only place ffmpeg is invoked.

**`media/`** — the Media Brain. Probe → proxy → shot detection → per-shot visual
analysis (optical flow, composition, colour) → dual quality scoring → embedding.
Analysis is cached by content fingerprint, so re-ingesting a file is free.

**`audio/`** — music intelligence built from first principles on numpy/scipy:
spectral-flux onsets, autocorrelation tempo, phase-locked beat grid refined by
onset snapping, energy curve, self-similarity section boundaries, drop detection.

**`reference/`** — extracts an editing *language* (StyleGrammar): pacing
distribution, rhythm class, intensity curve, transition tendencies, grading
direction, narrative arc. Stores characteristics, never content.

**`intelligence/`** — planner (grammar + music → beat-locked slots), selector
(explainable ranking with continuity reasoning), critic (measures the rendered
file), and swappable model providers.

**`timeline/`** — the DSL, the builder, and the validator that gates rendering.

**`render/`** — the `Renderer` interface and the ffmpeg implementation. A
Remotion/WebGL renderer can implement the same interface for motion graphics.

## Three decisions worth knowing

**1. Timeline duration is authoritative, not derived.**
Computing clip length from `(source_out - source_in)/speed` accumulates rounding
error and silently walks a long edit off the beat. The plan owns duration; the
source range says what to read into it. Measured effect: beat alignment went from
20% to 85%.

**2. Rendering is segment-based, not one giant filter graph.**
Each clip renders to a normalized intermediate keyed by a content hash. A
revision re-renders only what changed, which is what makes the critique loop
affordable. It also isolates failures to a single clip.

**3. The critic inspects the render, not the plan.**
It re-measures the output file the same way it measured the inputs — recovering
actual cut positions, actual beat error, actual luminance jumps. That is how it
catches what went wrong in the *render* rather than what was supposed to happen.
It distinguishes "cuts are too slow" from "cuts are invisible because adjacent
shots look alike" — different diagnoses needing opposite fixes.

## Replaceability

Models, renderers, storage and the job queue are all interfaces. Swapping in a
real LLM, a GPU renderer, S3, or a distributed worker pool is a provider change,
not a rewrite. Every provider declares its capabilities honestly; a capability
that isn't installed raises rather than degrading into plausible nonsense.
