# ALLEDITS — Implementation Ledger

Required by the master specification. This is the honest account of what exists,
what is stubbed, what is deferred, and what was added. **Nothing in the codebase
is presented as more complete than it is.**

Phases complete: **0 (vertical slice)**, **1 (brief → creative constraints)**,
**2 (project state + human override)**, **3 (footage rescue)**, **4 (sound)**,
**5 (FIND)**, **6 (MASTER)**, **7 (AUTOPILOT)**, **8 (STYLE)**, **9 (DIRECTOR)**, **10 (anchoring)**,
**11 (model orchestration)**, **12 (nothing is wasted)**,
**13 (shoot assistant)**, **14 (creator profile)**, **15 (async jobs)**.

Every mode named in the specification is now implemented and verified.

Status: **the pipeline runs end to end and produces a real rendered video, the
brief measurably changes the edit, and human directives outrank the system.** Verified output: `1080x1920 @ 30fps, 18.00s`, H.264/AAC,
16 MB, from 16 source clips + 1 reference + 1 music track.

---

## 1. IMPLEMENTED AND VERIFIED

Each item below was tested against generated media with known ground truth
(`scripts/make_test_media.py`), not merely written.

| Capability | Evidence |
|---|---|
| Beat/tempo detection (spectral flux → autocorrelation → phase lock → onset snap) | 129.2 BPM measured vs **128.0 ground truth** (0.9% error); 16 downbeats for 16 bars |
| Musical drop detection | detected 9.33s vs **9.375s ground truth** (45 ms error) |
| Section segmentation (self-similarity novelty) | boundary placed exactly at the drop |
| Shot boundary detection (content-aware + prominence gate) | 3-shot take → **exactly 3 shots**; continuous pan → **1 shot** (no over-segmentation) |
| Camera movement classification (dense optical flow) | **13/13** across two corpora: static, pan L/R, tilt up/down, push in, pull out, handheld |
| Reference pacing extraction | 1.61 cuts/s measured vs **1.64 ground truth**; rhythm class "accelerating" correct |
| Reference transition inference | recovers inserted flash transitions; distribution-based (peak concentration), not magnitude-based |
| Two-score quality model (technical ⟂ creative) | 4K clip → `use` (left alone); degraded clip → `replace`/`use_briefly` |
| Subject-aware reframing 16:9 → 9:16 | crop placed on detected subject, e.g. `(0.42, 0.51)`, not frame centre |
| Beat-locked slot planning | 31 slots, cuts quantized to the measured grid |
| Explainable clip selection with continuity scoring | ledger: *"preferred because its movement and composition match the preceding shot (0.70 vs 0.62)"* |
| Transition mix reproduction | output ≈20% flash / 13% dissolve / 10% whip vs reference 20/15/10 |
| Timeline DSL v1 + hard validator | 5 validator tests; render is impossible without passing |
| FFmpeg renderer w/ segment caching + beat-centred transitions | incremental re-render on revision confirmed |
| Self-critique on the **rendered file** | pass 1 → 7.4/10 with issues; pass 2 → **8.4/10, no issues** |
| Revision loop measurably improves the edit | indistinct cuts 22/30 → **29/30**; beat alignment 20% → **85%** |
| Decision ledger | 292 recorded decisions with rationale, alternatives and evidence |

### Phase 1 — brief → creative constraints

| Capability | Evidence |
|---|---|
| `CreativeConstraints`: ~20 typed, clamped knobs | empty brief provably reproduces pre-Phase-1 behaviour |
| Lexicon parser (~120 terms, intensifiers, negation, phrases) | clause-scoped negation; "not too fast, very warm" no longer inverts *warm* |
| LLM parser emitting the same schema, then clamped | falls back visibly, never silently (test) |
| Constraints reach planner / selector / builder / critic | see differential below |

**A/B proof — same footage, reference and music; only the brief differs
(`scripts/ab_brief_test.py`, 10/10 checks):**

| metric | "cinematic, slow, restrained" | "chaotic, aggressive, fast" |
|---|---|---|
| clips | 12 | 39 |
| mean shot | 1.50s | 0.46s |
| cuts/sec | 0.67 | 2.17 |
| effects per clip | 2.0 | 2.85 |
| dissolve share | 0.27 | 0.00 |
| flash+whip share | 0.18 | 0.45 |
| shake effects | 0 | 18 |

Both were fully rendered; these are timeline measurements from real runs.

### Phase 2 — project state and human override

| Capability | Evidence |
|---|---|
| `Project` persisted atomically; reopen and continue | round-trip test; `alledits project` |
| Directives: pin, reject (global/per-slot), lock, force transition, force/ban effect | one test each |
| Directives are stored against SLOTS, so they survive re-plan and revision | verified across a second full CLI run |
| **A lock outranks the critic** | critic revision provably cannot alter a locked clip's duration or effects |
| Pinned choices are not credited to the system | provenance test on `selection_reason` |
| Incremental re-solve | second run reused cached segments; only affected clips re-rendered |

Verified through the real CLI end to end: pin applied to slot 3, `ban_effect
shake` removed all 18 shake effects, 3 segments reused from cache.

### Phase 3 — footage rescue

Quality analysis previously *described* a defect and did nothing about it. This
phase acts on it — with targeted, per-defect treatment rather than a blanket
"enhance" pass, because applying a treatment footage doesn't need degrades it.

| Capability | Evidence |
|---|---|
| `defects: [{defect, severity, treatment, detail}]` from measured signal | every defect carries the measurement that justifies it |
| denoise / sharpen / deblock / stabilize / expand_contrast | measured before→after, below |
| Restoration ordered before creative effects; denoise before sharpen | chain-order test (sharpening noise amplifies it) |
| Two-pass `vidstab` (detect → transform) in the renderer | stabilization cannot be a single filter |
| Stabilize gated on LOW directional consistency | a deliberate fast pan is not "wobble"; test asserts it is left alone |
| Treatments recorded on the clip | provenance: output is never presented as untouched original |

**Measured on real renders (`scripts/rescue_test.py`, all checks pass):**

| treatment | metric | before | after |
|---|---|---|---|
| sharpen | sharpness | 45.6 | 140.4 |
| deblock | blockiness | 1.000 | 0.147 |
| denoise | noise | 0.335 | 0.298 |
| stabilize | shake | 3.87 | 0.15 |

**Restraint (the harder half):** `11_4k_high_quality`, `01_static_wide` and
`02_pan_right` each receive **zero** treatments.

Two defects found and fixed while proving this:
- Defect detection was gated on the *aggregate* handling verdict, so a clip that
  scored well overall but carried one strong fixable defect received nothing.
  Defects are now judged on their own merits; thresholds protect good footage.
- The noise estimator was miscalibrated. Measuring the metric's distribution
  across the whole corpus showed clean footage reading 0.08–0.14 and overlapping
  with genuinely noisy material. Tightening the flat-region mask from the 55th to
  the 25th percentile puts clean footage at 0.000 against 0.190 for real noise.

A third defect was found in the *tests themselves*: two Phase 1 tests reused the
names of existing tests, and because collection reads `globals()`, each pair
collapsed into one — silently dropping coverage of colour-delta neutrality and of
clamping via stacked intensifiers. The tests were merged rather than deleted, and
the runner now fails hard on duplicate test names, since a test suite that
quietly shrinks is worse than one that fails loudly.

**Test suite: 48/48 passing** at the end of Phase 3.

### Phase 4 — sound

`plan_mix`, `validate_mix` and the renderer's mix graph already existed but
nothing ever *called* them, so every edit shipped a naive full-volume music bed.
This phase wires the mix in, gates it with the validator exactly as the picture
is gated, and re-decides it whenever the critique loop rebuilds the timeline.

| Capability | Evidence |
|---|---|
| Loudness normalised to a platform target | measured on the encoded file, below |
| True peak held under the ceiling after lossy encoding | measured, below |
| Music ducked under diegetic speech | measured attenuation, below |
| Speech detected from the clips' own audio | music correctly reads as non-speech |
| Mix plan hard-validated before render | invalid plan raises rather than silently downgrading |

**Measured on the encoded deliverable (`scripts/sound_test.py`):**

| | source | delivered | target |
|---|---|---|---|
| loudness (social) | -17.5 LUFS | **-14.3** | -14.0 ± 0.5 |
| loudness (broadcast) | -17.5 LUFS | **-23.1** | -23.0 ± 0.5 |
| true peak | +0.6 dBFS | **-1.4** | ≤ -1.0 |

Three defects found by measuring rather than trusting the plan:

- **`loudnorm`'s dynamic mode lands short.** It protects peaks by backing the
  whole programme off, measuring -15.4 LUFS against a -14.0 target. Replaced with
  gain-to-target plus a limiter, solved by measurement and verified by a second
  measurement, because the loudness cost of limiting cannot be predicted from the
  input. An intermediate diagnosis — that the target was physically unreachable
  under the ceiling — was **wrong**, and the sweep disproved it.
- **A limiter set at the ceiling ships over it.** AAC at 192k reconstructs peaks
  1.2–2.0 dB above the samples handed to it, so the limiter now sits a measured
  margin below the delivery ceiling.
- **The duck depth was fiction.** `depth_db` was reported to the user, validated
  and recorded, but never reached the filter, which used a fixed ratio: the
  system announced 9 dB while applying roughly 18. A first analytical fix was
  also wrong — sweeping ffmpeg showed attenuation is governed almost entirely by
  *threshold* (ratio 2→20 moved it under 1.5 dB) and saturates just under 7 dB.
  Parameters are now calibrated from measurement, and a request beyond
  saturation is reported as capped instead of claimed.

| duck asked | delivered |
|---|---|
| 3 dB | 4.1 |
| 6 dB | 7.1 |
| 9 dB | 7.7 (saturated — stated as capped) |
| 12 dB | 7.7 (saturated — stated as capped) |

#### Speech-aware in-point selection

Wiring the mix in exposed a weakness in *picture* selection rather than sound.
In a live run the duck never engaged, and it was right not to: the
middle-of-shot rule had chosen a 0.47s range from the one talking clip ending
10 ms before the dialogue began, so the clip contributed silence and the mix had
nothing to duck for. Correct behaviour, wasted dialogue.

`_pick_in_point` now centres on the speaking window when a shot carries speech,
and is unchanged for silent footage. Detection is cached per source file and any
failure degrades to the previous rule rather than breaking the build. After the
change the same run reports `2 track(s), 1 carrying speech` and the duck engages.

A provenance defect surfaced alongside it: `Timeline.to_dict()` never emitted the
mix, so every sound decision — including the measured delivered loudness — was
discarded on save. Now serialised and covered by a test.

**Test suite: 77/77 passing** (`python3 tests/test_alledits.py`).

### Known limitations (Phase 4)

- The duck saturates just under 7 dB with this sidechain; deeper requests are
  reported as capped rather than claimed.
- The duck saturates just under 7 dB; deeper requests are reported as capped.

#### Loudness solver — fixed

The -14.7 LUFS miss was not a duck-ordering problem. Two bugs in the solver:

- It returned the gain produced by the LAST correction, which was **never
  measured** — the render applied a gain that had not been verified.
- That value could exceed the make-up clamp, so the render silently used the
  clamped gain while the plan recorded the unclamped one (+19.1 dB recorded,
  +18.0 applied).

The loop now keeps only a (gain, loudness) pair measured *together*, clamps
before returning, and iterates to the EBU tolerance. The clamp was also raised
to 24 dB: a speech-led mix (music bedded at -10 dB, voice intermittent) is
legitimately quiet and was hitting the ceiling.

Verified end to end on the case that failed: solver reports -14.2 LUFS, the
delivered file measures **-14.3 LUFS at -1.3 dBTP** — and the plan's claim now
matches delivery to 0.1 LU.

### Phase 5 — FIND

**Capability honesty first.** There is no vision-language model in this build,
so open-vocabulary semantic search is impossible. Rather than keyword-guess over
feature vectors and call it semantic, FIND maps a query onto attributes the
analysers genuinely measured and executes it as a typed, inspectable filter.
`search_by_text()` still refuses outright; every FIND result carries
`semantic: False`.

| Capability | Evidence |
|---|---|
| Natural language → typed criteria over measured signal | camera move, shot size, exposure, colour, motion, quality, faces, speech |
| Every result explains itself in measured terms | "brightness below 0.35; camera movement is static" |
| Clause-scoped negation | "no handheld, dark" does not invert *dark* |
| Synonyms collapse to one condition | "talking head interview with dialogue" = one speech criterion |
| Un-understood terms are reported | not silently dropped |
| Unmeasurable criteria declared inert | see below |

**Retrieval measured against corpus ground truth (`scripts/find_test.py`):**
mean recall **1.00** over 10 queries naming known clips (pan right, tilt up,
handheld, dark low-key, bright high-key, talking head, poor quality, …).

Two defects found in the *proof harness itself*, both of which would have let a
weak result look strong:

- Ground truth never matched, because ingest renames clips to `asset_<hash>`.
  Every query scored 0.00 until the mapping was captured at ingest time.
- The negation test ran against faces, and the corpus contains **zero** faces
  anywhere — so it passed vacuously, demonstrating nothing. It now uses an
  attribute that actually varies, and fails loudly if the positive query matches
  nothing.

That second one exposed a real product gap: `shot_size` is `unknown` for every
clip in the corpus, so "wide" and "close up" can never match. Returning nothing
would read as *"no wide shots exist"* — a different and misleading claim. FIND
now reports such a filter as **inert** rather than applying it silently.

### Phase 6 — MASTER

A finished edit is not a deliverable. Each destination has its own container,
codec, resolution, frame rate and loudness expectation; missing any of them means
the platform re-encodes and undoes work that was measured upstream.

| Capability | Evidence |
|---|---|
| Six delivery profiles (Shorts, TikTok, Reels, YouTube 1080p, EBU broadcast, preview) | coherence test over every profile |
| Mastering transcode fits-and-pads, never stretches | a 16:9 source to 9:16 is letterboxed, not distorted |
| QC re-measures the encoded file against the contract | 11 checks: resolution, fps, codec, pix_fmt, audio, duration, loudness, true peak, faststart |
| QC can FAIL | vertical file vs broadcast profile fails on resolution, fps and loudness |
| Unmeasurable checks report SKIP, never PASS | a tick derived from ignorance is false assurance |
| Upscaling refused unless permitted, disclosed when allowed | `resolution_provenance` |
| Audio re-normalised on transcode | a new codec at a new bitrate reconstructs peaks differently |

Verified end to end: `alledits edit --deliver tiktok` produces a conformant
1080x1920 H.264 master at **-14.4 LUFS / -1.6 dBTP**, with the full QC report
persisted into the project for audit.

Two defects found while proving it:

- **`plan_scaling` contradicted the encoder.** It compared `max()` of the
  dimension ratios while the encoder fits *inside* the frame and pads, so a
  1080x1920 edit delivered to 1920x1080 was announced as a 1.78x upscale when the
  picture is in fact downscaled and letterboxed. A false disclosure erodes trust
  in the real ones exactly as much as a missing disclosure does.
- **The pipeline mastered the preview.** The delivery step used `res.path`, which
  still held the last half-scale preview from the critique loop, so
  `--deliver` silently shipped an upscaled 540x960 preview instead of the
  finished edit. It now masters `fres` — the final render.

### shot_size — investigated, deliberately NOT shipped

FIND declares framing filters inert because `shot_size` is only derived from face
detection, and is `unknown` on faceless footage. Three faceless estimators were
built and measured against ground-truth footage before this was left alone:

| approach | result |
|---|---|
| saliency bbox extent | tracked synthetic subjects (0.47→0.54, 0.22→0.23, 0.08→0.13) but returned 1.000 on real footage — a sprawling merged blob has a tall bounding box and no subject in it |
| blob compactness | no separation: synthetic close-up 0.296 sits inside the 0.206–0.316 band of subject-free corpus frames |
| depth-of-field ratio | 1.37 close-up vs 1.17 wide, while corpus frames ranged 1.26–4.53 |

A wrong framing label is worse than none: it would silently mis-select shots and
make FIND confidently return the wrong footage. `shot_size` now carries
`shot_size_basis` and `shot_size_confidence` so consumers know the provenance,
and the negative result is recorded in the source so it is not re-attempted.
Reliable faceless framing needs person/subject segmentation — deferred.

### Phase 7 — AUTOPILOT

Given footage and music and no brief, the system explores several genuinely
different treatments, critiques each rendered result, and delivers the winner.

| Capability | Evidence |
|---|---|
| Candidates explored at PREVIEW scale | `stop_after_preview` — full raster per discarded candidate costs minutes |
| Scored by the existing critic on the RENDERED file | a candidate cannot win on a persuasive plan |
| Analysis shared across candidates via one workdir | ingest/quality/music computed once |
| Candidates are genuinely different | test asserts pacing varies by >1.5x |
| A failed candidate is reported, not hidden, and cannot win | |
| Every candidate stays auditable, not just the winner | |
| Optional delivery in the same pass | `--deliver` |

Measured run (8s, three treatments): restrained **7.4** (5 clips, 0.63 cuts/s),
balanced **7.4** (13 clips, 1.64 cuts/s), energetic **4.8** (17 clips, 2.14
cuts/s). Winner re-rendered at full scale and mastered.

One defect found in this phase was mine: autopilot reported the full
max-minus-min **spread** (2.56) as "a clear preference" when the top two
candidates had *tied*. The rejected candidate being poor says nothing about the
winner being right. Decisiveness is now the **winner-versus-runner-up margin**,
and that run correctly reports itself as too close to call.

### Phase 8 — STYLE: blending and reuse

| Capability | Evidence |
|---|---|
| Grammar deserialisation (`from_dict`/`from_json`) | exact round-trip test |
| A stale grammar version is refused, not defaulted | filling gaps with defaults would present a stale style as current |
| Blend N references by weight | interpolation and weight-shift tests |
| Categorical qualities are NOT averaged | no midpoint exists between "accelerating" and "steady" |
| Source disagreement disclosed in `notes` | a blend whose sources disagreed is a weaker claim |
| Transition shares renormalise to a distribution | sums to 1.0 after averaging |
| Curves of different lengths resampled before averaging | zipping would silently truncate the longer |
| Named style library, reusable without the reference | save / list / show / blend / delete |
| `edit --style` runs with **no reference file at all** | verified end to end |

Measured blend of a fast reference (1.61 cuts/s, accelerating) and a slow one
(0.35 cuts/s, steady):

| weights | cuts/s | mean shot | rhythm |
|---|---|---|---|
| fast only | 1.613 | 0.59s | accelerating |
| fast 3 : slow 1 | 1.296 | 1.04s | accelerating |
| fast 1 : slow 1 | 0.980 | 1.50s | accelerating |
| fast 1 : slow 3 | 0.664 | 1.95s | steady |
| slow only | 0.347 | 2.40s | steady |

The blended `house` style then drove a real edit with no reference supplied: the
rendered result measured **0.62 cuts/s against the style's 0.66**, scoring
8.1/10.

**A stored style is a measurement, not a copy.** It holds no frames, no audio and
no path back to the source work — asserted by a test that scans the saved JSON
for anything resembling a source reference. That is what makes reusing a learned
style safe: it never redistributes the footage it was learned from.

### Phase 9 — DIRECTOR

The mode where someone watches the cut and says what is wrong with it. Notes
become DIRECTIVES against the project — auditable, reversible, surviving a
re-plan — never edits to a rendered file.

| Capability | Evidence |
|---|---|
| Positional reference ("the third shot", "clip 2", "the last one") | test |
| Attribute reference resolved from MEASURED signal ("the shaky one") | picks the highest measured shake |
| Whole-piece notes routed to the shared brief vocabulary | "punch it up" |
| Multi-clause notes | "hold the first shot longer, no flashes" |
| Out-of-range references refused | "the ninth clip" of a 4-clip edit |
| Absent signal does not get a guess | "the shaky one" with no shake data resolves to nothing |
| Everything unactionable is reported | a half-applied note is the dangerous case |
| `--dry-run` shows the plan without saving | |

`SET_DURATION` was added as a real directive kind, because "hold that longer"
— the most common note there is — could not previously be expressed at all.
Positions come from the beat plan, so a longer clip re-flows everything after
it. That knocks later cuts off the musical grid, so those clips **drop their
`beat_locked` flag** rather than leaving the timeline asserting a sync it no
longer honours, and the total-duration change is reported as a warning.

Verified end to end: `alledits note --text "hold the second shot longer"` on a
saved project, then a rebuild — clip 1 grew 1.83s → 2.73s, downstream clips
shifted, and their beat-lock was correctly dropped.

**A silent no-op found by a test.** DIRECTOR routed whole-piece notes to the
brief parser, but six of its sixteen global hints (`punch`, `calmer`, `warmer`,
`colder`, `moodier`, `brighter`) were absent from the brief lexicon. It reported
"re-plan with 'punch it up'" and the re-plan changed *nothing* — a note that
appeared applied and was not. Fixed on both sides: the missing vocabulary was
added, and DIRECTOR now **verifies the brief will actually act** before
promising it will, reporting the phrase as unresolved otherwise. `brighter`
remains unresolved on purpose — there is no brightness knob, and inventing one
that does nothing is the bug being fixed. A guard test fails if any hint ever
becomes a no-op again.

While fixing it I reproduced the same class of error twice more in my own
additions: the lexicon convention is a *signed value* on `+field`, not a
`-field` operator, so `"-warmth_delta": 0.25` was silently ignored; and a
`brightness_delta` entry referred to a field that does not exist.

### Phase 10 — content-anchored overrides

Directives were slot-indexed, so a re-plan silently moved them onto footage the
human never looked at. Each directive now records WHAT was on screen when the
note was made.

| Behaviour | Result |
|---|---|
| shot still at the same slot | applied (`exact`) |
| shot moved elsewhere in the edit | directive follows it, move **reported** |
| shot no longer in the edit | `lost` — **not applied**, and reported |
| directive saved before anchoring existed | legacy slot behaviour, unchanged |

Retargeting quietly is worse than failing: the user sees an unrelated clip
change and cannot tell why.

### Phase 11 — model orchestration (Spec §21)

A capability registry for all eight capabilities this build cannot serve, each
with an abstract interface, the reason it is unavailable, and what it unlocks.
It ships **empty (0/8)** — pre-registering a stub would make the registry lie.

Proven rather than asserted: registering a stub provider enables open-vocabulary
semantic search **end to end with no change to the editing engine**. A provider
whose `available()` raises counts as absent. A connected model with an
un-embedded library reports a *different* failure from a missing model.
`alledits capabilities` prints the whole picture.

### Phase 12 — "nothing is wasted" (Spec §25)

Poor footage is offered named creative roles (flash frame, transition, rapid
montage, texture, background), each with a duration cap, instead of competing as
a hero shot. Genuinely unusable footage — black, blown, frozen — is still
rejected, or the principle means nothing.

Verified with a deliberately scarce pool (4 clips, 2 poor): both poor clips were
used at **0.47s each**, marked `use_briefly`, zero over-cap violations. With a
healthy pool they are eligible but not chosen, which is correct — salvage is a
fallback, not a preference.

Four defects found while proving it:

- The cap took the **max across all roles**, so a flash-frame clip inherited a
  background plate's 2.5s cap. Caps now come only from roles the renderer can
  actually perform; `texture` and `background` need layered compositing that
  does not exist, and are named as blocked rather than silently dropped.
- Trimming the source range was not enough — a salvaged clip still *occupied* a
  long slot and held a frozen frame for the remainder. The cap is now enforced
  at **selection**.
- The 0.45s cap was a number I picked arbitrarily, and it sat just under **one
  beat** (0.465s at 129 BPM). Cuts quantise to the beat grid, so salvage was
  ineligible for every slot in the edit at any tempo above ~133 BPM. The cap is
  now one beat at dance tempo, which is also what "rapid montage" means.
- **Analysis was cached by content fingerprint alone.** Changing an analyser
  silently reused stale results, so the new salvage logic would never have run
  on any previously-indexed clip. The cache is now keyed by fingerprint AND
  analyser version.

Punctuation slots were added so short stabs exist for salvage to fill, gated on
whether the reference *itself* punctuates — measured as its shortest shots being
distinctly shorter than its median, not an absolute cut length. Both test
references have uniform shot lengths (p10/median 1.00 and 0.92), so **no
punctuation fires on them**, which is the gate working rather than a feature
delivering.

### Phase 13 — shoot assistant and coverage gaps (Spec §5, §24)

The first mode that answers the question *before* editing: what should I film?

`alledits shoot plan` produces per-shot direction following Spec §5's checklist
— placement, height, distance, subject position, action, gaze, camera move,
record duration, settings, DO NOTs, and what the shot becomes in the edit.
`alledits shoot coverage` compares that plan against footage already in hand and
prints recording instructions for whatever is missing.

**Shot lengths derive from the style, not a template** (Spec §24: "exact plan
must depend on reference/style"). The same "Detail" shot asks for a 4s take
under the fast reference and 7s under the slower blended `house` style. Record
time is always ~3x the implied cut length, because a clip trimmed to exactly its
cut length has no handles.

**The honesty boundary is the design.** A shot requirement has a measurable part
(camera move, sharpness, energy, duration — checked by reusing FIND's criteria)
and a semantic part ("the ball rolls into frame") that needs a vision-language
model this build does not have. So coverage never reports a shot as *covered*:

| status | meaning |
|---|---|
| `MISSING` | nothing has the right measurable shape — go and film it |
| `LIKELY` | the shape matches; a human must confirm the content |
| `UNVERIFIABLE` | the requirement is not measurable on this library at all |

`LIKELY` always names what was not verified. Reporting "covered" on a camera-move
match alone would send someone into an edit believing they had a shot they never
filmed.

The INSPECT step judges a newly recorded clip against its own spec: a handheld
take fails a static direction; good footage is approved without demanding a
retake (Spec §5); footage with *fixable* defects is approved with a note that
enhancement will treat it, rather than triggering a needless reshoot.

One defect found while proving it: salvage-grade footage (usable ~0.5s as a
flash) was **approved** for a shot the plan wants held for seconds — telling
someone they had the shot when they had a flash frame. It is now a reshoot with
the cap explained.

### Phase 14 — creator profile and adaptive guidance (Spec §4)

The same shot has to be described very differently to two people. Instruction
text is therefore generated from the profile rather than stored as one string.

| Capability | Evidence |
|---|---|
| Four guidance levels (teach_me / normal / technical / minimal) | beginner output is >2x the length of minimal |
| Jargon translated for beginners, kept for technical | "locked off" vs "completely still, resting on something solid" |
| Explanations fade once a concept has been taught | parentheticals drop from 3 to 1 across a sequence |
| Declared knowledge is never explained | `known_concepts` |
| Adaptive state persists across sessions | `explained` counters survive save/load |
| Shots requiring absent gear are REDESIGNED and disclosed | see below |

**Redesign, not advice.** A tracking shot is not a suggestion to someone filming
alone without a tripod — it is an instruction they cannot follow. Those shots
are rewritten (Spec §5) and the substitution is always stated:

> ADAPTED for your setup: no one else is there to operate a moving camera, so
> the camera stays put and YOU provide the movement; this reads differently, and
> is a real substitution

Silently swapping a tracking shot for a static one would leave someone believing
they filmed what was planned. The original spec is never mutated.

**The profile holds what the person told us.** It never infers skill from their
footage — concluding someone is a beginner because a clip was shaky is a
judgement they did not ask for. A test asserts the module reads no quality signal
at all.

Three defects found while proving it:

- The concept explanation was chosen from the ADAPTED wording, so once "locked
  off" became "rest it on something solid" the static shot was explained as a
  tracking shot — teaching the wrong thing. It now reads the original direction.
- `adapt_shot` rewrote "60fps" for the device, then `plainify` rewrote the
  replacement's own wording: "the highest frame-rate your phone offers (often the
  higher frame-rate setting)". One substitution per term, at one stage.
- `--skill` carried a default, which silently overrode the stored guidance level
  for anyone who never passed the flag — a saved `minimal` profile rendered full
  beginner output. The flag now defaults to None and overrides only when given.

### Phase 15 — asynchronous jobs (Spec §28)

`core/jobs.py` already existed with `Job`, `JobState` and `InlineJobQueue`, was
imported in one file, and `.submit()` was **never called anywhere** — the fifth
time a subsystem turned out to be written but unwired. Checking first saved
rewriting it.

Three things were genuinely missing, and each blocks a UI:

| added | why |
|---|---|
| `BackgroundJobQueue` on a thread pool | inline running blocks the caller for the length of a render |
| persistence on every state change | a job that vanishes with the process cannot be reported to a client that reconnects |
| cooperative cancellation | a render cannot be safely killed mid-ffmpeg |
| `pipeline/tasks.py` | ingest / edit / autopilot / master, all with one `progress=` signature |

Verified on real work, not toys: an ingest of 17 clips ran through the queue with
18 progress reports; a full edit was **cancelled at 70% during preview render**
and came back `cancelled` with `error=None`, because a user stopping a render did
nothing wrong.

Two honesty decisions worth naming:

- A job left `RUNNING` on disk is recovered as **failed with "the process ended
  while this job was running, so its outcome is unknown"**. No thread is carrying
  it any more, so showing it as live would be a lie, and guessing that it
  finished would be worse.
- Progress advances by pipeline **stage**, not by log volume. Deriving a
  percentage from how much text has been printed would be a fabricated number.

Distributed queueing is deliberately not built: it needs infrastructure this
environment does not have, and stubbing it would be the same failure as a fake
model provider. The `JobQueue` interface is what a Redis/RQ worker pool would
implement.

**Test suite: 173/173 passing** (`python3 tests/test_alledits.py`).
All five proof harnesses pass: `ab_brief_test`, `rescue_test`, `sound_test`,
`find_test`, `master_test`.

### Phase 4 — sound

The mix planner and the renderer's mix graph already existed, but nothing ever
called `plan_mix`, so every edit shipped the naive full-volume bed. Wiring it in
and then *measuring the result* exposed four defects.

| Capability | Evidence |
|---|---|
| Loudness normalised to a platform target | measured on the encoded file, below |
| True peak held under the ceiling after AAC | limiter sits below the delivery ceiling |
| Music ducked under speech, sidechained | measured -5.2 dB during real detected speech |
| Speech located by voice-band energy + syllable-rate modulation | windows match ground truth to <0.15 s |
| Invalid mix plans stop the render | `validate_mix` gates it exactly as the picture validator gates the timeline |

**Measured on the encoded deliverable (`scripts/sound_test.py`):**

| | source | delivered | target |
|---|---|---|---|
| loudness (social) | -17.5 LUFS | **-14.3** | -14.0 ±0.5 |
| loudness (broadcast) | -17.5 LUFS | **-23.1** | -23.0 ±0.5 |
| true peak | +0.6 dBFS | **-1.4** | ≤ -1.0 |

Four defects found by measurement, two of them in my own first attempts:

- **`loudnorm`'s dynamic mode landed 1.4–2.2 LU short.** I first concluded the
  target was physically unreachable under the peak ceiling. That was wrong:
  explicit gain-to-target plus a limiter reaches it. The normaliser now measures,
  applies gain, and verifies — correcting for what the limiter costs, which
  cannot be predicted from the input.
- **AAC overshoots the limiter by 1.2–2.0 dB**, so a limiter set at the delivery
  ceiling ships a file that violates it. The limiter now sits a measured margin
  below.
- **`DuckSpec.depth_db` never reached the filter.** It was reported to the user
  and validated, while the filter used a hardcoded ratio — announcing a 9 dB duck
  and applying roughly 18. My first fix derived the ratio analytically and was
  *also* wrong: sweeping ffmpeg showed threshold dominates almost entirely (ratio
  2→20 moved it under 1.5 dB) and attenuation saturates just under 7 dB. The
  parameters are now calibrated from measurement, and a request deeper than the
  saturation point is reported as capped rather than claimed.
- **Ducking was unreachable.** `plan_mix` was called with `voice_tracks=None`, so
  `has_voice` was always false. `audio/speech.py` now locates speech in the
  clips' own audio and feeds it in.

**On speech detection:** this is voice activity, not transcription. Transcription
is a deferred dependency and unavailable in this build; claiming it would be a
lie. Band energy alone would flag any midrange-heavy music, so a window must also
show syllable-rate (2–8 Hz) envelope modulation — which is why `music.wav` is
correctly *not* detected as speech.

**Test suite: 62/62 passing** (`python3 tests/test_alledits.py`).

---

## 2. STUBBED (interface exists, implementation deliberately absent)

These are *not* silently broken — they raise or report unavailability.

- **`AnthropicProvider`** — fully implemented against the Messages API, but
  `available()` returns `False` without `ANTHROPIC_API_KEY`. It was **never
  exercised in this environment** (no network egress). Untested against a live
  endpoint.
- **`EmbeddingProvider.embed_text` / semantic search** — raises
  `ProviderUnavailable` with an explanatory message. It does **not** fall back to
  keyword matching dressed up as semantic search.
- **Transcription** (`ShotRecord.transcript`) — field exists, always `None`.
- **Semantic tags** (`ShotRecord.semantic_tags`) — field exists, always empty.

## 3. DEFERRED (scoped, not started)

- Whisper-class transcription and word-level timing → FIND mode, subtitle work
- CLIP/SigLIP visual embeddings → open-vocabulary search ("someone realizing…")
- Segmentation/tracking (SAM-class) → object removal, masked grading, tracked text
- Super-resolution / frame interpolation / generative fill → RESCUE mode
- Voice isolation, stem separation, auto-ducking → SOUND mode
- Distributed job queue and GPU worker pool (interface in `core/jobs.py` is ready)
- Web UI (deliberately last, per the spec's instruction not to decorate before
  the engine works)
- AUTOPILOT and DIRECTOR modes (depend on the above)

## 4. ADDED (engineering decisions beyond the spec)

- **Decision ledger** (`core/ledger.py`) — every editorial choice with rationale,
  alternatives and measured evidence. Makes "explain your reasoning" and human
  override real rather than aspirational.
- **Flow coherence** as a first-class signal — distinguishes camera motion from
  subject motion. Without it, a locked-off shot of a busy scene misclassifies as
  handheld.
- **Relative jitter + directional consistency** for handheld detection — absolute
  jitter is confounded with pan speed.
- **Prominence gate** in shot detection — a cut is a *locally prominent* spike,
  not merely a high-change frame. Fixes over-segmentation of dynamic footage.
- **Authoritative timeline duration** — durations are owned by the plan and
  frame-quantized, not derived from source ranges. Derived durations accumulated
  rounding error and walked the edit off the beat (measured: 20% → 85% alignment).
- **Transition quota allocation** — picking the best transition per cut
  independently collapses to "hard cut everywhere". Quotas reproduce the
  reference's texture while still placing each transition where it fits.
- **Segment-level render cache** — makes the critique/revision loop affordable.

## 5. KNOWN LIMITATIONS (stated plainly)

- **The creative planner is rule-based, not a language model, in this build.**
  It is labelled `rule_based_planner` in every response and in the ledger. As of
  Phase 1 the brief *does* steer the edit, but through a lexicon rather than
  genuine language understanding: an unusual or metaphorical brief ("like a
  half-remembered dream of my grandmother's kitchen") will match few terms and
  fall back toward the reference style. The parser reports how many terms it
  matched, and confidence in the ledger drops when it matched none.
- **The noise estimator cannot distinguish sensor noise from extremely
  high-frequency CONTENT.** Nearest-neighbour-upscaled pixel art is uncorrelated
  at pixel scale by construction, and `15_fast_action` reads 1.000 for this
  reason. Spatial autocorrelation was tested as a discriminator and did not
  separate the two. Real camera footage does not have this property; denoise
  strength is capped so a false positive costs modest detail rather than
  destroying the image.
- **Overrides are slot-indexed.** If the plan changes shape (a different brief or
  duration produces a different number of slots), a directive pinned to "slot 7"
  refers to a different moment. Directives are not yet re-anchored to content.
- **Optical flow degrades on textureless footage.** Verified: a smooth gradient
  returns zero flow (the aperture problem). Camera classification on low-texture
  material is unreliable and should be reported as low confidence.
- **Camera classification is imperfect on procedurally animated content** where
  every pixel moves independently (4/6 on such material vs 6/6 on realistic
  footage). Real footage is closer to the latter case.
- **`radial_blur` is an approximation** (centre-weighted blur + vignette), not a
  true per-pixel rotational warp. Documented at the point of implementation.
- **Beat grid inherits ~1% tempo error**, which accumulates over long tracks;
  per-beat onset snapping limits, but does not eliminate, the drift.
- **Face detection is Haar-cascade** — frontal faces only, no tracking, no
  identity. Adequate for framing heuristics, not for subject continuity.
- **No colour management.** Everything is treated as BT.709. HDR/log footage
  will not be handled correctly.
- **Renderer is CPU x264.** No GPU path, no hardware decode.
- **Effects are applied per-clip in a fixed order** (geometry → motion → optics →
  colour → texture). There is no node graph or arbitrary compositing.

## 6. RULES OBSERVED

- No fake AI: the rule-based planner reports `is_llm=False` everywhere, and there
  is a test asserting it (`test_heuristic_provider_never_claims_to_be_an_llm`).
- No mock buttons / no fake completion: unavailable capabilities raise
  `ProviderUnavailable` rather than returning plausible output
  (`test_semantic_search_fails_loudly_rather_than_faking`).
- No "under one minute = copyright safe" claim appears anywhere in the codebase.
- No design around third-party watermark bypass.
- Style grammar stores measured characteristics only — no frames, no audio, no
  path back to the reference (`test_grammar_stores_no_reference_content`).
- Source media is copied in and never mutated; all work happens on proxies and
  in separate storage namespaces.
- Provenance fields (`resolution_provenance`, `fps_provenance`) exist on
  `ProjectSettings` so enhanced output is never mislabelled as native.
