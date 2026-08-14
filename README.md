# ALLEDITS

An AI-native editorial system. Give it your footage, a reference edit whose style
you want, and a music track — it produces a genuinely edited video and can
explain every decision it made.

**This is a working vertical slice, not a mockup.** It ingests real media, does
real analysis, and renders a real file.

## Quick start

```bash
# generate a test corpus with known ground truth
python3 scripts/make_test_media.py /tmp/testmedia

# run the full pipeline
python3 -m alledits.cli edit \
    --clips /tmp/testmedia/clips \
    --reference /tmp/testmedia/reference.mp4 \
    --music /tmp/testmedia/music.wav \
    --instruction "cinematic, high-energy, build to the drop" \
    --duration 18 --size 1080x1920 --out ./out.mp4

# ask it why it did what it did
python3 -m alledits.cli explain --run ./alledits_work --stage clip_selection

# tests
python3 tests/test_alledits.py
```

## What it actually does

1. **Ingests** each clip: proxy, shot detection, per-shot visual analysis
   (camera movement via optical flow, composition, colour, faces), dual quality
   scoring, feature embedding — all cached by content hash.
2. **Analyzes the music**: tempo, beat grid, downbeats, energy curve, sections,
   drops.
3. **Analyzes the reference** and extracts a *StyleGrammar* — pacing, rhythm,
   intensity arc, transition mix, grading direction. Characteristics only; no
   frames or audio from the reference are retained.
4. **Plans** a beat-locked slot structure that maps the reference's narrative arc
   onto your target duration.
5. **Selects** a clip for each slot by scoring energy fit, motion fit, length,
   technical quality, creative value and — critically — **continuity with the
   preceding shot**.
6. **Builds** a Timeline (typed, versioned, human-readable) and **validates** it.
   Nothing renders without passing.
7. **Renders** a preview, **critiques the rendered file**, revises, and renders
   final.

## The two scores

Technical quality and creative value are tracked **separately**. A grainy phone
clip can be the best shot you have — it gets used as a flash frame or accent
rather than discarded. Conversely, footage that is already good is left alone:
processing it would only degrade it.

## Explainability

Every decision is recorded with its rationale, the alternatives considered, and
the measured evidence behind it:

```
[clip_selection/slot_01] -> asset_a33cdb4a_s000 (confidence 0.78)
    Chose ... for the hook slot (2.32-3.25s). energy 1.00, motion 0.60,
    length 1.00, technical 0.56, creative 0.68, continuity 0.70 (movement
    matches (pan_right, 0° apart) — the cut will feel like one continuous
    gesture). Close call against asset_fdb18bf8_s000 — preferred because its
    movement and composition match the preceding shot (0.70 vs 0.62).
```

## Honesty

See `IMPLEMENTATION_LEDGER.md` for the full account of what is implemented,
stubbed, deferred and limited. Two rules the code enforces with tests:

- The rule-based planner **never** claims to be a language model.
- Capabilities that aren't installed (semantic text search, transcription)
  **raise** rather than returning plausible-looking output.

## Requirements

Python 3.11+, ffmpeg 6+, numpy, scipy, opencv-python, scikit-learn/image.
Optional: `ANTHROPIC_API_KEY` to enable the reasoning-tier model provider.

## Directing the edit

The brief is not decoration — it sets measurable constraints:

```bash
alledits edit --instruction "cinematic, slow and restrained, no shake" ...
alledits edit --instruction "chaotic, aggressive and fast, whip pans" ...
```

Those two produce 12 clips at 1.50s average versus 39 clips at 0.46s, with zero
shake effects versus 18. Run `python3 scripts/ab_brief_test.py` to reproduce.

When you disagree with a decision, override it. Directives persist with the
project and survive re-planning and the critique loop:

```bash
alledits project   --workdir ./work                 # see the edit + directives
alledits direct    --workdir ./work --kind pin_shot --slot 3 --shot <shot_id> \
                   --note "I want this shot here"
alledits direct    --workdir ./work --kind ban_effect --value shake
alledits direct    --workdir ./work --kind lock_clip --slot 5   # critic can't touch it
alledits undirect  --workdir ./work --id <directive_id>
alledits edit      ... --workdir ./work             # rebuilds honouring directives
```

A lock outranks the critic: the revision loop will not undo a cut you chose to keep.


## Sound

Every edit is normalised to a platform loudness target and held under a true-peak
ceiling, measured on the encoded file rather than assumed:

| | source | delivered |
|---|---|---|
| loudness | -17.5 LUFS | -14.3 (target -14.0) |
| true peak | +0.6 dBFS | -1.4 (ceiling -1.0) |

When a clip carries speech in its own audio, the music ducks under it
automatically — detected by voice-band energy plus syllable-rate modulation, so a
midrange-heavy music bed is not mistaken for talking. Reproduce with
`python3 scripts/sound_test.py`.

## Sound

The mix is decided, validated, then measured — never assumed:

- **Loudness** is normalised to a platform target (-14 LUFS social, -23 broadcast)
  by measuring the mix, applying gain, then re-measuring to confirm.
- **True peak** is held below the ceiling *after* AAC encoding, which
  reconstructs peaks above what the limiter was given.
- **Music ducks under speech** found in the clips' own audio. If a clip's chosen
  range lands in a silent gap, no duck is applied — the file containing speech
  elsewhere is not a reason to move the music.

```bash
python3 scripts/sound_test.py     # measures loudness, true peak and duck depth
```

Where the sidechain cannot reach a requested duck depth, the plan reports the
depth it can actually deliver rather than the one requested.

When a clip carries dialogue, the editor cuts *onto* the speech rather than the
middle of the shot, so the talking is actually used and the music has something
to move under.

## FIND — searching your footage

```bash
alledits find --clips ./footage --query "dark static shots, no handheld"
```

FIND searches by what was actually **measured** — camera movement, exposure,
colour, motion, technical quality, faces, speech — and every result says why it
matched:

```
1. 07_dark_lowkey.mp4  [0.00-3.00s]  score 1.00
     matched: not camera movement is handheld; camera movement is static;
              brightness below 0.35
```

It is not semantic search. A query like *"the shot where she looks relieved"*
returns no criteria and lists the words it could not interpret, rather than
guessing. Where a filter refers to something this library never measured, FIND
says the filter is inert instead of quietly returning nothing.

```bash
python3 scripts/find_test.py      # scores retrieval against known ground truth
```

## MASTER — delivering

```bash
alledits profiles                                   # list delivery targets
alledits master --input edit.mp4 --output out.mp4 --profile tiktok
alledits edit ... --deliver youtube_shorts          # edit and deliver in one pass
```

Every master is QC'd by re-measuring the encoded file — resolution, frame rate,
codec, pixel format, audio, duration, loudness, true peak, faststart. A check
that cannot be measured reports `SKIP`, never `PASS`.

ALLEDITS will not silently invent pixels. Mastering a 540x960 source to a
1080x1920 profile is refused unless you pass `--allow-upscale`, and when you do,
the report records `resolution provenance: upscaled` rather than presenting
interpolated detail as captured.

## AUTOPILOT — no brief required

```bash
alledits autopilot --clips ./footage --reference ref.mp4 --music track.wav \
  --duration 12 --deliver youtube_shorts
```

Explores several genuinely different treatments, renders each as a preview,
critiques the actual rendered result, and delivers the best one.

Every candidate's score is reported, not just the winner's — and when the top
treatments score alike, ALLEDITS says the choice is weakly held rather than
dressing up an arbitrary pick as a confident one.

## STYLE — learn it once, reuse it anywhere

```bash
alledits style save  --name punchy --reference some_edit.mp4
alledits style save  --name calm   --reference another_edit.mp4
alledits style blend --name house  --sources punchy:1,calm:3
alledits style list

alledits edit --clips ./footage --music track.wav --style house    # no reference needed
```

A saved style is a **measurement** — pacing, motion, colour, transition mix,
intensity shape. It contains no frames, no audio, and no path back to the work it
was learned from, so reusing a style never redistributes someone else's footage.

Blending averages what can meaningfully be averaged. It does not average
categorical qualities — there is no midpoint between "accelerating" and
"steady" — so the heaviest-weighted wins and the disagreement is recorded:

```
note: blended from 2 references: punchy 25%, calm 75%
note: sources disagreed on rhythm (accelerating, steady); took the highest-weighted, steady
```

## DIRECTOR — notes in plain language

```bash
alledits note --workdir ./work --dry-run --text "hold the second shot longer, no flashes"
alledits note --workdir ./work --text "lose the shaky one"
alledits edit ... --workdir ./work        # rebuild with the notes applied
```

References resolve by position ("the third shot"), by measured attribute ("the
shaky one", "the dark one"), or globally ("punch it up"). Notes become
directives, so they are reversible and survive a re-plan.

Anything DIRECTOR cannot act on is reported rather than quietly dropped:

```
could NOT act on:
  - 'make it feel like a half-remembered dream'
```

Holding a shot longer pushes everything after it, which takes those cuts off the
musical grid — so ALLEDITS drops their beat-lock and warns that the total
duration changed, rather than claiming a sync it no longer has.

## SHOOT — what to film next

```bash
alledits shoot sequences
alledits shoot plan     --sequence skill --style house --skill beginner
alledits shoot coverage --sequence skill --style house --clips ./footage
```

Shot lengths follow the style: the same shot asks for a 4-second take under a
fast reference and 7 under a slow one.

Coverage compares the plan against footage you already have. It never says a
shot is *covered* — only that the measurable shape matches and the content still
needs your eye:

```
LIKELY        4. The action
    the measurable shape matches; whether it actually shows "the skill or
    action being performed" needs a human eye
MISSING       2. Detail
    no clip satisfies 2 measured requirement(s); closest match met 50% of them
```

Missing shots come back with full direction, including what *not* to do.

## PROFILE — how you want to be told

```bash
alledits profile set --level teach_me --device phone --no-tripod --crew 1
alledits profile set --level minimal --device camera --tripod --crew 3
alledits profile show
```

Guidance adapts to the level: a beginner gets plain language, the reason for each
shot, and short explanations that **stop appearing** once a concept has been
taught. A professional gets four lines.

Shots you cannot physically film are redesigned rather than demanded — and the
substitution is always stated, so you never think you filmed the original:

```
ADAPTED for your setup: no one else is there to operate a moving camera, so the
camera stays put and YOU provide the movement; this reads differently, and is a
real substitution
```
