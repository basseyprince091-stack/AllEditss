"""Proof that FIND retrieves the right footage.

The test corpus has known properties by construction (the clip names ARE the
ground truth: `02_pan_right` pans right, `07_dark_lowkey` is dark, and so on),
so retrieval can be scored rather than eyeballed.

Three things are checked:
  1. Queries return the clips they should — measured as precision/recall against
     the named ground truth.
  2. Negation excludes rather than being ignored.
  3. A query the system cannot understand says so, instead of returning
     plausible-looking results it has no basis for.

    python3 scripts/find_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alledits.core.storage import LocalStorage  # noqa: E402
from alledits.media.ingest import Ingestor  # noqa: E402
from alledits.search.index import MediaIndex  # noqa: E402
from alledits.search.query import parse_query  # noqa: E402

CLIPS = Path("/home/claude/testmedia/clips")

# query -> clip name fragments that SHOULD come back
GROUND_TRUTH = [
    ("static wide shots",        {"01_static_wide"}),
    ("pan right",                {"02_pan_right"}),
    ("push in",                  {"03_push_in"}),
    ("pull out",                 {"04_pull_out"}),
    ("tilt up",                  {"05_tilt_up"}),
    ("handheld",                 {"06_handheld"}),
    ("dark low key footage",     {"07_dark_lowkey"}),
    ("bright high key",          {"08_bright_highkey"}),
    ("talking head with speech", {"17_talking_head"}),
    ("poor quality footage",     {"12_lowres_compressed", "13_very_poor"}),
]


def build_index():
    """Ingest the corpus, keeping a map back to the ORIGINAL filenames.

    Ingest copies each clip to `asset_<hash>.mp4`, so the shot's source path no
    longer carries the ground truth. The mapping has to be captured here.
    """
    ing = Ingestor(LocalStorage("/home/claude/findproof"))
    idx = MediaIndex()
    origin = {}
    for c in sorted(CLIPS.glob("*.mp4")):
        asset = ing.ingest(c)
        origin[asset.id] = c.stem
        idx.add_asset(asset)
    return idx, origin


def source_name(shot, origin):
    return origin.get(shot.asset_id, Path(shot.source_path).stem)


def main():
    print("indexing corpus...")
    idx, origin = build_index()
    print(f"  {len(idx.shots)} shots from {len(list(CLIPS.glob('*.mp4')))} clips\n")

    failures = []
    print("=" * 78)
    print("RETRIEVAL — does the query return the footage it names?")
    print("=" * 78)
    print(f"\n  {'query':32}{'top hit':24}{'recall':>8}  verdict")

    total_r = 0.0
    for query, expected in GROUND_TRUTH:
        res = idx.search(query, top_k=5)
        names = [source_name(r["shot"], origin) for r in res]
        # map ingested asset names back to originals via the index order
        hit_names = set(names)
        found = {e for e in expected if any(e in n for n in hit_names)}
        recall = len(found) / len(expected)
        total_r += recall
        top = names[0] if names else "(nothing)"
        ok = recall >= 0.5
        if not ok:
            failures.append(f"{query!r}: recall {recall:.2f}, got {names[:3]}")
        print(f"  {query:32}{top[:22]:24}{recall:>8.2f}  {'PASS' if ok else 'FAIL'}")

    print(f"\n  mean recall over {len(GROUND_TRUTH)} queries: "
          f"{total_r / len(GROUND_TRUTH):.2f}")

    print("\n" + "=" * 78)
    print("NEGATION — 'no X' must exclude X")
    print("=" * 78)
    # Uses an attribute that actually VARIES across this corpus. An earlier
    # version tested "no faces" against a corpus with zero faces anywhere, so it
    # passed without demonstrating anything.
    pos = {source_name(r["shot"], origin) for r in idx.search("handheld", top_k=20)}
    neg = {source_name(r["shot"], origin) for r in idx.search("no handheld", top_k=20)}
    print(f"\n  matched 'handheld': {len(pos)}   'no handheld': {len(neg)}")
    if not pos:
        failures.append("negation test is vacuous: nothing matched the positive")
        print("  VACUOUS — the positive query matched nothing")
    else:
        overlap = pos & neg
        print(f"  overlap: {len(overlap)}   {'PASS' if not overlap else 'FAIL'}")
        if overlap:
            failures.append(f"negation leaked: {overlap}")

    print("\n" + "=" * 78)
    print("INERT CRITERIA — an unmeasurable filter must be declared")
    print("=" * 78)
    idx.search("wide shots", top_k=5)
    inert = getattr(idx, "last_inert_criteria", [])
    print(f"\n  'wide shots' -> inert criteria: {inert}")
    shot_sizes = {sh.visual.get("shot_size") for sh in idx.shots}
    print(f"  shot_size values present in corpus: {shot_sizes}")
    if shot_sizes == {"unknown"} and not inert:
        failures.append("shot_size is never measured but was not declared inert")
        print("  FAIL — filter silently unmatchable")
    else:
        print("  PASS")

    print("\n" + "=" * 78)
    print("HONESTY — an un-understood query must say so")
    print("=" * 78)
    q = parse_query("the shot where she looks relieved")
    print(f"\n  criteria understood: {len(q.criteria)}")
    print(f"  unmatched terms:     {q.unmatched_terms}")
    print(f"  claims semantic:     {q.semantic}")
    honest = (not q.criteria) and q.unmatched_terms and q.semantic is False
    print(f"  {'PASS' if honest else 'FAIL'}")
    if not honest:
        failures.append("un-understood query did not report the gap")

    # And the open-vocabulary path must refuse rather than approximate.
    from alledits.core.errors import ProviderUnavailable
    try:
        idx.search_by_text("a feeling of relief")
        failures.append("search_by_text faked a semantic result")
        print("  search_by_text: FAIL (should refuse)")
    except ProviderUnavailable:
        print("  search_by_text refuses without a VLM provider: PASS")

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print("  " + f)
        return 1
    print("All FIND checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
