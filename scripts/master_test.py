"""Proof that MASTER produces conformant deliverables — and that QC can fail.

A QC report that always passes is a rubber stamp. This harness checks both
directions:

  1. Mastering to a profile produces a file that measures conformant.
  2. QC FAILS a file that does not conform, naming which fields.
  3. Upscaling is refused unless explicitly permitted, and disclosed when it is.
  4. The scaling decision matches what the encoder actually does — a false
     disclosure ("1.78x upscaled" for a letterboxed downscale) is its own defect.

    python3 scripts/master_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alledits.core.ffmpeg import ffmpeg  # noqa: E402
from alledits.master import (master, run_qc, get_profile, plan_scaling,  # noqa: E402
                             PROFILES)
from alledits.media.probe import probe  # noqa: E402

TMP = Path("/tmp/mastertest")
TMP.mkdir(exist_ok=True)


def make_source():
    """A 6s 1080x1920 clip with real audio, standing in for a finished edit."""
    src = TMP / "edit.mp4"
    if not src.exists():
        ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                "-t", "6", "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                "-shortest", str(src)])
    return src


def main():
    failures = []
    src = make_source()
    i = probe(src)
    print(f"source: {i.width}x{i.height} @ {i.fps:.2f} {i.codec} {i.duration:.2f}s\n")

    print("=" * 78)
    print("CONFORMANCE — each profile must produce a file that measures right")
    print("=" * 78)
    for name in ("youtube_shorts", "tiktok", "youtube_1080p", "broadcast_ebu"):
        out = TMP / f"m_{name}.mp4"
        r = master(src, out, name, allow_upscale=True)
        bad = [f"{c.name}({c.measured})" for c in r.qc.failed]
        print(f"\n  {name:16} {r.qc.summary()}   "
              f"{'PASS' if r.conformant else 'FAIL'}")
        if bad:
            print(f"    failed: {', '.join(bad)}")
            failures.append(f"{name}: {bad}")

    print("\n" + "=" * 78)
    print("QC MUST BE ABLE TO FAIL — vertical file against a broadcast profile")
    print("=" * 78)
    rep = run_qc(TMP / "m_youtube_shorts.mp4", get_profile("broadcast_ebu"))
    named = {c.name for c in rep.failed}
    print(f"\n  {rep.summary()}")
    for c in rep.failed:
        print(f"    {c.name}: expected {c.expected}, measured {c.measured}")
    expect_fail = {"resolution", "frame rate", "loudness"}
    if rep.passed:
        failures.append("QC passed a non-conformant file")
        print("  FAIL — QC is a rubber stamp")
    elif not expect_fail <= named:
        failures.append(f"QC missed failures: {expect_fail - named}")
        print(f"  FAIL — did not catch {expect_fail - named}")
    else:
        print("  PASS — caught resolution, frame rate and loudness")

    print("\n" + "=" * 78)
    print("UPSCALING — refused by default, disclosed when permitted")
    print("=" * 78)
    small = TMP / "small.mp4"
    if not small.exists():
        ffmpeg(["-i", str(src), "-vf", "scale=540:960", "-c:v", "libx264",
                "-crf", "20", "-c:a", "copy", "-t", "3", str(small)])
    si = probe(small)
    dec = plan_scaling(si.width, si.height, get_profile("youtube_shorts"))
    print(f"\n  {si.width}x{si.height} -> 1080x1920: upscaling={dec.upscaling} "
          f"factor={dec.factor:.2f}")
    if not dec.upscaling:
        failures.append("a genuine upscale was not detected")
        print("  FAIL — should be flagged as upscaling")

    try:
        master(small, TMP / "refused.mp4", "youtube_shorts", allow_upscale=False)
        failures.append("upscaling was not refused")
        print("  FAIL — should have refused")
    except ValueError:
        print("  refused without permission: PASS")

    r = master(small, TMP / "m_upscaled.mp4", "youtube_shorts", allow_upscale=True)
    disclosed = (r.qc.resolution_provenance == "upscaled"
                 and any("provenance" in c.name for c in r.qc.checks))
    print(f"  disclosed when permitted: {'PASS' if disclosed else 'FAIL'} "
          f"(provenance={r.qc.resolution_provenance})")
    if not disclosed:
        failures.append("upscaling was performed without disclosure")

    print("\n" + "=" * 78)
    print("NO FALSE DISCLOSURE — a letterboxed downscale is not an upscale")
    print("=" * 78)
    dec2 = plan_scaling(1080, 1920, get_profile("broadcast_ebu"))
    print(f"\n  1080x1920 -> 1920x1080: upscaling={dec2.upscaling} "
          f"factor={dec2.factor:.2f}  ({dec2.provenance})")
    if dec2.upscaling:
        failures.append("claimed an upscale for a letterboxed downscale")
        print("  FAIL — crying wolf")
    else:
        print("  PASS")

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print("  " + f)
        return 1
    print("All MASTER checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
