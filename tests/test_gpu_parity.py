"""The GPU clip maker (PyTorch) must make what the CPU reference (numpy) makes.

These tests run the GPU code on the CPU (and on Apple MPS when it is safe to), on a small
sample of the fixture that scripts/check_gpu_parity.py uses in full on the 4090.
"""

import os
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from datagen import generate as g  # noqa: E402
from datagen import parity  # noqa: E402
from sonar.features import stream_features, stream_from_columns  # noqa: E402
from sonar.features_torch import TorchFrontend, frames_for  # noqa: E402
from sonar.frontend import run_frontend  # noqa: E402


def _job(seed: int, kind: str, **over) -> tuple:
    base = dict(profile=None, min_snr=6.0, keep_audio=False, backgrounds=(), bg_share=0.0, swap_mics=False,
                swap_speakers=False, mics=None, tweak=None)
    base.update(over)
    return (seed, kind, *base.values())


@pytest.fixture(scope="module")
def clips():
    """Three CPU reference clips with everything captured: a swipe, a push with clock mismatch, a fidget."""
    out = []
    for job in (_job(9000003, "swipe_left"), _job(9000210, "push", tweak={"hw": {"clock_ppm": 30.0, "hum_hz": 60.0}}),
                _job(9000407, "none")):
        cap: dict = {}
        g.make_clip(job, capture=cap)
        out.append((job, cap))
    return out


def test_torch_frontend_equals_engine_frontend(cfg, clips):
    """Doppler columns from features_torch == the columns dsp.Frontend emits, for 2 microphones and for 1."""
    for mics, fir in ((2, "conv"), (1, "conv"), (2, "fft"), (1, "fft")):     # fft: the long filter through FFTs (--gen-features fft)
        fe = TorchFrontend(cfg, mics, "cpu", fir=fir)
        assert fe.fir == fir
        x = np.stack([cap["rx"][:, :mics] for _, cap in clips])
        db, lat = fe.columns(torch.from_numpy(x))
        assert db.shape == (len(clips), frames_for(fe.cfg, x.shape[1]), cfg.n_tones * mics, cfg.doppler_bins)
        for i in range(len(clips)):
            frames = run_frontend(fe.cfg, x[i])
            ref_db = np.stack([f.doppler_db for f in frames])
            np.testing.assert_allclose(db[i].numpy(), ref_db, atol=2e-5)          # dB values stored as float32
            if mics == 2:
                ref_lat = np.stack([f.lateral for f in frames])
                np.testing.assert_allclose(lat[i].numpy(), ref_lat, atol=1e-9 * np.abs(ref_lat).max() + 1e-30)
            else:
                assert lat is None
            # ... and the engine's own burst code then finds the same bursts and the same pictures
            ref = stream_features(x[i], cfg)
            got = stream_from_columns(fe.cfg, db[i].numpy(), None if lat is None else lat[i].numpy())
            assert [(s.start, s.end, s.decided) for s in got.segments] == [(s.start, s.end, s.decided) for s in ref.segments]
            assert got.not_before == ref.not_before and np.array_equal(got.ready, ref.ready)
            np.testing.assert_allclose(got.feat, ref.feat, atol=1e-5)


def test_torch_renderer_equals_numpy_renderer(clips):
    """Audio before rounding agrees to parts in a million; after rounding at most one converter step apart."""
    from datagen.gpu_pipeline import GpuRenderer, plan_clip

    gpu = GpuRenderer(None, "cpu")
    assert gpu.precision == "ref"
    plans = [plan_clip((job, 0, None, True))["plan"] for job, _ in clips]
    out = gpu.render(plans, want_parts=True)
    for i, (_, cap) in enumerate(clips):
        if cap["attempts"] != 1:                                  # the capture holds the last attempt; plans the first
            continue
        dyn_ref, sta_ref = cap["parts"]["dynamic"], cap["parts"]["static"]
        dyn, sta = out["dynamic"][i].numpy(), out["static"][i].numpy()
        assert np.sqrt(np.mean((dyn - dyn_ref) ** 2)) < 2e-5 * np.sqrt(np.mean(dyn_ref ** 2))
        assert np.sqrt(np.mean((sta - sta_ref) ** 2)) < 1e-9 * np.sqrt(np.mean(sta_ref ** 2))
        # right before the converter: parts in a million (float32 echo noise, amplified by the virtual microphone)
        conv_ref = cap["parts"]["converter_in"]
        d = np.abs(out["converter_in"][i].numpy() - conv_ref)
        assert np.sqrt(np.mean(d ** 2)) < parity.LIMITS["ref"]["conv_rel"] * np.sqrt(np.mean(conv_ref ** 2))
        # after rounding: never more than one step (plus that difference), and only as many flipped samples as it predicts
        rx = out["rx"][i].numpy()
        step = 2.0 ** -(cap["hw"].adc_bits - 1)
        assert np.abs(rx - cap["rx"]).max() <= step * 1.0001 + d.max()
        assert np.mean(rx != cap["rx"]) <= 1.5 * d.mean() / step + 2e-5
        assert abs(float(out["snr_db"][i]) - cap["snr"]) < 1e-3


def test_render_does_not_depend_on_the_batch(clips):
    """A clip is the same whether it is rendered alone or next to other clips (bit for bit on the CPU)."""
    from datagen.gpu_pipeline import GpuRenderer, plan_clip

    gpu = GpuRenderer(None, "cpu")
    plans = [plan_clip((job, 0, None, True))["plan"] for job, _ in clips]
    together = gpu.render(plans)["rx"].numpy()
    for i, plan in enumerate(plans):
        alone = gpu.render([plan])["rx"].numpy()[0]
        assert np.array_equal(alone, together[i])


def test_fused_kernel_code_equals_torch_echo_loop(clips):
    """datagen/echo_paths.c (the CUDA kernel's source, compiled here as plain C) == the PyTorch echo loop."""
    import shutil

    if shutil.which(os.environ.get("CC", "cc")) is None:
        pytest.skip("no C compiler here")
    from datagen import echo_kernel
    from datagen.gpu_pipeline import GpuRenderer, plan_clip

    plans = [plan_clip((job, 0, None, True))["plan"] for job, _ in clips]
    ref = GpuRenderer(None, "cpu", kernel="off").render(plans, want_parts=True)
    got = GpuRenderer(None, "cpu", kernel="host").render(plans, want_parts=True)
    a, b = got["dynamic"].numpy(), ref["dynamic"].numpy()
    assert np.sqrt(np.mean((a - b) ** 2)) < 2e-6 * np.sqrt(np.mean(b ** 2))        # float32 rounding of the same sum
    np.testing.assert_allclose(got["snr_db"].numpy(), ref["snr_db"].numpy(), atol=1e-4, equal_nan=True)
    assert echo_kernel.usable(torch.device("cpu"))[0] is False                    # never claimed usable without CUDA


def test_plan_makes_the_same_random_draws_as_render(clips):
    """After planning, a clip's random generator is where render() would have left it."""
    from datagen.hardware import Profile
    from datagen.plan import make_plan
    from datagen.render import render

    core, (tx, pre, n) = g._setup(None)
    job = clips[0][0]
    j = g.unpack_job(job)
    states = []
    for use_plan in (False, True):
        rng = np.random.default_rng(j["seed"])
        scene, hw, _, _ = g.draw_world(rng, core, j, Profile(band_hz=core.band), False, n)
        if use_plan:
            make_plan(tx, hw, scene, n, rng, pre, core.band)
        else:
            render(tx, hw, scene, n, rng, pre=pre, band=core.band)
        states.append(rng.bit_generator.state)
    assert states[0] == states[1]


def test_noise_shaped_on_the_device_is_the_noise_of_render(clips):
    """--gen-noise gpu: the workers only draw the hall noise and the gain wander, the renderer shapes them.
    The draws use up the random generator exactly like render(), and the finished audio is the same."""
    from datagen.gpu_pipeline import GpuRenderer, plan_clip
    from datagen.hardware import Profile, sample_hardware
    from datagen.plan import _gain_split, _noise_split, make_plan
    from datagen.render import _gain_track, _noise, render

    # 1. the generator ends where render() leaves it, for a whole clip ...
    core, (tx, pre, n) = g._setup(None)
    j = g.unpack_job(clips[0][0])
    states = []
    for split in (None, True):
        rng = np.random.default_rng(j["seed"])
        scene, hw, _, _ = g.draw_world(rng, core, j, Profile(band_hz=core.band), False, n)
        if split:
            plan = make_plan(tx, hw, scene, n, rng, pre, core.band, shape_on_device=True)
            assert plan.shaped is False and plan.gain is None and plan.babble is not None
        else:
            render(tx, hw, scene, n, rng, pre=pre, band=core.band)
        states.append(rng.bit_generator.state)
    assert states[0] == states[1]
    # ... and for laptops with every kind of extra: interfering tones, clicks, a gain step, no babble, no wander
    for k, tweak in enumerate(({}, {"tones": [(19200.0, -15.0, 2.0)]}, {"gain_step_db": 6.0, "hf_burst_rate": 3.0},
                               {"babble_dbfs": -200.0}, {"gain_drift_db": 0.0})):
        hw = sample_hardware(np.random.default_rng(40 + k), Profile(), 48000)
        for name, value in tweak.items():
            setattr(hw, name, value)
        r1, r2 = np.random.default_rng(7 + k), np.random.default_rng(7 + k)
        _noise(hw, 9600, r1)
        _gain_track(hw, 9600, r1)
        _noise_split(hw, 9600, r2)
        _gain_split(hw, 9600, r2)
        assert r1.bit_generator.state == r2.bit_generator.state, tweak

    # 2. the same clips with the shaping in the workers and on the device
    jobs = [job for job, _ in clips] + [_job(9000611, "pull", tweak={"hw": {"gain_step_db": -6.0, "tones": [(19200.0, -15.0, 2.0)]}}),
                                        _job(9000612, "none", tweak={"hw": {"babble_dbfs": -200.0, "gain_drift_db": 0.0}})]
    ref = GpuRenderer(None, "cpu").render([plan_clip((job, 0, None, True))["plan"] for job in jobs], want_parts=True)
    gpu = GpuRenderer(None, "cpu", noise="gpu")
    assert gpu.shape_on_device
    got = gpu.render([plan_clip((job, 0, None, True, None, True))["plan"] for job in jobs], want_parts=True)
    a, b = got["converter_in"].numpy(), ref["converter_in"].numpy()
    assert np.abs(a - b).max() < 1e-12 * np.abs(b).max()            # float64 arithmetic in another order, nothing more
    assert np.mean(got["rx"].numpy() != ref["rx"].numpy()) < 1e-5   # a sample exactly on a rounding edge may fall the other way
    np.testing.assert_allclose(got["snr_db"].numpy(), ref["snr_db"].numpy(), atol=1e-6, equal_nan=True)


def test_staged_stack_equals_plain_stack():
    """The page-locked route of big batches to a CUDA device (run here with an ordinary buffer): same numbers, and
    a second, smaller batch through the reused buffer does not disturb the first batch's tensor."""
    from datagen import render_torch as rt

    rng = np.random.default_rng(0)
    first = [rng.standard_normal((2, 500)) for _ in range(5)]
    second = [rng.standard_normal((2, 500)) for _ in range(3)]
    rt._PINNED.clear()
    rt._PINNED["ok"] = True
    a = rt._stack_staged(first, torch.device("cpu"), torch.float64, pin=False)
    b = rt._stack_staged(second, torch.device("cpu"), torch.float32, pin=False)
    assert np.array_equal(a.numpy(), np.stack(first))
    assert np.array_equal(b.numpy(), np.stack(second).astype(np.float32))
    assert a.shape == (5, 2, 500) and b.shape == (3, 2, 500) and len(rt._PINNED) == 2      # one buffer, reused
    rt._PINNED.clear()
    rt._PINNED["ok"] = True


def test_stored_reference_matches_the_fixture():
    """tests/fixtures/gpu_parity_reference.npz belongs to today's fixture (regenerate it when the fixture changes)."""
    assert parity.REFERENCE_FILE.exists(), "run: python scripts/check_gpu_parity.py --device torch-cpu --write-reference"
    with np.load(parity.REFERENCE_FILE, allow_pickle=False) as z:
        stored = {k: z[k] for k in z.files}
    jobs = parity.fixture_jobs(1000, "background.npy")
    assert len(jobs) == 1000 and int(stored["version"]) == parity.FIXTURE_VERSION
    assert [parity.clip_key(name, job) for name, job in jobs] == stored["seeds"].tolist()
    assert len(str(stored["sha256"])) == 64 and stored["thumb"].shape[0] == 1000
    kinds = {job[1] for _, job in jobs}
    assert {"swipe_left", "swipe_right", "push", "pull", "none", "wave", "entering", "leaving", "present", "empty"} <= kinds
    names = " | ".join(name for name, _ in jobs)
    for needle in ("on a recording", "one microphone", "swapped", "clipping", "ppm", "fastest"):
        assert needle in names


def test_two_clip_types_do_not_stall_the_generator(tmp_path):
    """A run that mixes clips laid over a real recording with pure simulation must finish.

    The two types cannot share a GPU batch. Here 3 + 3 of them get planned, a batch holds 4 and 6 plans are
    made ahead: neither type fills a batch and no more plans are made. The scheduler used to spin at that
    point for ever (round 3 on the 4090: the "bg" sets, which lay 70 % of their clips over real recordings).
    """
    from datagen.gpu_pipeline import generate_gpu

    background = parity.build_background(tmp_path)
    kinds = ("none", "present", "swipe_left", "none", "wave", "none", "present", "none", "none", "present")
    jobs = [_job(9400000 + i, kind, backgrounds=(background,), bg_share=1.0 if i % 2 == 0 else 0.0)
            for i, kind in enumerate(kinds)]
    stats: dict = {}
    results = list(generate_gpu(jobs, None, "cpu", workers=1, batch=4, stats=stats))
    assert [r["meta"]["seed"] for r in results] == [job[0] for job in jobs]                 # every clip, in job order
    assert [r["meta"]["real_background"] for r in results] == [i % 2 == 0 for i in range(len(jobs))]
    assert stats["part_batches"] >= 1          # the hold-up was met, and a part-filled batch got things moving again


@pytest.mark.slow
def test_gpu_pipeline_matches_cpu_reference(tmp_path):
    """The whole check on a small sample of the fixture: CPU reference against the GPU code run on the CPU."""
    report = parity.run(24, "cpu", workers=2, batch=6, work=tmp_path)
    parity.print_report(report)
    assert report["passed"], report["failures"]
    assert report["precision"] == "ref" and report["pictures_compared"] >= 20


def _mps_is_safe() -> tuple[bool, str]:
    """Apple MPS only when asked for or when plenty of memory is free (another job on this Mac needs the rest)."""
    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return False, "no Apple MPS here"
    if os.environ.get("SONAR_TEST_MPS") == "1":
        return True, ""
    if os.environ.get("SONAR_TEST_MPS") == "0":
        return False, "SONAR_TEST_MPS=0"
    try:
        text = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        free = int(text.rsplit("free percentage:", 1)[1].strip().rstrip("%"))
    except Exception:
        return False, "could not read the free memory"
    return free >= 40, f"only {free} % of memory free (SONAR_TEST_MPS=1 forces the test)"


@pytest.mark.slow
def test_gpu_pipeline_on_mps(tmp_path):
    """The same check on Apple's GPU. MPS has no float64, so the limits are the looser 'single' ones."""
    ok, why = _mps_is_safe()
    if not ok:
        pytest.skip(why)
    report = parity.run(12, "mps", workers=2, batch=4, work=tmp_path)
    parity.print_report(report)
    assert report["precision"] == "single"
    assert report["passed"], report["failures"]
