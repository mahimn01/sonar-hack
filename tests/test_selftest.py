"""The hardware self-test, run against a stand-in laptop (delay, gain, weak highs, noise)."""

from sonar.selftest import SimProbe, find_offset, run_selftest, tone_program


def test_selftest_reads_back_the_simulated_laptop(cfg):
    report = run_selftest(SimProbe(cfg), cfg, "default", verbose=False)
    # the stand-in has -79 dBFS / 60 dB SNR up to 18.5 kHz and 10 dB less above (as measured on the real laptop)
    assert abs(report["tones"]["18000"]["ch0"]["level_dbfs"][0] + 79.0) < 1.0
    assert abs(report["tones"]["18000"]["ch0"]["snr_1hz_db"] - 60.0) < 3.0
    assert abs(report["tones"]["20000"]["ch1"]["snr_1hz_db"] - 50.0) < 3.0
    assert report["sonar"]["locked"] and abs(report["sonar"]["drift_ppm"]) < 1.0
    assert report["checks"]["lock"] == "PASS" and report["checks"]["clipping"] == "PASS"
    assert report["checks"]["speaker_separation"] == "PASS" and report["speaker_separation_us"] > 150
    assert report["recommended_profile"] == "default"
    assert report["profiles"]["strong"]["tone_snr_db"] > report["profiles"]["default"]["tone_snr_db"] + 5


def test_selftest_notices_when_both_channels_share_one_speaker(cfg):
    report = run_selftest(SimProbe(cfg, separate_speakers=False), cfg, "default", verbose=False)
    assert report["speaker_separation_us"] < 30 and report["checks"]["speaker_separation"] == "WARN"


def test_selftest_flags_a_weak_signal(cfg):
    report = run_selftest(SimProbe(cfg, cn0_db=38.0), cfg, "default", verbose=False)
    assert report["checks"]["chirp_snr"] in ("WARN", "FAIL") and report["checks"]["tone_snr"] in ("WARN", "FAIL")


def test_marker_is_found(cfg):
    prog, spans = tone_program(cfg, 0)
    rx = SimProbe(cfg).run(prog)
    assert abs(find_offset(cfg, rx) - 3301 - 33) < 12        # latency of the stand-in + speaker-to-mic delay
    assert len(spans) == 13
