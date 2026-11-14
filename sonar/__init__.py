"""sonar — camera-free laptop sonar (ultrasonic gestures, presence, breathing).

Plain words: the laptop speakers play a quiet signal just above human hearing,
the built-in microphones record the echoes, and this package turns those echoes
into three things: hand gestures, "is somebody there", and a breathing rate.

Module map (read DESIGN.md for the why, INTERFACES.md for the event schema):

    config.py     every signal / processing parameter in one dataclass
    tx.py         the transmitted waveform (2 pilot tones + 1 periodic chirp)
    dsp.py        filters, tone demodulation, Doppler STFT, chirp receiver
    frontend.py   streaming front end: raw mic audio -> per-frame Doppler + echo profile
    gestures.py   noise-floor normalisation, gesture segmenter, physics-only classifier
    presence.py   empty / entering / present state machine
    breathing.py  range-gated phase -> breathing waveform + rate
    sim.py        physics echo simulator (tests and offline development)
    audio.py      audio backends: Windows WASAPI exclusive, simulator, file replay
    selftest.py   1-minute hardware check (SNR per frequency)
    record.py     labelled data collection CLI
    dataset.py    recorded sessions -> training windows (+ augmentation)
    model.py      small CNN (PyTorch)
    train.py      training, metrics, ONNX export
    classifier.py ONNX runtime wrapper with the physics classifier as fallback
    engine.py     realtime engine (callback API + WebSocket JSON events)
    server.py     WebSocket broadcaster

Only numpy and scipy are needed to import the DSP; audio, torch and
onnxruntime are imported lazily by the modules that use them.
"""

from .config import SonarConfig, default_config, get_profile

__all__ = ["SonarConfig", "default_config", "get_profile"]
__version__ = "0.1.0"
