"""Room adaptation: the BatchNorm statistics really become the room's, the adapted model exports and runs in the
engine, and on simulated "other rooms" the offline replay shows the gain the calibration is for."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sonar import adapt                                                    # noqa: E402
from sonar.adapt import adapt_batchnorm, adapted_module, balanced_weights, evaluate, make_prototypes, split_calibration   # noqa: E402
from sonar.audio import FileAudio                                          # noqa: E402
from sonar.classifier import OnnxClassifier, load_classifier               # noqa: E402
from sonar.dataset import build_dataset, center_crop, find_sessions        # noqa: E402
from sonar.engine import Engine                                            # noqa: E402
from sonar.model import GestureNet                                         # noqa: E402

from .engine_helpers import room_session, small_model                      # noqa: E402

OTHER = dict(hand_db=-26.0, cn0_db=66.0, mic_gap=0.14, spk_gap=0.30, offset=(0.0, 0.08, 0.05), speed=(0.6, 1.0))


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return small_model(tmp_path_factory) / "gesture.onnx"


@pytest.fixture(scope="module")
def rooms(cfg, tmp_path_factory):
    """Two people in the room the model was trained in, two at another laptop with weaker echoes and hands further away."""
    root = tmp_path_factory.mktemp("rooms")
    for k, (name, kw) in enumerate((("same1", {}), ("same2", {}), ("other1", OTHER), ("other2", OTHER))):
        room_session(root / f"20261114-16{k:02d}00_{name}_gestures", cfg, name, seed=200 + k, reps=5, fidgets=5, **kw)
    return root


def test_batchnorm_statistics_become_the_rooms():
    torch.manual_seed(0)
    net = GestureNet(5, 4, width=8).eval()
    for block in net.features:                                              # pretend training left some statistics behind
        block[1].running_mean.normal_(0.0, 0.5)
        block[1].running_var.uniform_(0.5, 2.0)
    x = torch.rand(40, 5, 64, 80) * 0.6
    before = [(b[1].running_mean.clone(), b[1].running_var.clone()) for b in net.features]
    # an enormous prior: the trained statistics win, nothing moves
    frozen = __import__("copy").deepcopy(net)
    adapt_batchnorm(frozen, x, None, prior=1e12)
    for b, (m, v) in zip(frozen.features, before):
        torch.testing.assert_close(b[1].running_mean, m, atol=1e-6, rtol=0)
        torch.testing.assert_close(b[1].running_var, v, atol=1e-6, rtol=0)
    # no prior: every layer now holds exactly the statistics of the room pictures, measured behind the adapted layers before it
    room = __import__("copy").deepcopy(net)
    report = adapt_batchnorm(room, x, None, prior=0.0)
    assert len(report) == 4 and report[0]["room_share"] == 1.0
    h = x
    for block in room.features:
        with torch.no_grad():
            z = block[0](h)
        torch.testing.assert_close(block[1].running_mean, z.mean(dim=(0, 2, 3)), atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(block[1].running_var, z.var(dim=(0, 2, 3), unbiased=False), atol=1e-5, rtol=1e-4)
        with torch.no_grad():
            out = block[1](z)                                               # so each layer's output is centred on its own offset again
            assert float(((out.mean(dim=(0, 2, 3)) - block[1].bias).abs().max())) < 1e-3
            h = block[3](block[2](out))
    # weights: letting every class count the same equals repeating the rare class until the batch is balanced
    y = torch.tensor([0] * 30 + [1] * 10)
    weighted = __import__("copy").deepcopy(net)
    adapt_batchnorm(weighted, x, balanced_weights(y, 4), prior=0.0)
    repeated = __import__("copy").deepcopy(net)
    adapt_batchnorm(repeated, torch.cat([x[:30], x[30:].repeat(3, 1, 1, 1)]), None, prior=0.0)
    torch.testing.assert_close(weighted.features[2][1].running_mean, repeated.features[2][1].running_mean, atol=1e-5, rtol=1e-4)
    # in between: three room pictures against a prior of 16 move the statistics by 3/19 of the way
    part = __import__("copy").deepcopy(net)
    rep = adapt_batchnorm(part, x[:3], None, prior=16.0)
    assert rep[0]["room_share"] == pytest.approx(3 / 19, abs=1e-3)
    with torch.no_grad():
        z = net.features[0][0](x[:3])
    want = before[0][0] + (3 / 19) * (z.mean(dim=(0, 2, 3)) - before[0][0])
    torch.testing.assert_close(part.features[0][1].running_mean, want, atol=1e-5, rtol=1e-4)
    assert adapt_batchnorm(part, x[:0]) == []                              # nothing to learn from: nothing changes


def test_prototype_bonus_and_calibration_split():
    torch.manual_seed(1)
    net = GestureNet(5, 4, width=8).eval()
    x = torch.rand(12, 5, 64, 80)
    y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    protos, pmap = make_prototypes(net, x, y, 4)
    assert protos.shape == (3, 32) and pmap.shape == (3, 4) and pmap.sum().item() == 3.0        # class 3 has no example: no prototype
    torch.testing.assert_close(protos.norm(dim=1), torch.ones(3), atol=1e-5, rtol=0)
    base = adapted_module(net)(x)
    torch.testing.assert_close(base, net(x))
    torch.testing.assert_close(adapted_module(net, protos, pmap, 0.0)(x), base)
    with torch.no_grad():
        bonus = adapted_module(net, protos, pmap, 5.0)(x) - base
        likeness = torch.nn.functional.normalize(adapt.embed(net, x), dim=1) @ protos.t()           # how alike picture and prototype are
    assert float(bonus[:, 3].abs().max()) == 0.0                                                   # only classes with a prototype get a bonus
    torch.testing.assert_close(bonus[:, :3], 5.0 * likeness, atol=1e-4, rtol=0)                    # ... of beta x likeness
    one = make_prototypes(net, x[3:4], y[3:4], 4)[0]                                               # the prototype of a single picture is that picture
    assert float((torch.nn.functional.normalize(adapt.embed(net, x[3:4]), dim=1) @ one.t()).item()) == pytest.approx(1.0, abs=1e-5)
    # the first 3 of each gesture + a few non-gestures calibrate, the rest is the test
    order = np.array([1, 0, 2, 1, 0, 0, 2, 1, 2, 1, 0, 0, 2, 1, 2, 0, 0, 0])
    calib, test = split_calibration(order, shots=3)
    assert sorted(order[calib].tolist()) == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2] and len(set(calib) & set(test)) == 0
    assert calib.tolist() == sorted(calib.tolist()) and len(calib) + len(test) == len(order)
    assert all(order[i] == 0 or i > max(calib[order[calib] == order[i]]) for i in test)             # the test comes after the calibration


@pytest.mark.slow
def test_offline_replay_shows_the_gain_in_another_room(cfg, rooms, model):
    net, meta, model_cfg, labels = adapt.load_model(model, torch.device("cpu"))
    data = adapt.load_pictures([str(rooms)], [], labels, model_cfg)
    x = torch.from_numpy(np.ascontiguousarray(center_crop(data["X"], cfg, int(data["pad"]))))
    y = torch.from_numpy(data["y"].astype(np.int64))
    rows, mean = evaluate(net, x, y, data["person"], float(meta["temperature"]), shots=3)
    by = {r["unit"]: r for r in rows}
    assert set(by) == {"same1", "same2", "other1", "other2"} and all("skipped" not in r for r in rows)
    other = [by["other1"], by["other2"]]
    same = [by["same1"], by["same2"]]
    base_recall = np.mean([r["base"]["recall"] for r in other])
    tuned_recall = np.mean([r["bn"]["recall"] for r in other])
    assert base_recall < 0.5, "the other room must be hard for the unadapted model, or this test shows nothing"
    assert tuned_recall >= base_recall + 0.3 and np.mean([r["bn"]["accuracy"] for r in other]) >= np.mean([r["base"]["accuracy"] for r in other]) + 0.15
    # ... and where nothing changed, the adaptation does no harm
    assert np.mean([r["bn"]["accuracy"] for r in same]) >= np.mean([r["base"]["accuracy"] for r in same]) - 0.05
    assert np.mean([r["bn"]["recall"] for r in same]) >= np.mean([r["base"]["recall"] for r in same]) - 0.05
    assert mean["bn"]["units_worse"] <= 1 and mean["bn"]["accuracy"] > mean["base"]["accuracy"]
    # the command line prints the same table and writes it
    out = rooms / "eval.json"
    assert adapt.main(["eval", "--model", str(model), "--data", str(rooms), "--people", "other1,other2", "--device", "cpu", "--json", str(out)]) == 0
    saved = json.loads(out.read_text())
    assert [r["unit"] for r in saved["rows"]] == ["other1", "other2"] and saved["mean"]["bn"]["recall"] == pytest.approx(tuned_recall, abs=1e-6)
    assert adapt.main(["eval", "--model", str(model), "--data", str(rooms), "--people", "nobody", "--device", "cpu"]) == 2


@pytest.mark.slow
def test_fit_writes_a_model_the_engine_loads_and_it_recognises_more(cfg, rooms, model, tmp_path, capsys):
    sessions = {f.name.split("_")[1]: f for f in find_sessions(rooms)}
    out = tmp_path / "room"
    assert adapt.main(["fit", "--model", str(model), "--calib", str(sessions["other1"]), "--out", str(out), "--device", "cpu"]) == 0
    text = capsys.readouterr().out
    assert "BatchNorm statistics" in text and "use it:" in text and "max logit difference" in text
    assert sorted(p.name for p in out.iterdir()) == ["gesture.json", "gesture.onnx", "gesture.pt"]
    meta, old = json.loads((out / "gesture.json").read_text()), json.loads(model.with_suffix(".json").read_text())
    assert meta["labels"] == old["labels"] and meta["temperature"] == old["temperature"] and meta["config"] == old["config"]
    assert meta["adapted"]["batchnorm"]["layers"][0]["room_share"] > 0.3 and "prototypes" not in meta["adapted"]
    assert meta["gesture_peak_db_p10"] <= old["gesture_peak_db_p10"]          # fainter gestures here: the engine's gate comes down
    assert meta["adapted"]["gesture_peak_db_p10"]["trained"] == old["gesture_peak_db_p10"]
    # the engine's loader takes it as it is, and it does better on the OTHER person recorded at that laptop
    base, tuned = load_classifier(model, cfg), load_classifier(out / "gesture.onnx", cfg)
    assert isinstance(tuned, OnnxClassifier) and tuned.mismatch == []
    data = build_dataset([sessions["other2"]], verbose=False)
    x = center_crop(data["X"].astype(np.float32), cfg)
    names = [str(s) for s in data["labels"]]
    gestures = [i for i in range(len(x)) if data["y"][i] != 0]

    def recognised(clf) -> int:
        return sum(clf.predict(x[i])[0] == names[data["y"][i]] and clf.predict(x[i])[1] >= 0.6 for i in gestures)

    assert recognised(tuned) >= recognised(base) + 4, (recognised(base), recognised(tuned), len(gestures))

    # the same through the whole engine: replay the other person's recording, count the gestures it reports
    def replay(clf) -> list[str]:
        backend = FileAudio(cfg, sessions["other2"] / "audio.wav", realtime=False)
        engine = Engine(cfg, backend)
        engine.set_classifier(clf)
        seen = []
        while (rx := backend.read()) is not None:
            seen += [e["label"] for e in engine.step(rx) if e["type"] == "gesture"]
        return seen

    assert len(replay(tuned)) >= len(replay(base)) + 3
    # prototypes on top, and a second adaptation starting from the adapted model, also export cleanly
    assert adapt.main(["fit", "--model", str(out / "gesture.onnx"), "--calib", str(sessions["other1"]), "--out", str(tmp_path / "room2"),
                       "--prototypes", "--device", "cpu", "--room", str(sessions["other2"])]) == 0
    meta2 = json.loads((tmp_path / "room2" / "gesture.json").read_text())
    assert set(meta2["adapted"]["prototypes"]["classes"]) == {"none", "swipe_left", "swipe_right", "push", "pull"}
    assert OnnxClassifier(tmp_path / "room2" / "gesture.onnx", cfg).predict(x[0])[0] in names
    # a model without its .pt cannot be adapted: say so
    lone = tmp_path / "lone"
    lone.mkdir()
    for ext in (".onnx", ".json"):
        (lone / f"gesture{ext}").write_bytes(model.with_suffix(ext).read_bytes())
    assert adapt.main(["fit", "--model", str(lone / "gesture.onnx"), "--calib", str(sessions["other1"]), "--out", str(tmp_path / "x")]) == 2
    assert "needs gesture.json and gesture.pt" in capsys.readouterr().err
