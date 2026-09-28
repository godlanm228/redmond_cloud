"""The A1 hunter's decisions, with a fake OCI and no network.

The real account state on Sep 28, 2026 is the starting point: every AD in
Frankfurt reported OUT_OF_HOST_CAPACITY for A1.Flex, even at 1 OCPU / 2 GB.
"""

import random



from ops import a1_hunter as h

ADS = ["iYlP:EU-FRANKFURT-1-AD-1", "iYlP:EU-FRANKFURT-1-AD-2", "iYlP:EU-FRANKFURT-1-AD-3"]
OUT = "OUT_OF_HOST_CAPACITY"


def settings(**kw):
    base = dict(compartment_id="c", subnet_id="s", ssh_public_key="ssh-rsa AAA",
                availability_domains=ADS, probe_every=480, jitter=90, blind_every=2700,
                state_file="unused")
    base.update(kw)
    return h.Settings(**base)


class FakeCompute:
    def __init__(self, reports=None, launch_results=None, existing=""):
        self.reports = reports or {}          # ad -> {(o, m): status}
        self.launch_results = list(launch_results or [])
        self.existing_state = existing
        self.launched = []
        self.report_calls = 0

    def existing(self, name):
        return self.existing_state

    def capacity(self, ad, shapes):
        self.report_calls += 1
        r = self.reports.get(ad, {s: OUT for s in shapes})
        if isinstance(r, Exception):
            raise r
        return r

    def launch(self, ad, ocpus, mem, s):
        self.launched.append((ad, ocpus, mem))
        result = self.launch_results.pop(0) if self.launch_results else h.NoCapacity("out")
        if isinstance(result, Exception):
            raise result
        return result


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def hunter(compute, **kw):
    sent = []
    clock = Clock()
    hu = h.Hunter(settings(**kw), compute, sent.append, clock=clock, rng=random.Random(1))
    return hu, sent, clock


def test_sold_out_everywhere_means_scout_and_wait():
    """Sep 28 state: no launch attempts, one report per AD, next round in ~8 min."""
    compute = FakeCompute()
    hu, sent, _ = hunter(compute)
    pause = hu.tick()
    assert compute.launched == []
    assert compute.report_calls == 3
    assert 480 - 90 <= pause <= 480 + 90
    assert sent == [] and not hu.done


def test_available_report_triggers_an_immediate_launch():
    compute = FakeCompute(reports={ADS[1]: {(2, 12): "AVAILABLE", (1, 6): "AVAILABLE"}},
                          launch_results=[{"id": "i1", "public_ip": "198.51.100.7"}])
    hu, sent, _ = hunter(compute)
    hu.tick()
    assert compute.launched == [(ADS[1], 2, 12)]
    assert hu.done
    assert "198.51.100.7" in sent[0] and "2 ядра / 12 ГБ" in sent[0]


def test_smaller_shape_is_taken_when_only_it_fits():
    compute = FakeCompute(reports={ADS[0]: {(2, 12): OUT, (1, 6): "AVAILABLE"}},
                          launch_results=[{"id": "i1", "public_ip": ""}])
    hu, sent, _ = hunter(compute)
    hu.tick()
    assert compute.launched == [(ADS[0], 1, 6)]
    assert "меньше целевых 2/12" in sent[0]


def test_report_can_lie_and_launch_fails_gracefully():
    """AVAILABLE in the report, but the launch hits no capacity: keep hunting."""
    compute = FakeCompute(reports={ADS[2]: {(2, 12): "AVAILABLE", (1, 6): OUT}},
                          launch_results=[h.NoCapacity("Out of host capacity.")])
    hu, sent, _ = hunter(compute)
    pause = hu.tick()
    assert not hu.done and sent == []
    assert pause >= 60


def test_blind_attempt_runs_on_schedule_and_rotates_ads():
    compute = FakeCompute()
    hu, _sent, clock = hunter(compute)
    hu.tick()
    assert compute.launched == [], "blind attempt before its time"
    clock.t += 2700
    hu.tick()
    assert compute.launched == [(ADS[0], 2, 12), (ADS[0], 1, 6)]
    clock.t += 2700
    hu.tick()
    assert compute.launched[-2:] == [(ADS[1], 2, 12), (ADS[1], 1, 6)]


def test_throttling_backs_off_exponentially_up_to_an_hour():
    compute = FakeCompute(reports={ad: h.Throttled("429") for ad in ADS})
    hu, _sent, _ = hunter(compute)
    pauses = [hu.tick() for _ in range(8)]
    assert pauses[:4] == [60, 120, 240, 480]
    assert pauses[-1] == 3600


def test_backoff_resets_after_a_clean_round():
    compute = FakeCompute(reports={ad: h.Throttled("429") for ad in ADS})
    hu, _sent, _ = hunter(compute)
    hu.tick(); hu.tick()
    compute.reports = {}
    assert hu.tick() <= 480 + 90
    assert hu.backoff == 0


def test_missing_permissions_are_reported_once_and_not_hammered():
    compute = FakeCompute(reports={ad: h.NotAuthorized("404 NotAuthorizedOrNotFound") for ad in ADS})
    hu, sent, _ = hunter(compute)
    first, second = hu.tick(), hu.tick()
    assert first == second == h.STUCK_PAUSE
    assert len(sent) == 1 and "прав" in sent[0]


def test_never_creates_a_second_instance():
    compute = FakeCompute(reports={ADS[0]: {(2, 12): "AVAILABLE", (1, 6): "AVAILABLE"}},
                          existing="RUNNING")
    hu, sent, _ = hunter(compute)
    hu.tick()
    assert compute.launched == []
    assert hu.done and "уже есть" in sent[0]


def test_run_stops_after_success():
    compute = FakeCompute(reports={ADS[0]: {(2, 12): "AVAILABLE", (1, 6): OUT}},
                          launch_results=[{"id": "i1", "public_ip": "198.51.100.7"}])
    hu, _sent, _ = hunter(compute, state_file=str(__import__("pathlib").Path(
        __import__("tempfile").mkdtemp()) / "state.json"))
    sleeps = []
    hu.run(sleep=sleeps.append)
    assert hu.done and sleeps == []


def test_settings_from_env():
    env = {"A1_COMPARTMENT_ID": "c", "A1_SUBNET_ID": "s", "A1_SSH_PUBLIC_KEY": "ssh-rsa AAA",
           "A1_ADS": ",".join(ADS), "A1_SHAPES": "2:12, 1:6", "TELEGRAM_BOT_TOKEN": "t",
           "MAIN_CHAT_ID": "-100"}
    s = h.Settings.from_env(env)
    assert s.shapes == [(2, 12), (1, 6)]
    assert s.availability_domains == ADS
    assert s.telegram_chat == "-100"


def test_state_file_is_valid_json_after_a_round(tmp_path):
    """Shape configs are tuples in memory; the status file must still serialise."""
    import json
    state = tmp_path / "state.json"
    hu, _sent, _ = hunter(FakeCompute(), state_file=str(state))
    hu.tick()
    hu.save_state()
    data = json.loads(state.read_text(encoding="utf-8"))
    assert data["last"][ADS[0]] == {"2/12": OUT, "1/6": OUT}


def test_importing_the_hunter_does_not_import_the_oci_sdk():
    """CI and the hub run without the SDK; only the adapter needs it."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run([sys.executable, "-c",
                          "import sys, ops.a1_hunter; print('oci' in sys.modules)"],
                         capture_output=True, text=True, cwd=root, stdin=subprocess.DEVNULL)
    assert out.stdout.strip() == "False", out.stderr
