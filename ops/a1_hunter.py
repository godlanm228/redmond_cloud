"""A1 capacity hunter: waits for free Ampere capacity and grabs it.

The hub runs on an Always Free VM.Standard.E2.1.Micro (1/8 OCPU, 1 GB RAM).
The same free tier includes Ampere A1 capacity (this account: up to 2 OCPU /
12 GB per availability domain), but in Frankfurt it is usually sold out:
on Sep 28, 2026 every AD answered OUT_OF_HOST_CAPACITY, even for 1 OCPU / 2 GB.
Capacity frees up at random moments, so the only way to get it is to be
there when it does.

How this differs from the usual "launch every minute" loop:

  * Scout, then shoot. A compute capacity report tells whether a shape fits in
    an AD without creating anything. One report per AD covers every shape
    config at once. Only when a report says AVAILABLE do we launch.
  * A blind launch attempt still runs every BLIND_EVERY seconds, because the
    report can lag behind reality.
  * Randomised intervals, exponential backoff on 429 (60 s up to 1 h), and a
    long pause plus one Telegram message on errors that retries cannot fix
    (missing permissions, quota).
  * Idempotent: if an instance with our display name already exists, the
    hunter reports it and exits instead of creating a second one.

About 3 capacity reports every ~8 minutes plus ~30 launch attempts a day,
far below anything that looks like abuse.

The core (Hunter) knows nothing about the OCI SDK: it talks to a small
adapter, so the decision logic is tested without network or credentials.
Run on the VM: python -m ops.a1_hunter (see deploy/a1-hunter.service).
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("a1_hunter")

SHAPE = "VM.Standard.A1.Flex"
AVAILABLE = "AVAILABLE"


# ---------------------------------------------------------------------------
# Errors the adapter maps provider failures into
# ---------------------------------------------------------------------------

class HunterError(Exception):
    """Base for classified provider failures."""


class NoCapacity(HunterError):
    """Out of host capacity: expected, try again later."""


class Throttled(HunterError):
    """429 from the API: slow down."""


class NotAuthorized(HunterError):
    """Missing IAM permissions: retrying will not help."""


class QuotaExceeded(HunterError):
    """Service limit reached: retrying will not help."""


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _shapes(raw: str) -> List[Tuple[int, int]]:
    """'2:12,1:6' -> [(2, 12), (1, 6)], largest first as written."""
    out = []
    for part in (raw or "").split(","):
        part = part.strip()
        if part:
            ocpus, mem = part.split(":")
            out.append((int(ocpus), int(mem)))
    return out


@dataclass
class Settings:
    compartment_id: str
    subnet_id: str
    ssh_public_key: str
    availability_domains: List[str]
    image_id: str = ""                      # empty: newest Ubuntu 24.04 aarch64
    display_name: str = "redmond-a1"
    shapes: List[Tuple[int, int]] = field(default_factory=lambda: [(2, 12), (1, 6)])
    boot_volume_gb: int = 50
    probe_every: int = 480                  # seconds between scouting rounds
    jitter: int = 90
    blind_every: int = 2700                 # a launch attempt without a report
    state_file: str = str(Path.home() / ".a1-hunter.state.json")
    telegram_token: str = ""
    telegram_chat: str = ""

    @classmethod
    def from_env(cls, env: Optional[Dict[str, str]] = None) -> "Settings":
        e = dict(os.environ if env is None else env)
        key = e.get("A1_SSH_PUBLIC_KEY", "")
        key_file = e.get("A1_SSH_PUBLIC_KEY_FILE", "")
        if not key and key_file:
            key = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
        return cls(
            compartment_id=e["A1_COMPARTMENT_ID"],
            subnet_id=e["A1_SUBNET_ID"],
            ssh_public_key=key,
            availability_domains=[a.strip() for a in e["A1_ADS"].split(",") if a.strip()],
            image_id=e.get("A1_IMAGE_ID", ""),
            display_name=e.get("A1_DISPLAY_NAME", "redmond-a1"),
            shapes=_shapes(e.get("A1_SHAPES", "2:12,1:6")),
            boot_volume_gb=int(e.get("A1_BOOT_GB", "50")),
            probe_every=int(e.get("A1_PROBE_EVERY", "480")),
            jitter=int(e.get("A1_JITTER", "90")),
            blind_every=int(e.get("A1_BLIND_EVERY", "2700")),
            state_file=e.get("A1_STATE_FILE", str(Path.home() / ".a1-hunter.state.json")),
            telegram_token=e.get("TELEGRAM_BOT_TOKEN", ""),
            telegram_chat=e.get("MAIN_CHAT_ID", ""),
        )


# ---------------------------------------------------------------------------
# The decision loop
# ---------------------------------------------------------------------------

BACKOFF_START = 60
BACKOFF_MAX = 3600
STUCK_PAUSE = 6 * 3600          # after an error retries cannot fix


class Hunter:
    """Scout-then-shoot loop. `compute` is an adapter (see OciCompute)."""

    def __init__(self, settings: Settings, compute: Any,
                 notify: Callable[[str], None],
                 clock: Callable[[], float] = time.time,
                 rng: Optional[random.Random] = None):
        self.s = settings
        self.compute = compute
        self.notify = notify
        self.clock = clock
        self.rng = rng or random.Random()
        self.backoff = 0
        self.last_blind = clock()
        self.blind_turn = 0
        self.stats: Dict[str, Any] = {"started": clock(), "rounds": 0, "reports": 0,
                                      "launches": 0, "throttled": 0, "last": {}}
        self.done = False
        self._warned: set = set()

    # -- one round ------------------------------------------------------------

    def tick(self) -> float:
        """Run one round. Returns seconds to sleep before the next one."""
        self.stats["rounds"] += 1
        try:
            existing = self.compute.existing(self.s.display_name)
            if existing:
                self._finish(f"🦞 Сервер «{self.s.display_name}» уже есть ({existing}) — "
                             f"ловец остановлен, второй не создаю.")
                return 0

            for ad in self._ads_in_random_order():
                self.stats["reports"] += 1
                report = self.compute.capacity(ad, self.s.shapes)
                self.stats["last"][ad] = {f"{o}/{g}": st for (o, g), st in report.items()}
                for ocpus, mem in self.s.shapes:
                    if report.get((ocpus, mem)) == AVAILABLE:
                        if self._try_launch(ad, ocpus, mem):
                            return 0

            if self.clock() - self.last_blind >= self.s.blind_every:
                self.last_blind = self.clock()
                ad = self.s.availability_domains[self.blind_turn % len(self.s.availability_domains)]
                self.blind_turn += 1
                for ocpus, mem in self.s.shapes:
                    if self._try_launch(ad, ocpus, mem):
                        return 0

            self.backoff = 0
            return self._interval()

        except Throttled:
            self.stats["throttled"] += 1
            self.backoff = min(BACKOFF_MAX, max(BACKOFF_START, self.backoff * 2))
            logger.warning("429 от OCI — жду %d с", self.backoff)
            return self.backoff
        except NotAuthorized as e:
            self._warn_once("auth", f"🦞 Ловцу A1 не хватает прав в Oracle: {e}. "
                                    f"Проверь динамическую группу и политику.")
            return STUCK_PAUSE
        except QuotaExceeded as e:
            self._warn_once("quota", f"🦞 Ловец A1 упёрся в лимит аккаунта: {e}.")
            return STUCK_PAUSE
        except HunterError as e:
            logger.warning("Сбой OCI: %s", e)
            self.backoff = min(BACKOFF_MAX, max(BACKOFF_START, self.backoff * 2))
            return self.backoff

    def _try_launch(self, ad: str, ocpus: int, mem: int) -> bool:
        self.stats["launches"] += 1
        try:
            info = self.compute.launch(ad, ocpus, mem, self.s)
        except NoCapacity:
            logger.info("%s %d/%d: места нет", ad, ocpus, mem)
            return False
        size = f"{ocpus} ядра / {mem} ГБ" if ocpus > 1 else f"{ocpus} ядро / {mem} ГБ"
        ip = info.get("public_ip") or "IP появится в консоли через минуту"
        extra = ("" if (ocpus, mem) == self.s.shapes[0]
                 else f" Это меньше целевых {self.s.shapes[0][0]}/{self.s.shapes[0][1]} — "
                      f"расширим, когда в зоне освободится место.")
        self._finish(f"🦞 Поймал ARM-сервер: {size}, {ad.split(':')[-1]}, {ip}.{extra} "
                     f"Переезд хаба — по твоей команде.")
        return True

    # -- helpers --------------------------------------------------------------

    def _ads_in_random_order(self) -> List[str]:
        ads = list(self.s.availability_domains)
        self.rng.shuffle(ads)
        return ads

    def _interval(self) -> float:
        return max(60.0, self.s.probe_every + self.rng.uniform(-self.s.jitter, self.s.jitter))

    def _finish(self, text: str) -> None:
        self.done = True
        logger.info(text)
        self.notify(text)

    def _warn_once(self, kind: str, text: str) -> None:
        logger.error(text)
        if kind not in self._warned:
            self._warned.add(kind)
            self.notify(text)

    def save_state(self) -> None:
        # The state file is a status page, not something the hunt depends on:
        # a failure to write it must never stop the loop.
        try:
            Path(self.s.state_file).write_text(json.dumps(self.stats, ensure_ascii=False,
                                                          default=str), encoding="utf-8")
        except Exception:  # noqa: BLE001
            logger.warning("Не удалось записать состояние", exc_info=True)

    def run(self, sleep: Callable[[float], None] = time.sleep) -> None:
        logger.info("Ловец A1 запущен: зоны %s, размеры %s",
                    ", ".join(self.s.availability_domains), self.s.shapes)
        while not self.done:
            pause = self.tick()
            self.save_state()
            if self.done:
                break
            sleep(pause)


# ---------------------------------------------------------------------------
# OCI adapter (the only part that touches the SDK)
# ---------------------------------------------------------------------------

class OciCompute:
    """Thin adapter over oci.core.ComputeClient with instance-principal auth."""

    _ALIVE = {"PROVISIONING", "STARTING", "RUNNING", "STOPPING", "STOPPED", "MOVING", "CREATING_IMAGE"}

    def __init__(self, settings: Settings):
        import oci  # imported here: tests and the hub itself do not need the SDK
        self.oci = oci
        self.s = settings
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        self.compute = oci.core.ComputeClient(config={}, signer=signer)
        self.network = oci.core.VirtualNetworkClient(config={}, signer=signer)

    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except self.oci.exceptions.ServiceError as e:
            text = f"{e.status} {e.code}: {e.message}"
            if e.status == 429:
                raise Throttled(text) from e
            if "out of host capacity" in (e.message or "").lower() or e.code == "OutOfHostCapacity":
                raise NoCapacity(text) from e
            if e.status in (401, 403, 404) or e.code == "NotAuthorizedOrNotFound":
                raise NotAuthorized(text) from e
            if e.code in ("LimitExceeded", "QuotaExceeded"):
                raise QuotaExceeded(text) from e
            raise HunterError(text) from e
        except self.oci.exceptions.RequestException as e:
            raise HunterError(f"сеть: {e}") from e

    def existing(self, name: str) -> str:
        items = self._call(self.compute.list_instances,
                           compartment_id=self.s.compartment_id, display_name=name).data
        alive = [i for i in items if i.lifecycle_state in self._ALIVE]
        return alive[0].lifecycle_state if alive else ""

    def capacity(self, ad: str, shapes: List[Tuple[int, int]]) -> Dict[Tuple[int, int], str]:
        m = self.oci.core.models
        details = m.CreateComputeCapacityReportDetails(
            compartment_id=self.s.compartment_id,
            availability_domain=ad,
            shape_availabilities=[
                m.CreateCapacityReportShapeAvailabilityDetails(
                    instance_shape=SHAPE,
                    instance_shape_config=m.CapacityReportInstanceShapeConfig(
                        ocpus=o, memory_in_gbs=g))
                for o, g in shapes
            ],
        )
        data = self._call(self.compute.create_compute_capacity_report, details).data
        return {shapes[i]: a.availability_status for i, a in enumerate(data.shape_availabilities)}

    def _image(self) -> str:
        if self.s.image_id:
            return self.s.image_id
        images = self._call(self.compute.list_images, compartment_id=self.s.compartment_id,
                            operating_system="Canonical Ubuntu", operating_system_version="24.04",
                            shape=SHAPE, sort_by="TIMECREATED", sort_order="DESC", limit=1).data
        if not images:
            raise HunterError("нет образа Ubuntu 24.04 для A1")
        return images[0].id

    def launch(self, ad: str, ocpus: int, mem: int, s: Settings) -> Dict[str, str]:
        m = self.oci.core.models
        details = m.LaunchInstanceDetails(
            availability_domain=ad,
            compartment_id=s.compartment_id,
            display_name=s.display_name,
            shape=SHAPE,
            shape_config=m.LaunchInstanceShapeConfigDetails(ocpus=ocpus, memory_in_gbs=mem),
            source_details=m.InstanceSourceViaImageDetails(
                image_id=self._image(), boot_volume_size_in_gbs=s.boot_volume_gb),
            create_vnic_details=m.CreateVnicDetails(subnet_id=s.subnet_id, assign_public_ip=True),
            metadata={"ssh_authorized_keys": s.ssh_public_key},
        )
        inst = self._call(self.compute.launch_instance, details).data
        return {"id": inst.id, "public_ip": self._public_ip(inst.id)}

    def _public_ip(self, instance_id: str, wait: int = 600) -> str:
        """Best effort: wait for RUNNING and read the public IP."""
        deadline = time.time() + wait
        while time.time() < deadline:
            try:
                state = self.compute.get_instance(instance_id).data.lifecycle_state
                if state == "RUNNING":
                    att = self.compute.list_vnic_attachments(
                        compartment_id=self.s.compartment_id, instance_id=instance_id).data
                    if att:
                        return self.network.get_vnic(att[0].vnic_id).data.public_ip or ""
            except Exception:  # noqa: BLE001 — IP is a nicety, the instance exists
                logger.warning("Не удалось прочитать IP", exc_info=True)
                return ""
            time.sleep(15)
        return ""


def telegram_notifier(token: str, chat: str) -> Callable[[str], None]:
    """Send as the Redmond bot. Failures are logged, never raised."""
    def send(text: str) -> None:
        if not (token and chat):
            logger.warning("Telegram не настроен, сообщение: %s", text)
            return
        try:
            import requests
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": text}, timeout=20)
            if r.status_code != 200:
                logger.warning("Telegram ответил %s: %s", r.status_code, r.text[:200])
        except Exception:  # noqa: BLE001
            logger.warning("Не удалось отправить в Telegram", exc_info=True)
    return send


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    args = argv if argv is not None else sys.argv[1:]
    settings = Settings.from_env()
    compute = OciCompute(settings)
    if "--check" in args:
        # One scouting round, no launch: verifies permissions end to end.
        print("existing:", compute.existing(settings.display_name) or "нет")
        for ad in settings.availability_domains:
            print(ad, compute.capacity(ad, settings.shapes))
        return 0
    hunter = Hunter(settings, compute, telegram_notifier(settings.telegram_token,
                                                         settings.telegram_chat))
    hunter.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
