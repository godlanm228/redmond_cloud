"""
Computed day/situation state for proactive decisions.

This module deliberately does not own storage. It reads existing sources of
truth (`coach_storage`, `week_schedule`) and turns them into a small read-model
for policy code such as `logic.pings`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from logic import coach_storage
from logic.week_schedule import STUDY_TIMETABLE, get_shift, get_shift_record
from utils.time import now_local


def parse_hm(s: str, base: datetime) -> Optional[datetime]:
    try:
        h, m = map(int, str(s).split(":"))
        return base.replace(hour=h, minute=m, second=0, microsecond=0)
    except (ValueError, AttributeError):
        return None


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _same_tz(dt: datetime, base: datetime) -> datetime:
    if dt.tzinfo is None and base.tzinfo is not None:
        return dt.replace(tzinfo=base.tzinfo)
    return dt


@dataclass(frozen=True)
class ShiftSituation:
    record: Optional[Dict[str, Any]]
    active_record: Optional[Dict[str, Any]]
    start_at: Optional[datetime]
    end_at: Optional[datetime]
    status: str
    source: str
    confidence: str
    updated_at: Optional[datetime]
    confirmed_at: Optional[datetime]

    @property
    def active(self) -> bool:
        return self.active_record is not None and self.start_at is not None

    def confirmed_on(self, day: datetime) -> bool:
        return bool(self.confirmed_at and self.confirmed_at.date() == day.date())

    def minutes_until_start(self, now: datetime) -> Optional[int]:
        if not self.start_at:
            return None
        return int((self.start_at - now).total_seconds() // 60)

    def needs_confirmation(self, now: datetime) -> bool:
        """Whether it is useful to ask once if today's shift is still real.

        Conservative by design: confirmed/high-confidence text shifts do not get
        daily confirmation pings. We ask for uncertain/legacy/stale records only.
        """
        if not self.active or self.confirmed_on(now):
            return False
        mins = self.minutes_until_start(now)
        if mins is None or not (75 <= mins <= 180):
            return False
        if self.status == "uncertain":
            return True
        if self.source in ("", "unknown", "legacy"):
            return True
        if self.confidence not in ("high", "certain"):
            return True
        # A schedule imported long ago may be stale enough to deserve one check.
        if self.updated_at is None:
            return True
        return (now - self.updated_at) > timedelta(days=7)


@dataclass(frozen=True)
class DaySituation:
    now: datetime
    day_state: Dict[str, Any]
    pings: Dict[str, str]
    owner_seen: bool
    muted: bool
    tags: set
    entries_today: int
    wake_time: Optional[str]
    shift: ShiftSituation
    in_study_block: bool

    @property
    def has_work_today(self) -> bool:
        return "работа" in self.tags

    @property
    def has_meal_today(self) -> bool:
        return "питание" in self.tags

    @property
    def has_training_today(self) -> bool:
        return "спорт" in self.tags

    @property
    def has_study_today(self) -> bool:
        return bool({"учёба", "учеба"} & self.tags)

    def last_ping_at(self) -> Optional[datetime]:
        if not self.pings:
            return None
        return max(parse_hm(t, self.now) or self.now for t in self.pings.values())

    def proactive_allowed(self, max_pings: int, min_gap_min: int) -> bool:
        if self.muted or len(self.pings) >= max_pings or self.in_study_block:
            return False
        last = self.last_ping_at()
        return last is None or (self.now - last) >= timedelta(minutes=min_gap_min)


def _build_shift_situation(now: datetime) -> ShiftSituation:
    record = get_shift_record(now.date())
    active = get_shift(now.date())
    start_at = parse_hm(active.get("start"), now) if active else None
    end_at = parse_hm(active.get("end"), now) if active else None
    return ShiftSituation(
        record=record,
        active_record=active,
        start_at=start_at,
        end_at=end_at,
        status=str((record or {}).get("status") or "planned").strip().lower(),
        source=str((record or {}).get("source") or "legacy").strip().lower(),
        confidence=str((record or {}).get("confidence") or "medium").strip().lower(),
        updated_at=(
            _same_tz(parsed, now)
            if (parsed := _parse_iso((record or {}).get("updated"))) else None
        ),
        confirmed_at=(
            _same_tz(parsed, now)
            if (parsed := _parse_iso((record or {}).get("last_confirmed_at"))) else None
        ),
    )


def _in_study_block(now: datetime) -> bool:
    for start, end, _what in STUDY_TIMETABLE.get(now.weekday(), []):
        s = parse_hm(start, now)
        e = parse_hm(end, now)
        if s and e and (s - timedelta(minutes=30)) <= now <= e:
            return True
    return False


def build_day_situation(now: Optional[datetime] = None) -> DaySituation:
    current = now or now_local()
    state = coach_storage.get_day_state()
    return DaySituation(
        now=current,
        day_state=state,
        pings=dict(state.get("pings", {})),
        owner_seen=coach_storage.owner_seen_today(),
        muted=coach_storage.muted_now(),
        tags=coach_storage.today_tags(),
        entries_today=coach_storage.entries_today(),
        wake_time=coach_storage.wake_time_today(),
        shift=_build_shift_situation(current),
        in_study_block=_in_study_block(current),
    )
