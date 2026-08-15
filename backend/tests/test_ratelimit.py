"""Failed-login throttling."""

from __future__ import annotations

from app.core.ratelimit import LoginThrottle


def _throttle(**kwargs) -> LoginThrottle:
    defaults = {"max_failures": 3, "window_seconds": 60.0, "lockout_seconds": 120.0}
    return LoginThrottle(**{**defaults, **kwargs})


def test_a_fresh_key_is_allowed():
    assert _throttle().retry_after("a|b") == 0.0


def test_failures_below_the_limit_do_not_lock():
    throttle = _throttle()
    throttle.record_failure("k", now=0.0)
    throttle.record_failure("k", now=1.0)

    assert throttle.retry_after("k", now=2.0) == 0.0


def test_the_limit_triggers_a_lockout():
    throttle = _throttle()
    for i in range(3):
        throttle.record_failure("k", now=float(i))

    assert throttle.retry_after("k", now=3.0) > 0


def test_the_lockout_expires():
    throttle = _throttle()
    for i in range(3):
        throttle.record_failure("k", now=float(i))

    assert throttle.retry_after("k", now=1000.0) == 0.0


def test_failures_outside_the_window_do_not_accumulate():
    # Two typos an hour apart are not an attack, and must never add up to one.
    throttle = _throttle()
    throttle.record_failure("k", now=0.0)
    throttle.record_failure("k", now=1.0)
    throttle.record_failure("k", now=500.0)

    assert throttle.retry_after("k", now=501.0) == 0.0


def test_a_successful_sign_in_clears_the_history():
    throttle = _throttle()
    throttle.record_failure("k", now=0.0)
    throttle.record_failure("k", now=1.0)
    throttle.reset("k")
    throttle.record_failure("k", now=2.0)

    assert throttle.retry_after("k", now=3.0) == 0.0


def test_keys_are_independent():
    # Keyed per (client, email), so failing logins against someone else's
    # address cannot lock them out of their own account.
    throttle = _throttle()
    for i in range(3):
        throttle.record_failure("attacker|victim@example.com", now=float(i))

    assert throttle.retry_after("victim|victim@example.com", now=4.0) == 0.0


def test_tracked_keys_stay_bounded():
    throttle = _throttle()
    for i in range(6000):
        throttle.record_failure(f"key-{i}", now=float(i))

    assert len(throttle._entries) <= 4096  # noqa: SLF001 - asserting the bound
