"""Tests for the Event classes and EventList container."""
import numpy as np
import pytest

from src.pdmodel import Event, EventList, EclipseLC


def test_event_center_zero_is_valid():
    assert Event(center=0.0).center == 0.0


def test_event_center_from_bbox():
    ev = Event(bbox=(1.0, 3.0))
    assert ev.center == 2.0


def test_event_ordering():
    e1, e2 = Event(center=1.0), Event(center=2.0)
    assert e1 < e2 and e2 > e1
    assert e1 < 1.5 and e2 > 1.5
    assert sorted([e2, e1])[0] is e1


def test_eventlist_iadd_sorts_and_tracks_tref():
    el = EventList(1.5)
    el += [Event(center=2.0), Event(center=1.0)]
    assert [e.center for e in el] == [1.0, 2.0]
    assert el.itref == 0  # last event before tref=1.5


def test_eventlist_add_returns_new_list_with_tref():
    el = EventList(1.5, [Event(center=1.0)])
    el2 = el + [Event(center=2.0)]
    assert isinstance(el2, EventList)
    assert len(el2) == 2 and len(el) == 1
    assert el2.tref == 1.5 and el2.itref == 0


def test_eclipselc_requires_fr(two_body_sim):
    time = np.linspace(0.9, 1.1, 10)
    ev = EclipseLC(time, 0.0, [0], 0, 1, 0.0, np.ones(time.size))
    with pytest.raises(ValueError, match="flux ratios"):
        ev.compute(two_body_sim, k=np.array([0.1]), fr=None)
