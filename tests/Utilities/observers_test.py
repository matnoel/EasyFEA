# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

import pytest

from EasyFEA.Utilities import Observable, _IObserver


class Recorder(_IObserver):
    def __init__(self):
        self.events: list[tuple[Observable, str]] = []

    def _Update(self, observable: Observable, event: str) -> None:
        self.events.append((observable, event))


def test_Observable_starts_without_observers():
    assert Observable().observers == []


def test_Observable_observers_is_a_copy():
    observable = Observable()
    observable.observers.append(Recorder())
    assert observable.observers == []


def test_Observable__Add_observer_adds_an_observer_once():
    observable, observer = Observable(), Recorder()
    observable._Add_observer(observer)
    observable._Add_observer(observer)
    assert observable.observers == [observer]


def test_Observable__Add_observer_rejects_a_non_observer(capsys):
    observable = Observable()
    observable._Add_observer(object())
    assert observable.observers == []
    assert "observer must be an _IObserver" in capsys.readouterr().out


def test_Observable__Remove_observer_removes_it():
    observable, observer = Observable(), Recorder()
    observable._Add_observer(observer)
    observable._Remove_observer(observer)
    assert observable.observers == []


def test_Observable__Remove_observer_ignores_an_unknown_observer():
    observable, observer = Observable(), Recorder()
    observable._Add_observer(observer)
    observable._Remove_observer(Recorder())
    assert observable.observers == [observer]


def test_Observable__Notify_sends_the_event_to_every_observer():
    observable, first, second = Observable(), Recorder(), Recorder()
    observable._Add_observer(first)
    observable._Add_observer(second)
    observable._Notify("meshChanged")
    assert first.events == second.events == [(observable, "meshChanged")]


def test_IObserver_cannot_be_instantiated_without__Update():
    with pytest.raises(TypeError):
        _IObserver()  # type: ignore[abstract]


def test_IObserver__Update_is_called_on_each_notification():
    observable, observer = Observable(), Recorder()
    observable._Add_observer(observer)
    observable._Notify("first")
    observable._Notify("second")
    assert observer.events == [(observable, "first"), (observable, "second")]
