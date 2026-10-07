# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Tic class for timing tasks (code profiling)."""

from __future__ import annotations
import time

from ._requires import Create_requires_decorator
from ._mpi import MPI_RANK, CAN_USE_MPI

if CAN_USE_MPI:
    from mpi4py import MPI

requires_matplotlib = Create_requires_decorator("matplotlib")


class Tic:

    __get_time = MPI.Wtime if CAN_USE_MPI else time.perf_counter

    def __init__(self):
        self.__start = Tic.__get_time()

    @staticmethod
    def Get_time_unity(time: float) -> tuple[float, str]:
        """Returns time with unity"""
        if time >= 86400:
            unite, coef = "j", 1 / 86400
        elif time >= 3600:
            unite, coef = "h", 1 / 3600
        elif time >= 60:
            unite, coef = "m", 1 / 60
        elif time >= 1:
            unite, coef = "s", 1.0
        elif time >= 1e-3:
            unite, coef = "ms", 1e3
        else:
            unite, coef = "µs", 1e6

        return time * coef, unite

    @staticmethod
    def Get_Remaining_Time(i: int, N: int, time: float) -> str:
        """Returns remaining time asssuming that time is in s."""

        if i == 0:
            return ""
        else:
            timeLeft = (N - i) * time
            timeLeft, unit = Tic.Get_time_unity(timeLeft)
            return f"({i / N * 100:3.2f} %) {timeLeft:3.2f} {unit}"

    def Tac(self, category="", text="", verbosity=False) -> float:
        """Returns the time elapsed since the last `Tic` or `Tac`."""

        tf = Tic.__get_time() - self.__start

        tfCoef, unite = Tic.Get_time_unity(tf)

        textWithTime = f"{text} ({tfCoef:.3f} {unite})"

        if category not in Tic.__History:
            Tic.__History[category] = {}
        cat = Tic.__History[category]
        if text not in cat:
            cat[text] = [0.0, 0]
        cat[text][0] += tf
        cat[text][1] += 1

        self.__start = Tic.__get_time()

        if verbosity and MPI_RANK == 0:
            print(textWithTime)

        return tf

    @staticmethod
    def Clear() -> None:
        """Deletes history."""
        Tic.__History = {}

    __History: dict[str, dict[str, list]] = {}
    """history = { category: { text: [total_time, count] } }"""

    @staticmethod
    def nTic() -> int:
        return len(Tic.__History)

    @staticmethod
    def Resume(verbosity=True) -> str:
        """Returns the TicTac summary"""

        if Tic.__History == {}:
            return ""

        resume = ""

        for category in Tic.__History:
            timesCategory = float(sum(v[0] for v in Tic.__History[category].values()))
            timesCategory, unite = Tic.Get_time_unity(timesCategory)
            resumeCategory = f"{category}: {timesCategory:.3f} {unite}"
            if verbosity:
                print(resumeCategory)
            resume += "\n" + resumeCategory

        return resume

    @staticmethod
    @requires_matplotlib
    def Plot_History(folder="", details=False) -> None:
        """Plots history.

        Parameters
        ----------
        folder : str, optional
            save folder, by default ""
        details : bool, optional
            History details, by default False
        """
        from . import Matplotlib

        Matplotlib._Plot_Tic_History(Tic.__History, folder, details)
