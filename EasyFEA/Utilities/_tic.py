# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Tic class for timing tasks (code profiling)."""

from __future__ import annotations
import time
import numpy as np

from ._requires import Create_requires_decorator

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
requires_matplotlib = Create_requires_decorator("matplotlib")


class Tic:
    def __init__(self):
        self.__start = time.time()

    @staticmethod
    def Get_time_unity(time: float) -> tuple[float, str]:
        """Returns time with unity"""
        if time > 1:
            if time < 60:
                unite = "s"
                coef = 1.0
            elif time > 60 and time < 3600:
                unite = "m"
                coef = 1 / 60
            elif time > 3600 and time < 86400:
                unite = "h"
                coef = 1 / 3600
            else:
                unite = "j"
                coef = 1 / 86400
        elif time < 1 and time > 1e-3:
            coef = 1e3
            unite = "ms"
        elif time < 1e-3:
            coef = 1e6
            unite = "µs"
        else:
            unite = "s"
            coef = 1.0

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

        tf = np.abs(self.__start - time.time())

        tfCoef, unite = Tic.Get_time_unity(tf)

        textWithTime = f"{text} ({tfCoef:.3f} {unite})"

        value = (text, tf)

        if category in Tic.__History:
            old = list(Tic.__History[category])
            old.append(value)
            Tic.__History[category] = old
        else:
            Tic.__History[category] = [value]

        self.__start = time.time()

        if verbosity:
            print(textWithTime)

        return tf

    @staticmethod
    def Clear() -> None:
        """Deletes history."""
        Tic.__History = {}

    __History: dict[str, list[tuple[str, float]]] = {}
    """history = { category: list( [text, time] ) }"""

    @staticmethod
    def nTic() -> int:
        return len(Tic.__History)

    @staticmethod
    def Resume(verbosity=True) -> str:
        """Returns the TicTac summary"""

        if Tic.__History == {}:
            return ""

        resume = ""

        for categorie in Tic.__History:
            histCategory = np.array(
                np.array(Tic.__History[categorie])[:, 1], dtype=np.float64
            )
            timesCategory = np.sum(histCategory).astype(float)
            timesCategory, unite = Tic.Get_time_unity(timesCategory)
            resumeCategory = f"{categorie} : {timesCategory:.3f} {unite}"
            if verbosity:
                print(resumeCategory)
            resume += "\n" + resumeCategory

        return resume

    @staticmethod
    @requires_matplotlib
    def __plotBar(
        ax: plt.Axes,
        categories: list[str],
        times: list[float],
        reps: list[int],
        title: str,
    ) -> None:
        # Axis parameters
        ax.xaxis.set_tick_params(labelbottom=False, labeltop=True, length=0)
        ax.yaxis.set_visible(False)
        ax.set_axisbelow(True)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)

        ax.grid(axis="x", lw=1.2)

        timeMax = np.max(times)
        Ncategory = len(categories)

        # I want to display the text on the right if the time represents < 0.5 timeTotal
        # Otherwise, we'll display it on the left

        for i, (category, time, rep) in enumerate(
            zip(categories, times, reps)
        ):  # noqa: F402
            # height=0.55
            # ax.barh(i, t, height=height, align="center", label=c)
            y_pos = Ncategory - 1 - i
            ax.barh(y_pos, time, align="center", label=category)

            # We add a space at the end of the text
            space = " "

            unitTime, unit = Tic.Get_time_unity(time / rep)

            if rep > 1:
                repTemps = f" ({rep} x {np.round(unitTime, 2)} {unit})"
            else:
                repTemps = f" ({np.round(unitTime, 2)} {unit})"

            category = space + category + repTemps + space

            if time / timeMax < 0.6:
                ax.text(
                    time,
                    y_pos,
                    category,
                    color="black",
                    verticalalignment="center",
                    horizontalalignment="left",
                )
            else:
                ax.text(
                    time,
                    y_pos,
                    category,
                    color="white",
                    verticalalignment="center",
                    horizontalalignment="right",
                )

        # plt.legend()
        ax.set_title(title)

    @requires_matplotlib
    @staticmethod
    def Plot_History(folder="", details=False) -> None:
        """Plots history.

        Parameters
        ----------
        folder : str, optional
            save folder, by default ""
        details : bool, optional
            History details, by default True
        """

        from EasyFEA import Display

        if Tic.__History == {}:
            return

        history = Tic.__History

        # Calculate total time per category
        categories = np.array(list(history.keys()))
        timesPerCategory = np.array(
            [
                np.sum(np.array(history[category])[:, 1].astype(np.float64))
                for category in categories
            ]
        )

        # Sort categories by descending time
        sorted_indices = np.argsort(timesPerCategory)[::-1]
        categories = categories[sorted_indices]
        timesPerCategory = timesPerCategory[sorted_indices]

        totalTime = []
        for i, c in enumerate(categories):
            # Extract data
            data = np.array(history[c])
            subCategories = data[:, 0].astype(str)
            timeSubCategory = data[:, 1].astype(np.float64)

            # Calculate time and repetitions per subcategory
            unique_subcats, indices = np.unique(subCategories, return_inverse=True)
            time_by_subcat = np.zeros(len(unique_subcats))
            rep_by_subcat = np.zeros(len(unique_subcats), dtype=int)

            for s in np.arange(unique_subcats.size):
                mask = indices == s
                time_by_subcat[s] = np.sum(timeSubCategory[mask])
                rep_by_subcat[s] = np.sum(mask)

            totalTime.append(np.sum(timeSubCategory))

            # Plot subcategories if needed
            if len(unique_subcats) > 1 and details and totalTime[-1] > 0:
                # Sort subcategories by descending time
                sorted_subcat_indices = np.argsort(time_by_subcat)[::-1]
                ax = plt.subplots()[1]
                Tic.__plotBar(
                    ax,
                    unique_subcats[sorted_subcat_indices],
                    time_by_subcat[sorted_subcat_indices],
                    rep_by_subcat[sorted_subcat_indices],
                    c,
                )
                if folder != "":
                    Display.Save_fig(folder, f"TicTac{i}_{c}")

        # Plot summary of categories
        ax = plt.subplots()[1]
        Tic.__plotBar(
            ax, categories, timesPerCategory, [1] * len(categories), "Summary"
        )
        if folder != "":
            Display.Save_fig(folder, "TicTac_Summary")
