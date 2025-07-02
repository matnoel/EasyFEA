# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""Module containing the Tic class for timing tasks (code profiling)."""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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

        # I want to display the text on the right if the time represents < 0.5 timeTotal
        # Otherwise, we'll display it on the left

        for i, (category, time, rep) in enumerate(
            zip(categories, times, reps)
        ):  # noqa: F402
            # height=0.55
            # ax.barh(i, t, height=height, align="center", label=c)
            ax.barh(i, time, align="center", label=category)

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
                    i,
                    category,
                    color="black",
                    verticalalignment="center",
                    horizontalalignment="left",
                )
            else:
                ax.text(
                    time,
                    i,
                    category,
                    color="white",
                    verticalalignment="center",
                    horizontalalignment="right",
                )

        # plt.legend()
        ax.set_title(title)

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

        historique = Tic.__History
        totalTime = []
        categories: list[str] = list(historique.keys())

        # recovers the time for each category
        tempsCategorie = [
            np.sum(np.array(np.array(historique[c])[:, 1], dtype=np.float64))
            for c in categories
        ]

        categories = np.asarray(categories)[np.argsort(tempsCategorie)][::-1].tolist()

        for i, c in enumerate(categories):
            # c subcategory times
            timeSubCategory = np.array(np.array(historique[c])[:, 1], dtype=np.float64)
            totalTime.append(
                np.sum(timeSubCategory)
            )  # somme tout les temps de cette catégorie

            subCategories = np.array(np.array(historique[c])[:, 0], dtype=str).tolist()

            # We build a table to sum them over the sub-categories
            dfSubCategory = pd.DataFrame(
                {"sub-categories": subCategories, "time": timeSubCategory, "rep": 1}
            )
            dfSubCategory = dfSubCategory.groupby(["sub-categories"]).sum()
            dfSubCategory = dfSubCategory.sort_values(by="time")

            # print(dfSousCategorie)

            if len(subCategories) > 1 and details and totalTime[-1] > 0:
                fig, ax = plt.subplots()
                Tic.__plotBar(
                    ax,
                    subCategories,
                    dfSubCategory["time"].tolist(),
                    dfSubCategory["rep"].tolist(),
                    c,
                )

                if folder != "":
                    Display.Save_fig(folder, f"TicTac{i}_{c}")

        # We build a table to sum them over the sub-categories
        dfCategory = pd.DataFrame({"categories": categories, "time": totalTime})
        dfCategory = dfCategory.groupby(["categories"]).sum()
        dfCategory = dfCategory.sort_values(by="time")
        categories = dfCategory.index.tolist()

        fig, ax = plt.subplots()
        Tic.__plotBar(
            ax, categories, dfCategory["time"], [1] * dfCategory.shape[0], "Summary"
        )

        if folder != "":
            Display.Save_fig(folder, "TicTac_Summary")
