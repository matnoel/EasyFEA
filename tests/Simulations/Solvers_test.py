# Copyright (C) 2021-2024 Université Gustave Eiffel.
# Copyright (C) 2025-2026 Université Gustave Eiffel, INRIA.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

from EasyFEA import AlgoType


def test_AlgoType_prints_its_name():
    assert [str(algo) for algo in AlgoType] == [algo.value for algo in AlgoType]


def test_AlgoType_Get_Hyperbolic_Types_lists_every_second_order_scheme():
    hyperbolic = AlgoType.Get_Hyperbolic_Types()

    assert set(hyperbolic) == set(AlgoType) - {AlgoType.elliptic, AlgoType.parabolic}


def test_AlgoType_Get_Hyperbolic_and_Parabolic_Types_adds_parabolic():
    timeDependent = AlgoType.Get_Hyperbolic_and_Parabolic_Types()

    assert set(timeDependent) == set(AlgoType) - {AlgoType.elliptic}
