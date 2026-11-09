import numpy as np
import pytest

from lanchester import (rhs_square, rhs_linear, simulate, exact_square,
                        max_error, main)

A, B = 0.05, 0.06


def test_square_invariant():
    # при dt = 0.001 за время 20 инвариант a*x^2 - b*y^2 меняется на доли
    # процента (у метода Эйлера он убывает как (1 - a*b*dt^2)^n), берём 1e-3
    t, s = simulate(rhs_square, 100, 80, A, B, 0.001, 20)
    inv = A * s[:, 0] ** 2 - B * s[:, 1] ** 2
    assert np.abs(inv - inv[0]).max() < 1e-3 * abs(inv[0])


def test_linear_invariant():
    # для линейного закона Эйлер сохраняет a*x - b*y до ошибок округления
    t, s = simulate(rhs_linear, 100, 80, A, B, 0.01, 30)
    inv = A * s[:, 0] - B * s[:, 1]
    assert np.abs(inv - inv[0]).max() < 1e-9


def test_close_to_exact():
    # при малом шаге численное решение почти совпадает с точным
    t, s = simulate(rhs_square, 100, 80, A, B, 0.001, 20)
    ex, ey = exact_square(t, 100, 80, A, B)
    assert np.abs(s[:, 0] - ex).max() < 0.1
    assert np.abs(s[:, 1] - ey).max() < 0.1


def test_order_of_method():
    # ошибка пропорциональна dt: наклон в логарифмических осях около 1
    dts = [0.2, 0.1, 0.05, 0.025]
    errors = [max_error(dt, 100, 80, A, B) for dt in dts]
    slope = np.polyfit(np.log(dts), np.log(errors), 1)[0]
    assert 0.8 < slope < 1.2


def test_winner():
    # при a*x0^2 > b*y0^2 побеждает X, при обратном Y
    t, s = simulate(rhs_square, 100, 80, A, B, 0.1, 60)
    assert s[-1, 1] == 0 and s[-1, 0] > 0
    t, s = simulate(rhs_square, 80, 100, A, B, 0.1, 60)
    assert s[-1, 0] == 0 and s[-1, 1] > 0
    # равные силы: оба убывают одинаково, ни один не побеждает
    t, s = simulate(rhs_square, 100, 100, 0.05, 0.05, 0.1, 60)
    assert abs(s[-1, 0] - s[-1, 1]) < 1e-6 * s[0, 0]
    assert s[-1].min() > 0


@pytest.mark.parametrize("rhs", [rhs_square, rhs_linear])
def test_non_negative(rhs):
    # даже при крупном шаге численность не уходит ниже нуля,
    # а расчёт останавливается, когда одна сторона равна нулю
    t, s = simulate(rhs, 100, 80, A, B, 1.0, 60)
    assert s.min() >= 0
    if s[-1].min() == 0:
        assert len(t) < 61


def test_bad_parameters():
    with pytest.raises(SystemExit):
        main(["--dt", "-1"])
    with pytest.raises(SystemExit):
        main(["--a", "0"])
