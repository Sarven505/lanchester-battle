"""
Лабораторная работа: уравнения Ланчестера.

Бой двух группировок X и Y считаем явным методом Эйлера для двух законов:
закона квадратов (хорошо организованный бой) и линейного закона
(плохо организованный бой). Закон квадратов сравниваем с точным решением.

Запуск:  python lanchester.py            (все параметры - по умолчанию)
         python lanchester.py --gif      (ещё и анимация images/animation.gif)
         python lanchester.py -h         (справка)
Нужны библиотеки numpy, matplotlib и pillow (pip install -r requirements.txt).
Графики сохраняются в папку results.
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

# ---------- параметры расчёта ----------
FRAMES = 100                                 # число кадров в анимации
DTS = [1 / 2 ** k for k in range(7)]         # шаги для сходимости: 1 ... 1/64
GRID = np.linspace(10, 200, 40)              # сетка начальных численностей
# Состояние боя - пара (x, y). Для сетки начальных условий x и y - массивы.

NAMES = {"square": "закон квадратов", "linear": "линейный закон"}


def rhs_square(s, a, b):
    """Правая часть закона квадратов: dx/dt = -b*y, dy/dt = -a*x."""
    x, y = s
    return np.array([-b * y, -a * x])


def rhs_linear(s, a, b):
    """Правая часть линейного закона: dx/dt = -b*x*y, dy/dt = -a*x*y."""
    x, y = s
    return np.array([-b * x * y, -a * x * y])


def margin_square(x, y, a, b):
    """Инвариант закона квадратов a*x^2 - b*y^2 (больше нуля - сильнее X)."""
    return a * x ** 2 - b * y ** 2


def margin_linear(x, y, a, b):
    """Инвариант линейного закона a*x - b*y (больше нуля - сильнее X)."""
    return a * x - b * y


def simulate(rhs, x0, y0, a, b, dt, tmax):
    """Явный метод Эйлера. Возвращает времена и массив численностей (x, y).
    Численность не бывает отрицательной. Расчёт кончается, когда одна из
    сторон стала равна нулю или вышло время."""
    n = int(round(tmax / dt))
    s = np.zeros((n + 1, 2))
    s[0] = [x0, y0]
    last = n
    for i in range(n):
        s[i + 1] = np.maximum(s[i] + dt * rhs(s[i], a, b), 0.0)
        if s[i + 1].min() == 0.0:
            last = i + 1
            break
    t = dt * np.arange(last + 1)
    return t, s[:last + 1]


def exact_square(t, x0, y0, a, b):
    """Точное решение закона квадратов."""
    k = np.sqrt(a * b)
    ch, sh = np.cosh(k * t), np.sinh(k * t)
    return x0 * ch - y0 * np.sqrt(b / a) * sh, y0 * ch - x0 * np.sqrt(a / b) * sh


def exact_end_time(x0, y0, a, b):
    """Время гибели проигравшего по точному решению (при ничьей - бесконечность)."""
    if a * x0 ** 2 > b * y0 ** 2:
        r = y0 * np.sqrt(b / a) / x0
    else:
        r = x0 * np.sqrt(a / b) / y0
    if r >= 1:
        return np.inf
    return np.arctanh(r) / np.sqrt(a * b)


def exact_remainder(law, x0, y0, a, b):
    """Остаток победителя по формулам: (победитель, остаток)."""
    if law == "square":
        m = margin_square(x0, y0, a, b)
        return ("X", np.sqrt(x0 ** 2 - b / a * y0 ** 2)) if m > 0 else \
               ("Y", np.sqrt(y0 ** 2 - a / b * x0 ** 2))
    m = margin_linear(x0, y0, a, b)
    return ("X", x0 - b / a * y0) if m > 0 else ("Y", y0 - a / b * x0)


def end_time(t, s, level):
    """Первое время, когда одна из сторон не больше level (или None)."""
    idx = np.where(s.min(axis=1) <= level)[0]
    return t[idx[0]] if len(idx) else None


def max_error(dt, x0, y0, a, b):
    """Максимальная ошибка x(t) против точного решения на интервале,
    где обе стороны ещё живы (до половины времени гибели проигравшего)."""
    t_end = 0.5 * exact_end_time(x0, y0, a, b)
    t, s = simulate(rhs_square, x0, y0, a, b, dt, t_end)
    return np.abs(s[:, 0] - exact_square(t, x0, y0, a, b)[0]).max()


def convergence(x0, y0, a, b):
    """Ошибки для набора шагов и наклон прямой в логарифмических осях."""
    errors = [max_error(dt, x0, y0, a, b) for dt in DTS]
    slope = np.polyfit(np.log(DTS), np.log(errors), 1)[0]
    return np.array(errors), slope


def victory_map(a, b, dt, tmax):
    """Карта победы закона квадратов: метод Эйлера сразу для всей сетки
    начальных условий. Возвращает относительный остаток победителя
    (плюс - победил X, минус - победил Y) и признак победы X."""
    xx, yy = np.meshgrid(GRID, GRID)          # строки - y0, столбцы - x0
    s = np.array([xx, yy])
    for _ in range(int(round(tmax / dt))):
        s = np.maximum(s + dt * rhs_square(s, a, b), 0.0)
    x, y = s
    x_wins = margin_square(x, y, a, b) > 0
    rel = np.where(x_wins, x / xx, -y / yy)
    return rel, x_wins


def find_boundary(margin, rhs, a, b, dt, tmax, x0=100.0):
    """Граничное y0/x0 методом деления пополам: при меньшем y0 побеждает X."""
    lo, hi = 0.1 * x0, 3.0 * x0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        t, s = simulate(rhs, x0, mid, a, b, dt, tmax)
        if margin(s[-1, 0], s[-1, 1], a, b) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / x0


def make_gif(runs, x0, y0):
    """Анимация боя по закону квадратов: сверху поле боя (каждая точка -
    один боец), снизу численность сторон во времени."""
    os.makedirs("images", exist_ok=True)
    law, t, s = runs[0]
    rng = np.random.default_rng(1)
    bg = "#1b1f2a"

    fig, (field, plot) = plt.subplots(
        2, 1, figsize=(6, 5.5), gridspec_kw={"height_ratios": [3, 2]},
        facecolor=bg)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.92, bottom=0.1,
                        hspace=0.35)

    # начальные места бойцов: X слева, Y справа, сетка 10 столбцов
    def army(n, x_start, sign):
        k = np.arange(int(n))
        col, row = k % 10, k // 10
        px = x_start + sign * col * 0.35
        py = 1 + row * 0.55 - (len(k) // 10) * 0.27
        return np.column_stack([px, py])

    base_x = army(x0, 0.5, 1)
    base_y = army(y0, 9.5, -1)
    # в каком порядке бойцы выбывают
    order_x = rng.permutation(len(base_x))
    order_y = rng.permutation(len(base_y))

    field.set_facecolor(bg)
    field.set_xlim(0, 10)
    field.set_ylim(-1.5, 6.5)
    field.axis("off")
    dots_x = field.scatter([], [], s=22, color="#4aa3ff")
    dots_y = field.scatter([], [], s=22, color="#ff5a5a")
    shots = LineCollection([], colors="#ffd84a", linewidths=0.6, alpha=0.8)
    field.add_collection(shots)

    plot.set_facecolor(bg)
    plot.set_xlim(0, t[-1])
    plot.set_ylim(0, 1.05 * max(x0, y0))
    plot.set_xlabel("t", color="white")
    plot.set_ylabel("численность", color="white")
    plot.tick_params(colors="white")
    for side in plot.spines.values():
        side.set_color("gray")
    line_x, = plot.plot([], [], color="#4aa3ff", label="X")
    line_y, = plot.plot([], [], color="#ff5a5a", label="Y")
    plot.legend(loc="upper right", facecolor=bg, labelcolor="white")

    idx = np.linspace(0, len(t) - 1, FRAMES).round().astype(int)

    def update(i):
        j = idx[i]
        nx, ny = int(round(s[j, 0])), int(round(s[j, 1]))
        # живые бойцы: те, чей номер в порядке выбывания меньше числа живых
        alive_x = order_x < nx
        alive_y = order_y < ny
        # стороны понемногу сходятся, бойцы чуть дрожат
        move = 1.2 * j / (len(t) - 1)
        px = base_x[alive_x] + [move, 0] + rng.normal(0, 0.04, (nx, 2))
        py = base_y[alive_y] - [move, 0] + rng.normal(0, 0.04, (ny, 2))
        dots_x.set_offsets(px)
        dots_y.set_offsets(py)
        # выстрелы: несколько случайных пар
        segs = []
        if nx > 0 and ny > 0 and i < FRAMES - 1:
            for _ in range(8):
                segs.append([px[rng.integers(nx)], py[rng.integers(ny)]])
        shots.set_segments(segs)
        field.set_title(f"t = {t[j]:.1f}     X: {nx}     Y: {ny}",
                        color="white", fontsize=12)
        line_x.set_data(t[:j + 1], s[:j + 1, 0])
        line_y.set_data(t[:j + 1], s[:j + 1, 1])

    anim = FuncAnimation(fig, update, frames=FRAMES)
    anim.save("images/animation.gif", writer=PillowWriter(fps=15), dpi=70)
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Уравнения Ланчестера: закон квадратов и линейный закон, "
                    "метод Эйлера.",
        add_help=False)
    parser.add_argument("-h", "--help", action="help",
                        help="показать справку и выйти")
    parser.add_argument("--x0", type=float, default=100,
                        help="начальная численность X (100)")
    parser.add_argument("--y0", type=float, default=80,
                        help="начальная численность Y (80)")
    parser.add_argument("--a", type=float, default=0.05,
                        help="эффективность бойца X (0.05)")
    parser.add_argument("--b", type=float, default=0.06,
                        help="эффективность бойца Y (0.06)")
    parser.add_argument("--dt", type=float, default=0.1,
                        help="шаг по времени (0.1)")
    parser.add_argument("--tmax", type=float, default=60,
                        help="максимальное время расчёта (60)")
    parser.add_argument("--gif", action="store_true",
                        help="сделать анимацию images/animation.gif")
    args = parser.parse_args(argv)

    x0, y0, a, b, dt, tmax = (args.x0, args.y0, args.a, args.b,
                              args.dt, args.tmax)
    for name, value in [("x0", x0), ("y0", y0), ("a", a), ("b", b),
                        ("dt", dt), ("tmax", tmax)]:
        if not value > 0:
            parser.error(f"параметр {name} должен быть положительным")
    if dt > tmax:
        parser.error("шаг dt не должен быть больше tmax")
    if a * b * dt ** 2 >= 1:
        parser.error("шаг dt слишком велик для таких a и b")
    if tmax / dt > 1e6:
        parser.error("слишком много шагов: увеличьте dt или уменьшите tmax")

    os.makedirs("results", exist_ok=True)
    laws = {"square": (rhs_square, margin_square, 0.0),
            "linear": (rhs_linear, margin_linear, 1.0)}
    runs = []
    res = {}
    print(f"x0 = {x0:g}, y0 = {y0:g}, a = {a:g}, b = {b:g}, "
          f"dt = {dt:g}, tmax = {tmax:g}")
    for law, (rhs, margin, level) in laws.items():
        t, s = simulate(rhs, x0, y0, a, b, dt, tmax)
        runs.append((law, t, s))
        side, rest = exact_remainder(law, x0, y0, a, b)
        winner = "X" if margin(s[-1, 0], s[-1, 1], a, b) > 0 else "Y"
        rest_num = s[-1, 0] if winner == "X" else s[-1, 1]
        res[law] = (t, s)
        print(f"\n{NAMES[law].capitalize()}: побеждает {winner} "
              f"(по формулам: {side})")
        print(f"  остаток победителя: численно {rest_num:.2f}, "
              f"аналитически {rest:.2f}")
        te = end_time(t, s, level)
        what = "одна из сторон уничтожена" if level == 0 else \
            "проигравший меньше 1 бойца"
        if te is None:
            print(f"  конец боя ({what}): за время tmax не наступил")
        elif law == "square":
            print(f"  конец боя ({what}): t = {te:.1f} "
                  f"(по формуле {exact_end_time(x0, y0, a, b):.1f})")
        else:
            print(f"  конец боя ({what}): t = {te:.1f}")

    # ---------- график 1: численность во времени ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, law in zip(axes, laws):
        t, s = res[law]
        ax.plot(t, s[:, 0], color="tab:blue", label="X, Эйлер")
        ax.plot(t, s[:, 1], color="tab:red", label="Y, Эйлер")
        if law == "square":
            ex, ey = exact_square(t, x0, y0, a, b)
            ax.plot(t, np.maximum(ex, 0), "--", color="navy",
                    label="X, точное")
            ax.plot(t, np.maximum(ey, 0), "--", color="darkred",
                    label="Y, точное")
        ax.set_title(NAMES[law])
        ax.set_xlabel("t")
        ax.set_ylabel("численность")
        ax.grid(alpha=0.3)
        ax.legend()
    plt.savefig("results/battle.png", dpi=150)
    plt.close()

    # ---------- график 2: фазовая плоскость закона квадратов ----------
    plt.figure(figsize=(6, 5.5))
    starts = [(x0, y0), (150, 100), (60, 120), (120, 60), (40, 40),
              (180, 190), (30, 100)]
    for xs, ys in starts:
        _, s = simulate(rhs_square, xs, ys, a, b, dt, 200)
        plt.plot(s[:, 0], s[:, 1], color="tab:blue", linewidth=1)
        plt.plot(xs, ys, "o", color="tab:blue", markersize=4)
    xl = np.linspace(0, 200, 2)
    plt.plot(xl, xl * np.sqrt(a / b), "k--", label="a·x² = b·y²")
    plt.xlim(0, 200)
    plt.ylim(0, 200)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Фазовая плоскость, закон квадратов")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/phase.png", dpi=150)
    plt.close()

    # ---------- график 3: сходимость ----------
    errors, slope = convergence(x0, y0, a, b)
    print("\nСходимость (закон квадратов): шаг dt - максимальная ошибка x(t)")
    for d, e in zip(DTS, errors):
        print(f"  {d:.5f}  {e:.4f}")
    print(f"Наклон в логарифмических осях: {slope:.2f}")
    plt.figure(figsize=(6, 4.5))
    plt.loglog(DTS, errors, "o-", label=f"Эйлер, наклон {slope:.2f}")
    plt.loglog(DTS, errors[0] * np.array(DTS) / DTS[0], "k--",
               label="порядок 1")
    plt.xlabel("dt")
    plt.ylabel("max |x_числ − x_точн|")
    plt.title("Сходимость метода Эйлера")
    plt.legend()
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig("results/convergence.png", dpi=150)
    plt.close()

    # ---------- график 4: карта победы ----------
    rel, x_wins = victory_map(a, b, dt, tmax)
    xx, yy = np.meshgrid(GRID, GRID)
    exact_wins = margin_square(xx, yy, a, b) > 0
    share = (x_wins == exact_wins).mean()
    print(f"\nКарта победы {len(GRID)}×{len(GRID)}: победитель совпал с "
          f"аналитическим в {100 * share:.1f}% клеток")
    plt.figure(figsize=(6.5, 5.5))
    step = GRID[1] - GRID[0]
    ext = (GRID[0] - step / 2, GRID[-1] + step / 2) * 2
    im = plt.imshow(rel, origin="lower", extent=ext, cmap="coolwarm_r",
                    vmin=-1, vmax=1, aspect="auto")
    xl = np.array([GRID[0], GRID[-1]])
    plt.plot(xl, xl * np.sqrt(a / b), "k--", label="y0 = x0·√(a/b)")
    plt.xlim(ext[0], ext[1])
    plt.ylim(ext[2], ext[3])
    plt.colorbar(im, label="остаток победителя / его начальная численность\n"
                           "(+ побеждает X, − побеждает Y)")
    plt.xlabel("x0")
    plt.ylabel("y0")
    plt.title("Карта победы, закон квадратов")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("results/victory_map.png", dpi=150)
    plt.close()

    # ---------- граница победы ----------
    print("\nГраница победы (отношение y0/x0 при x0 = 100)")
    bs = find_boundary(margin_square, rhs_square, a, b, dt, tmax)
    print(f"  закон квадратов: численно {bs:.4f}, по формуле "
          f"√(a/b) = {np.sqrt(a / b):.4f}")
    bl = find_boundary(margin_linear, rhs_linear, a, b, dt, tmax)
    print(f"  линейный закон: численно {bl:.4f}, по формуле "
          f"a/b = {a / b:.4f}")
    print("\nГрафики сохранены в папку results")

    if args.gif:
        make_gif(runs, x0, y0)
        print("Анимация сохранена: images/animation.gif")


if __name__ == "__main__":
    main()
