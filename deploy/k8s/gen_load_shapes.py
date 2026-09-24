"""Generate pgfplots source for all four Locust load shapes."""

import argparse
import pathlib

from load_schedule import SHAPES


STAGES = {
    name: [(stage.duration_s, stage.users, stage.spawn_rate) for stage in stages]
    for name, stages in SHAPES.items()
}

TITLES = {
    "phased": "phased",
    "oscillating": "oscillating",
    "staircase": "staircase",
    "flash_crowd": "flash crowd",
}


def compute_trajectory(stages, dt=1):
    points = [(0, 0)]
    t = 0
    current = 0
    for duration, target, rate in stages:
        ramp_time = abs(target - current) / rate if rate > 0 else 0
        ramp_time = min(ramp_time, duration)
        ramp_end = t + ramp_time
        while t < ramp_end:
            t += dt
            if t > ramp_end:
                t = ramp_end
            frac = (t - (ramp_end - ramp_time)) / ramp_time if ramp_time > 0 else 1
            u = current + frac * (target - current)
            points.append((t, u))
        current = target
        hold_end = t + (duration - ramp_time)
        if hold_end > t:
            t = hold_end
            points.append((t, current))
    return points


def coords_str(points):
    return " ".join(f"({t:.0f},{u:.0f})" for t, u in points)


def generate():
    shapes = list(STAGES.keys())
    trajectories = {name: compute_trajectory(stages) for name, stages in STAGES.items()}

    xmax = {name: trajectories[name][-1][0] for name in shapes}

    tex = r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{groupplot}[
    group style={
        group size=2 by 2,
        horizontal sep=1.2cm,
        vertical sep=0.9cm,
    },
    width=0.46\columnwidth,
    height=3.2cm,
    xmin=0,
    ymin=0,
    ymax=220,
    xlabel={time (s)},
    ylabel={users},
    xtick distance=60,
    ytick distance=50,
    tick label style={font=\scriptsize},
    label style={font=\small},
    title style={font=\small, yshift=-2pt},
    no markers,
    thick,
    every axis plot/.style={blue},
    grid=major,
    grid style={gray!20},
]
"""
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (row, col), name in zip(positions, shapes):
        tr = trajectories[name]
        title = TITLES[name]
        xm = xmax[name]
        tex += rf"""
\nextgroupplot[xmax={xm:.0f}, title={{{title}}}]
\addplot coordinates {{
    {coords_str(tr)}
}};
"""

    tex += r"""
\end{groupplot}
\end{tikzpicture}
\caption{Locust load shapes used in the HPA autoscaling experiment.}
\label{fig:load-shapes}
\end{figure}
"""
    return tex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("load_shapes.tex"),
        help="Output .tex file (default: ./load_shapes.tex)",
    )
    args = parser.parse_args()
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(generate())
    print(f"Wrote {out}")
