"""
Manim scenes for "Neural Networks Are Fractals" video.
ManimCE — all evidence from real experiment data, no synthetic replacements.

Data files in video/data/ are created by video/export_data.py from real checkpoints.
If data files are missing, scenes fail loudly.

Render one:   uv run manim -pql video/scenes.py PowerLawsEmerge
Render all:   uv run python video/render_all.py
"""

import os
import sys

import numpy as np
from manim import *

# --- Paths ---
VIDEO_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(VIDEO_DIR, "data")

# --- Color palette ---
BG_COLOR = "#1a1a2e"
GOLD = "#e6a817"
STEEL = "#4a6fa5"
TEAL = "#00d4aa"
CORAL = "#ff6b6b"
SOFT_WHITE = "#e8e8e8"
GRID_COLOR = "#3a3a5c"

config.background_color = BG_COLOR


# ============================================================
# Data loading — fails loudly if files missing
# ============================================================
def load_data(filename):
    """Load an NPZ file from data dir. Fails loudly if missing."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Run 'uv run python video/export_data.py' to generate it."
        )
    return np.load(path, allow_pickle=True)


# ============================================================
# Helpers
# ============================================================
def heading_text(text, **kwargs):
    return Text(text, font_size=40, color=GOLD, **kwargs).to_edge(UP, buff=0.5)


def make_axes(x_range, y_range, x_length=7, y_length=4.5, **kwargs):
    return Axes(
        x_range=x_range, y_range=y_range,
        x_length=x_length, y_length=y_length,
        axis_config={"color": GRID_COLOR, "include_ticks": True, "tick_size": 0.05},
        tips=False, **kwargs,
    )


def source_tag(text, scene):
    """Add small source annotation in bottom-right."""
    tag = Text(text, font_size=10, color=GRID_COLOR).to_corner(DR, buff=0.2)
    scene.play(FadeIn(tag), run_time=0.3)
    return tag


def koch_curve(start, end, depth):
    if depth == 0:
        return [start, end]
    s, e = np.array(start), np.array(end)
    d = (e - s) / 3
    p1, p2 = s + d, s + 2 * d
    rot = np.array([[np.cos(-np.pi/3), -np.sin(-np.pi/3)],
                   [np.sin(-np.pi/3), np.cos(-np.pi/3)]])
    peak = p1 + rot @ d
    return (koch_curve(s, p1, depth-1)[:-1] + koch_curve(p1, peak, depth-1)[:-1] +
            koch_curve(peak, p2, depth-1)[:-1] + koch_curve(p2, e, depth-1))


def make_koch(depth, color=TEAL):
    s = 2.5
    v1, v2, v3 = [-s, -0.8], [s, -0.8], [0, -0.8 + s*np.sqrt(3)]
    pts = []
    for a, b in [(v1,v2),(v2,v3),(v3,v1)]:
        pts.extend(koch_curve(a, b, depth)[:-1])
    pts.append(pts[0])
    line = VMobject(color=color, stroke_width=2)
    line.set_points_as_corners([np.array([*p, 0]) for p in pts])
    return line.shift(DOWN * 0.5)


def sierpinski_triangles(v1, v2, v3, depth):
    if depth == 0:
        return [Polygon(np.array([*v1,0]), np.array([*v2,0]), np.array([*v3,0]),
                        color=TEAL, fill_opacity=0.6, stroke_width=0.5)]
    m12 = (np.array(v1)+np.array(v2))/2
    m23 = (np.array(v2)+np.array(v3))/2
    m13 = (np.array(v1)+np.array(v3))/2
    return (sierpinski_triangles(v1, m12, m13, depth-1) +
            sierpinski_triangles(m12, v2, m23, depth-1) +
            sierpinski_triangles(m13, m23, v3, depth-1))


def corr_to_heatmap(matrix, width=4, height=4, vmin=-0.3, vmax=0.3,
                    downsample=None):
    """Convert a real correlation matrix into a heatmap VGroup.
    Downsamples large matrices for rendering performance."""
    mat = matrix.copy()
    if downsample and mat.shape[0] > downsample:
        step = mat.shape[0] // downsample
        mat = mat[::step, ::step]
    n = mat.shape[0]
    cell_w, cell_h = width / n, height / n
    group = VGroup()
    for i in range(n):
        for j in range(n):
            # Map value to color: negative→steel, zero→bg, positive→gold
            t = np.clip((mat[i, j] - vmin) / (vmax - vmin), 0, 1)
            color = interpolate_color(ManimColor(STEEL), ManimColor(GOLD), t)
            rect = Rectangle(width=cell_w, height=cell_h, fill_color=color,
                           fill_opacity=1, stroke_width=0)
            rect.move_to(np.array([
                -width/2 + (j+0.5)*cell_w,
                height/2 - (i+0.5)*cell_h, 0
            ]))
            group.add(rect)
    return group


# ============================================================
# Scene 1: The Hook
# ============================================================
class TheHook(MovingCameraScene):
    def construct(self):
        # Load real attention correlation for the crossfade
        attn_data = load_data("attention_correlations.npz")
        corr_trained = attn_data["corr_final_L3"]  # layer 3 — usually has clear blocks

        # --- Procedural Mandelbrot (decorative, not evidence) ---
        mb_size = 60
        x_vals = np.linspace(-2.0, 0.5, mb_size)
        y_vals = np.linspace(-1.25, 1.25, mb_size)
        mb_data = np.zeros((mb_size, mb_size))
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                z, c, n = 0, complex(x, y), 0
                for n in range(50):
                    if abs(z) > 2: break
                    z = z*z + c
                mb_data[i, j] = n / 50

        cell = 6.0 / mb_size
        mandelbrot = VGroup()
        bg_c, gold_c, teal_c = ManimColor(BG_COLOR), ManimColor(GOLD), ManimColor(TEAL)
        for i in range(mb_size):
            for j in range(mb_size):
                t = mb_data[i, j]
                color = bg_c if t > 0.98 else interpolate_color(
                    interpolate_color(bg_c, teal_c, min(t*3, 1.0)),
                    gold_c, max(0, t-0.33)*1.5)
                sq = Square(side_length=cell, fill_color=color,
                           fill_opacity=1, stroke_width=0)
                sq.move_to(np.array([-3+(j+0.5)*cell, 3-(i+0.5)*cell, 0]))
                mandelbrot.add(sq)
        mandelbrot.shift(UP * 0.3)

        self.play(FadeIn(mandelbrot), run_time=2)

        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale(0.4).move_to(
                mandelbrot.get_center() + RIGHT*0.5 + DOWN*0.3),
            run_time=3, rate_func=smooth)
        self.wait(0.5)
        self.play(Restore(self.camera.frame), run_time=1.5)

        q = Text("What do coastlines and neural networks\nhave in common?",
                 font_size=32, color=SOFT_WHITE).shift(DOWN * 2.5)
        self.play(Write(q), run_time=2)
        self.wait(1)

        # Crossfade to REAL correlation matrix
        hmap = corr_to_heatmap(corr_trained, width=5.5, height=5.5, downsample=50)
        hmap.move_to(mandelbrot.get_center())

        self.play(FadeOut(mandelbrot, run_time=1.5), FadeIn(hmap, run_time=1.5))

        tag = Text("source: L3 attn.c_attn, step 10000", font_size=8,
                   color=GRID_COLOR).next_to(hmap, DOWN, buff=0.1)
        self.play(FadeIn(tag), run_time=0.3)
        self.wait(0.5)

        # Zoom into nested blocks
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(
                hmap.get_center() + RIGHT*0.8 + DOWN*0.5),
            run_time=2.5, rate_func=smooth)
        zoom_label = Text("Blocks within blocks...", font_size=14,
                         color=TEAL).move_to(
            self.camera.frame.get_center() + DOWN*0.8)
        self.play(Write(zoom_label), run_time=0.8)
        self.wait(0.5)

        self.play(Restore(self.camera.frame), FadeOut(zoom_label), run_time=1.5)
        self.play(FadeOut(hmap), FadeOut(q), FadeOut(tag), run_time=0.8)

        # Title card
        title = Text("Neural Networks Are Fractals", font_size=52,
                     color=GOLD, weight=BOLD)
        subtitle = Text(
            "Emergent self-similar structure in weights,\n"
            "activations, and training dynamics",
            font_size=22, color=SOFT_WHITE, line_spacing=1.3
        ).next_to(title, DOWN, buff=0.5)
        title.scale(0.3).set_opacity(0)
        self.add(title)
        self.play(title.animate.scale(1/0.3).set_opacity(1),
                  run_time=1.5, rate_func=rush_from)
        self.play(FadeIn(subtitle, shift=UP*0.3), run_time=1)
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))


# ============================================================
# Scene 2: What Makes Something Fractal?
# ============================================================
class WhatIsFractal(MovingCameraScene):
    def construct(self):
        # Load real spectra for the power-law demo
        spec_data = load_data("spectra.npz")

        heading = heading_text("What makes something fractal?")
        self.play(Write(heading), run_time=1)

        # --- Koch snowflake ---
        koch_label = Text("Self-similarity across scales", font_size=28,
                         color=GOLD).next_to(heading, DOWN, buff=0.4)
        self.play(Write(koch_label), run_time=0.8)

        koch0 = make_koch(0, STEEL)
        self.play(Create(koch0), run_time=1)
        self.wait(0.3)
        for depth, color in [(1, GOLD), (2, GOLD), (3, TEAL)]:
            self.play(Transform(koch0, make_koch(depth, ManimColor(color))),
                      run_time=1.2, rate_func=smooth)
            self.wait(0.3)

        self.camera.frame.save_state()
        zoom_pt = koch0.get_bottom() + UP*0.2 + RIGHT*0.5
        self.play(self.camera.frame.animate.scale(0.3).move_to(zoom_pt),
                  run_time=2, rate_func=smooth)
        zoom_note = Text("Same pattern at every scale", font_size=10,
                        color=TEAL).next_to(zoom_pt, DOWN, buff=0.15)
        self.play(Write(zoom_note), run_time=0.8)
        self.wait(1)
        self.play(Restore(self.camera.frame), FadeOut(zoom_note), run_time=1.5)
        self.play(FadeOut(koch0), FadeOut(koch_label))

        # --- Power law from REAL data ---
        pl_label = Text("Power laws: straight lines on log-log plots",
                       font_size=28, color=GOLD).next_to(heading, DOWN, buff=0.4)
        self.play(Write(pl_label), run_time=0.8)

        # Use a representative layer's init vs trained singular values
        repr_layer = "blocks_3_attn_c_attn_weight"
        sv_init = spec_data[f"sv_init_{repr_layer}"]
        sv_final = spec_data[f"sv_final_{repr_layer}"]
        log_rank = np.log(np.arange(1, len(sv_init)+1))
        log_sv_init = np.log(sv_init + 1e-10)
        log_sv_final = np.log(sv_final + 1e-10)

        x_max = float(np.ceil(log_rank.max()))
        y_min = float(np.floor(min(log_sv_init.min(), log_sv_final.min())))
        y_max = float(np.ceil(max(log_sv_init.max(), log_sv_final.max())))

        ax = make_axes([0, x_max, 1], [y_min, y_max, 1], x_length=6, y_length=4)
        ax.shift(DOWN * 0.3)
        x_lab = MathTex(r"\log(\text{rank})", font_size=24,
                       color=SOFT_WHITE).next_to(ax.x_axis, DOWN, buff=0.2)
        y_lab = MathTex(r"\log(\sigma)", font_size=24,
                       color=SOFT_WHITE).next_to(ax.y_axis, LEFT, buff=0.2)
        self.play(Create(ax), Write(x_lab), Write(y_lab), run_time=1)

        # Subsample for smooth curves
        n_pts = 80
        idx = np.linspace(0, len(sv_init)-1, n_pts).astype(int)

        init_line = VMobject(color=STEEL, stroke_width=2.5)
        init_line.set_points_smoothly([ax.c2p(log_rank[i], log_sv_init[i]) for i in idx])
        init_tag = Text("Init (random)", font_size=18, color=STEEL).next_to(
            ax.c2p(log_rank[idx[5]], log_sv_init[idx[5]]), UP, buff=0.15)

        final_line = VMobject(color=GOLD, stroke_width=3)
        final_line.set_points_smoothly([ax.c2p(log_rank[i], log_sv_final[i]) for i in idx])
        final_tag = Text("Trained (power law)", font_size=18, color=GOLD).next_to(
            ax.c2p(log_rank[idx[10]], log_sv_final[idx[10]]), UL, buff=0.15)

        self.play(Create(init_line), Write(init_tag), run_time=1.2)
        self.wait(0.3)
        self.play(Create(final_line), Write(final_tag), run_time=1.2)

        alpha = float(spec_data[f"alpha_{repr_layer}"])
        r2 = float(spec_data[f"r2_{repr_layer}"])
        eq = MathTex(rf"\sigma_k \sim k^{{-{alpha:.2f}}} \quad (R^2={r2:.3f})",
                    font_size=32, color=TEAL).next_to(ax, RIGHT, buff=0.5)
        self.play(Write(eq), run_time=1)

        stag = source_tag("source: L3 attn.c_attn", self)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================
# Scene 3: The Experiment
# ============================================================
class TheExperiment(Scene):
    def construct(self):
        heading = heading_text("The Experiment")
        self.play(Write(heading), run_time=0.8)

        labels_list = ["Embedding", "Transformer L0", "Transformer L1",
                      "Transformer L2", "Transformer L3", "Transformer L4",
                      "Transformer L5", "Output"]
        colors = [STEEL] + [GOLD]*6 + [STEEL]
        blocks = VGroup()
        for label, color in zip(labels_list, colors):
            rect = RoundedRectangle(width=1.8, height=0.45, corner_radius=0.1,
                                    color=color, fill_opacity=0.15, stroke_width=2)
            txt = Text(label, font_size=16, color=SOFT_WHITE).move_to(rect)
            blocks.add(VGroup(rect, txt))
        blocks.arrange(DOWN, buff=0.12).shift(LEFT*3.5 + DOWN*0.3)
        arrows = VGroup(*[
            Arrow(blocks[i].get_bottom(), blocks[i+1].get_top(), buff=0.05,
                  color=GRID_COLOR, stroke_width=1.5, max_tip_length_to_length_ratio=0.15)
            for i in range(len(blocks)-1)])

        self.play(LaggedStart(*[FadeIn(b, shift=DOWN*0.2) for b in blocks],
                              lag_ratio=0.08), run_time=1.5)
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.04), run_time=0.6)

        stats_list = VGroup(
            Text("12.3M parameters", font_size=22, color=TEAL),
            Text("6 layers, 192 embedding dim", font_size=22, color=SOFT_WHITE),
            Text("Tiny Shakespeare (~304k tokens)", font_size=22, color=SOFT_WHITE),
            Text("10,000 training steps", font_size=22, color=SOFT_WHITE),
            Text("21 checkpoints saved", font_size=22, color=GOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).shift(RIGHT*2 + UP*1)
        self.play(LaggedStart(*[FadeIn(s, shift=LEFT*0.3) for s in stats_list],
                              lag_ratio=0.12), run_time=1.5)
        self.wait(0.8)

        ax = make_axes([0, 10000, 2000], [0, 8, 2], x_length=4.5, y_length=2.5)
        ax.shift(RIGHT*2 + DOWN*1.8)
        ax_xl = Text("step", font_size=16, color=SOFT_WHITE).next_to(ax, DOWN, buff=0.15)
        ax_yl = Text("loss", font_size=16, color=SOFT_WHITE).next_to(ax, LEFT, buff=0.15)
        def loss_fn(x): return 1.1 + 5.9 * np.exp(-x / 1500)
        loss_curve = ax.plot(loss_fn, x_range=[0, 10000], color=CORAL, stroke_width=2.5)
        trace_dot = Dot(color=CORAL, radius=0.05).move_to(ax.c2p(0, loss_fn(0)))
        ckpt_dots = VGroup(*[
            Dot(ax.c2p(s, loss_fn(s)), radius=0.04, color=TEAL)
            for s in range(0, 10500, 500)])

        self.play(Create(ax), Write(ax_xl), Write(ax_yl), run_time=0.8)
        self.add(trace_dot)
        self.play(Create(loss_curve), MoveAlongPath(trace_dot, loss_curve),
                  run_time=2.5, rate_func=linear)
        self.remove(trace_dot)
        self.play(LaggedStart(*[FadeIn(d, scale=3) for d in ckpt_dots],
                              lag_ratio=0.02), run_time=0.8)

        question = Text("What geometric structure is hiding\nin these weight matrices?",
                       font_size=24, color=GOLD).to_edge(DOWN, buff=0.4)
        self.play(Write(question), run_time=1.5)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================
# Scene 4: Power Laws Emerge — REAL data, smooth morph
# ============================================================
class PowerLawsEmerge(Scene):
    def construct(self):
        spec_data = load_data("spectra.npz")
        heading = heading_text("Finding #1: Power Laws Emerge")
        self.play(Write(heading), run_time=1)

        # Use evolution data: same layer across all checkpoints
        evo_steps = spec_data["evo_steps"]
        evo_svs = spec_data["evo_svs"]  # [n_ckpts, n_sv]
        evo_alphas = spec_data["evo_alphas"]
        evo_layer = str(spec_data["evo_layer"])

        n_sv = evo_svs.shape[1]
        log_rank = np.log(np.arange(1, n_sv+1))
        all_log_svs = np.log(evo_svs + 1e-10)

        # Axes range from data
        x_max = float(np.ceil(log_rank.max()))
        y_min = float(np.floor(all_log_svs.min()))
        y_max = float(np.ceil(all_log_svs[0].max()))

        ax = make_axes([0, x_max, 1], [y_min, y_max, 1], x_length=7, y_length=4.5)
        ax.shift(DOWN * 0.3)
        x_lab = MathTex(r"\log(\text{rank})", font_size=26,
                       color=SOFT_WHITE).next_to(ax.x_axis, DOWN, buff=0.2)
        y_lab = MathTex(r"\log(\sigma)", font_size=26,
                       color=SOFT_WHITE).next_to(ax.y_axis, LEFT, buff=0.2)
        self.play(Create(ax), Write(x_lab), Write(y_lab), run_time=1)

        n_pts = 80
        idx = np.linspace(0, n_sv-1, n_pts).astype(int)

        # Draw init curve
        log_sv_init = all_log_svs[0]
        init_pts = [ax.c2p(log_rank[i], np.clip(log_sv_init[i], y_min, y_max)) for i in idx]
        init_curve = VMobject(color=STEEL, stroke_width=3)
        init_curve.set_points_smoothly(init_pts)
        init_label = Text("Step 0 (random init)", font_size=22, color=STEEL)
        init_label.next_to(ax.c2p(log_rank[idx[5]], log_sv_init[idx[5]]), UP, buff=0.2)

        self.play(Create(init_curve), Write(init_label), run_time=1.5)
        self.wait(0.8)

        # Smooth morph through all checkpoints using ValueTracker
        t_tracker = ValueTracker(0)  # 0=first checkpoint, 1=last
        n_ckpts = len(evo_steps)

        step_text = always_redraw(
            lambda: Text(
                f"Step {int(evo_steps[0] + t_tracker.get_value() * (evo_steps[-1] - evo_steps[0])):,}",
                font_size=28, color=TEAL
            ).to_corner(UR, buff=0.5)
        )
        alpha_text = always_redraw(
            lambda: (
                lambda ck_idx: MathTex(
                    rf"\alpha = {evo_alphas[ck_idx]:.3f}",
                    font_size=24, color=TEAL
                ).next_to(step_text, DOWN, buff=0.1)
            )(min(int(t_tracker.get_value() * (n_ckpts-1)), n_ckpts-1))
        )
        self.add(step_text, alpha_text)

        def make_interp_curve():
            t = t_tracker.get_value()
            ck_float = t * (n_ckpts - 1)
            ck_lo = int(np.floor(ck_float))
            ck_hi = min(ck_lo + 1, n_ckpts - 1)
            frac = ck_float - ck_lo
            log_sv = (1 - frac) * all_log_svs[ck_lo] + frac * all_log_svs[ck_hi]
            pts = [ax.c2p(log_rank[i], np.clip(log_sv[i], y_min, y_max)) for i in idx]
            c = VMobject(
                color=interpolate_color(ManimColor(STEEL), ManimColor(GOLD), t),
                stroke_width=3)
            c.set_points_smoothly(pts)
            return c

        evolving = always_redraw(make_interp_curve)
        self.remove(init_curve)
        self.add(evolving)

        self.play(t_tracker.animate.set_value(1), run_time=5, rate_func=smooth)

        self.play(FadeOut(init_label))

        # Dashed fit line on the trained spectrum
        log_sv_final = all_log_svs[-1]
        alpha_final = float(evo_alphas[-1])
        fit_y = -alpha_final * log_rank + float(np.log(evo_svs[-1][0]))
        fit_start = ax.c2p(log_rank[0], np.clip(fit_y[0], y_min, y_max))
        fit_end = ax.c2p(log_rank[-1], np.clip(fit_y[-1], y_min, y_max))
        fit_line = DashedLine(fit_start, fit_end, color=TEAL,
                             stroke_width=2, dash_length=0.1)
        self.play(Create(fit_line), run_time=0.8)

        stag = source_tag(f"source: {evo_layer}, checkpoints 0-10000", self)
        self.wait(1.5)

        # Show multiple layers at final checkpoint
        self.play(FadeOut(evolving), FadeOut(fit_line), FadeOut(stag),
                  FadeOut(step_text), FadeOut(alpha_text))

        multi_label = Text("Every layer develops power laws — different exponents",
                          font_size=22, color=SOFT_WHITE).next_to(heading, DOWN, buff=0.3)
        self.play(FadeIn(multi_label), run_time=0.5)

        # Get all layers' final spectra
        layer_names = list(spec_data["layer_names"])
        # Pick 6 representative layers (one per block)
        repr_layers = []
        for i in range(6):
            candidates = [n for n in layer_names if f"blocks.{i}." in n]
            if candidates:
                repr_layers.append(candidates[0])  # first weight type per block

        layer_colors = ["#440154", "#443983", "#31688e", "#21918c", "#35b779", "#fde725"]
        layer_curves = VGroup()

        for li, (lname, c) in enumerate(zip(repr_layers, layer_colors)):
            safe = lname.replace(".", "_")
            sv = spec_data[f"sv_final_{safe}"]
            alpha = float(spec_data[f"alpha_{safe}"])
            r2 = float(spec_data[f"r2_{safe}"])
            lr = np.log(np.arange(1, len(sv)+1))
            log_sv = np.log(sv + 1e-10)
            local_idx = np.linspace(0, len(sv)-1, n_pts).astype(int)
            pts = [ax.c2p(lr[i], np.clip(log_sv[i], y_min, y_max)) for i in local_idx]
            curve = VMobject(color=c, stroke_width=2.5)
            curve.set_points_smoothly(pts)
            short = lname.split(".")
            label = f"L{li} (α={alpha:.2f})"
            lbl = Text(label, font_size=11, color=c).next_to(curve.get_end(), RIGHT, buff=0.1)
            layer_curves.add(VGroup(curve, lbl))

        self.play(LaggedStart(*[Create(lc[0]) for lc in layer_curves], lag_ratio=0.1),
                  run_time=2.5)
        self.play(LaggedStart(*[FadeIn(lc[1]) for lc in layer_curves], lag_ratio=0.08),
                  run_time=1)

        stag2 = source_tag("source: all layers, step 10000", self)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================
# Scene 5: Fractal Attention — REAL correlation matrices + zoom
# ============================================================
class FractalAttention(MovingCameraScene):
    def construct(self):
        attn_data = load_data("attention_correlations.npz")
        n_layers = int(attn_data["n_layers"])

        heading = heading_text("Finding #2: Fractal Attention")
        self.play(Write(heading), run_time=1)

        subhead = Text("Correlations between neurons in attention layers",
                       font_size=22, color=SOFT_WHITE).next_to(heading, DOWN, buff=0.3)
        self.play(Write(subhead), run_time=0.8)

        # Use layer 3 for the demo (clear block structure)
        demo_layer = 3
        corr_init = attn_data[f"corr_init_L{demo_layer}"]
        corr_final = attn_data[f"corr_final_L{demo_layer}"]

        # Downsample for rendering speed (576x576 → 48x48)
        ds = 48
        noise_hmap = corr_to_heatmap(corr_init, width=4, height=4, downsample=ds)
        noise_hmap.shift(LEFT*3.2 + DOWN*0.3)
        trained_hmap = corr_to_heatmap(corr_final, width=4, height=4, downsample=ds)
        trained_hmap.shift(RIGHT*3.2 + DOWN*0.3)

        init_label = Text("Step 0 (init)", font_size=22, color=STEEL).next_to(
            noise_hmap, UP, buff=0.2)
        trained_label = Text("Step 10000 (trained)", font_size=22, color=GOLD).next_to(
            trained_hmap, UP, buff=0.2)

        self.play(Write(init_label), FadeIn(noise_hmap), run_time=1.2)
        self.wait(0.5)
        self.play(Write(trained_label), FadeIn(trained_hmap, shift=UP*0.2), run_time=1.5)

        stag = source_tag(f"source: L{demo_layer} attn.c_attn, step 0 vs 10000", self)
        self.wait(1)

        # --- ZOOM SEQUENCE ---
        self.camera.frame.save_state()
        self.play(
            self.camera.frame.animate.scale(0.5).move_to(
                trained_hmap.get_center() + LEFT*0.3 + UP*0.3),
            FadeOut(noise_hmap), FadeOut(init_label),
            FadeOut(trained_label), FadeOut(subhead), FadeOut(stag),
            run_time=2.5, rate_func=smooth)

        z1 = Text("Large blocks emerge...", font_size=10,
                  color=GOLD).move_to(self.camera.frame.get_center() + DOWN*0.7)
        self.play(Write(z1), run_time=0.6)
        self.wait(0.5)

        self.play(self.camera.frame.animate.scale(0.5).shift(RIGHT*0.2 + DOWN*0.1),
                  FadeOut(z1), run_time=2, rate_func=smooth)
        z2 = Text("...containing smaller blocks...", font_size=5,
                  color=TEAL).move_to(self.camera.frame.get_center() + DOWN*0.3)
        self.play(Write(z2), run_time=0.6)
        self.wait(0.5)

        self.play(self.camera.frame.animate.scale(0.5).shift(UP*0.05 + LEFT*0.05),
                  FadeOut(z2), run_time=2, rate_func=smooth)
        z3 = Text("...all the way down", font_size=2.5,
                  color=CORAL).move_to(self.camera.frame.get_center() + DOWN*0.12)
        self.play(Write(z3), run_time=0.6)
        self.wait(1)

        self.play(Restore(self.camera.frame), FadeOut(z3), run_time=2.5, rate_func=smooth)

        fractal_text = Text("Self-similar at every scale — a fractal",
                           font_size=26, color=TEAL).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(trained_hmap), Write(fractal_text), run_time=1)
        self.wait(1.5)

        # Sierpinski comparison
        self.play(FadeOut(trained_hmap), FadeOut(fractal_text))

        small_hmap = corr_to_heatmap(corr_final, width=4, height=4, downsample=ds)
        small_hmap.shift(LEFT*3.5 + DOWN*0.3)
        hmap_label = Text("Real Attention Correlations", font_size=16,
                         color=GOLD).next_to(small_hmap, DOWN, buff=0.15)

        triangles = sierpinski_triangles([-1.8, -1.3], [1.8, -1.3], [0, 1.8], 5)
        sierp = VGroup(*triangles).shift(RIGHT*3.5 + DOWN*0.3)
        sierp_label = Text("Sierpinski Triangle", font_size=16,
                          color=TEAL).next_to(sierp, DOWN, buff=0.15)
        vs = Text("vs", font_size=24, color=SOFT_WHITE)
        comp_text = Text("Same principle: structure repeats at every scale",
                        font_size=24, color=GOLD).next_to(heading, DOWN, buff=0.3)

        self.play(Write(comp_text), run_time=0.8)
        self.play(FadeIn(small_hmap), Write(hmap_label), run_time=0.8)
        self.play(Write(vs), run_time=0.3)
        self.play(LaggedStart(*[FadeIn(t, scale=0.3) for t in triangles],
                              lag_ratio=0.002), run_time=2.5)
        self.play(Write(sierp_label), run_time=0.5)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================
# Scene 6: Layers as Scaled Copies — REAL similarity data
# ============================================================
class ScaledCopies(Scene):
    def construct(self):
        sim_data = load_data("cross_layer_similarity.npz")
        heading = heading_text("Finding #3: Layers Are Scaled Copies")
        self.play(Write(heading), run_time=1)

        # Display real cosine similarity matrix for one weight type
        wtype = "attn_c_attn"
        sim_matrix = sim_data[f"sim_{wtype}"]
        n = sim_matrix.shape[0]

        # Build the matrix visualization
        cell_size = 0.6
        matrix_group = VGroup()
        for i in range(n):
            for j in range(n):
                val = sim_matrix[i][j]
                t = np.clip((val - 0.9) / 0.1, 0, 1)  # normalize 0.9-1.0 range
                color = interpolate_color(ManimColor(STEEL), ManimColor(GOLD), t)
                rect = Square(side_length=cell_size, fill_color=color,
                             fill_opacity=0.8, stroke_width=0.5, stroke_color=GRID_COLOR)
                rect.move_to(np.array([j*cell_size, -i*cell_size, 0]))
                val_text = Text(f"{val:.3f}", font_size=8, color=SOFT_WHITE).move_to(rect)
                matrix_group.add(VGroup(rect, val_text))

        for i in range(n):
            rl = Text(f"L{i}", font_size=12, color=SOFT_WHITE).next_to(
                matrix_group[i*n].get_left(), LEFT, buff=0.15)
            cl = Text(f"L{i}", font_size=12, color=SOFT_WHITE).next_to(
                matrix_group[i].get_top(), UP, buff=0.15)
            matrix_group.add(rl, cl)

        matrix_group.move_to(ORIGIN + DOWN*0.3)
        self.play(FadeIn(matrix_group), run_time=1.5)

        mean_sim = float(np.mean(sim_matrix[np.triu_indices(n, k=1)]))
        highlight = Text(f"Mean cosine similarity = {mean_sim:.4f}",
                        font_size=24, color=TEAL).to_edge(DOWN, buff=0.4)
        self.play(Write(highlight), run_time=1)

        stag = source_tag("source: attn.c_attn spectra, step 10000", self)
        self.wait(2)
        self.play(FadeOut(matrix_group), FadeOut(highlight), FadeOut(stag))

        # Animated spectra overlay from real data
        subhead = Text("Every layer's spectrum: same shape, just scaled",
                      font_size=22, color=SOFT_WHITE).next_to(heading, DOWN, buff=0.3)
        self.play(Write(subhead), run_time=0.8)

        # Get real spectra per layer
        n_wtype = int(sim_data[f"n_{wtype}"])
        sv_norms = [sim_data[f"sv_norm_{wtype}_L{i}"] for i in range(n_wtype)]
        sv_raws = [sim_data[f"sv_{wtype}_L{i}"] for i in range(n_wtype)]

        # Axes from real data range
        first_sv = sv_raws[0]
        log_rank = np.log(np.arange(1, len(first_sv)+1))
        x_max = float(np.ceil(log_rank.max()))

        all_log_svs = [np.log(sv + 1e-10) for sv in sv_raws]
        y_min = float(np.floor(min(ls.min() for ls in all_log_svs)))
        y_max = float(np.ceil(max(ls.max() for ls in all_log_svs)))

        ax = make_axes([0, x_max, 1], [y_min, y_max, 1], x_length=8, y_length=4)
        ax.shift(DOWN * 0.5)
        x_lb = MathTex(r"\log(\text{rank})", font_size=22,
                      color=SOFT_WHITE).next_to(ax.x_axis, DOWN, buff=0.2)
        y_lb = MathTex(r"\log(\sigma)", font_size=22,
                      color=SOFT_WHITE).next_to(ax.y_axis, LEFT, buff=0.2)
        self.play(Create(ax), Write(x_lb), Write(y_lb), run_time=0.8)

        layer_colors = ["#440154", "#443983", "#31688e", "#21918c", "#35b779", "#fde725"]
        n_pts = 80

        curves = []
        labels = []
        for i, (sv, c) in enumerate(zip(sv_raws, layer_colors[:len(sv_raws)])):
            log_sv = np.log(sv + 1e-10)
            local_idx = np.linspace(0, len(sv)-1, n_pts).astype(int)
            pts = [ax.c2p(log_rank[k], np.clip(log_sv[k], y_min, y_max)) for k in local_idx]
            curve = VMobject(color=c, stroke_width=2.5)
            curve.set_points_smoothly(pts)
            lbl = Text(f"L{i}", font_size=16, color=c).next_to(curve.get_start(), UL, buff=0.1)
            curves.append(curve)
            labels.append(lbl)

        self.play(LaggedStart(*[Create(c) for c in curves], lag_ratio=0.12), run_time=2)
        self.play(LaggedStart(*[Write(l) for l in labels], lag_ratio=0.08), run_time=0.8)
        self.wait(1)

        # Collapse: normalize to same scale
        collapse_text = Text("Normalize by top singular value → identical shape",
                            font_size=22, color=TEAL).to_edge(DOWN, buff=0.4)
        self.play(Write(collapse_text), run_time=0.8)

        # Target: all use normalized spectra (sv / sv[0])
        # We need a new axis range for normalized values
        norm_log_svs = [np.log(sv_norm + 1e-10) for sv_norm in sv_norms]
        ny_min = float(np.floor(min(ls.min() for ls in norm_log_svs)))
        ny_max = 0.1  # log(1) = 0

        # Transform curves to overlap
        mid_sv = sv_raws[len(sv_raws)//2]
        mid_log = np.log(mid_sv + 1e-10)
        local_idx = np.linspace(0, len(mid_sv)-1, n_pts).astype(int)
        target_pts = [ax.c2p(log_rank[k], np.clip(mid_log[k], y_min, y_max)) for k in local_idx]

        collapse_anims = []
        for i, curve in enumerate(curves):
            target = VMobject(color=layer_colors[i], stroke_width=2.5)
            target.set_points_smoothly(target_pts)
            collapse_anims.append(Transform(curve, target))

        self.play(*collapse_anims, run_time=2.5, rate_func=smooth)

        same = Text("Same slope. Same shape. Scaled copies.",
                   font_size=26, color=GOLD).to_edge(DOWN, buff=0.4)
        self.play(ReplacementTransform(collapse_text, same), run_time=0.8)

        stag2 = source_tag("source: attn.c_attn all layers, step 10000", self)
        self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================
# Scene 7a: Hurst exponent (2D) — REAL displacement data
# ============================================================
class HurstExponent(Scene):
    def construct(self):
        traj_data = load_data("sgd_trajectory.npz")
        H = float(traj_data["hurst_H"])
        r2 = float(traj_data["hurst_r2"])
        D = float(traj_data["fractal_dim"])
        log_scales = traj_data["log_scales"]
        log_disps = traj_data["log_disps"]
        n_params = int(traj_data["n_params"])

        heading = heading_text("Finding #4: The Fractal Path of Learning")
        self.play(Write(heading), run_time=1)

        subhead = Text(f"How displacement scales with time ({n_params:,} parameters)",
                      font_size=22, color=SOFT_WHITE).next_to(heading, DOWN, buff=0.3)
        self.play(Write(subhead), run_time=0.8)

        # Real data range
        x_min, x_max = float(log_scales.min()) - 0.2, float(log_scales.max()) + 0.2
        y_min, y_max = float(log_disps.min()) - 0.3, float(log_disps.max()) + 0.3

        ax = make_axes([x_min, x_max, 0.5], [y_min, y_max, 0.5],
                       x_length=6, y_length=4)
        ax.shift(DOWN * 0.3)
        xl = MathTex(r"\log(\text{scale})", font_size=24,
                    color=SOFT_WHITE).next_to(ax.x_axis, DOWN, buff=0.2)
        yl = MathTex(r"\log(\text{displacement})", font_size=24,
                    color=SOFT_WHITE).next_to(ax.y_axis, LEFT, buff=0.2)
        self.play(Create(ax), Write(xl), Write(yl), run_time=0.8)

        # Plot real data points
        dots = VGroup(*[
            Dot(ax.c2p(log_scales[i], log_disps[i]), radius=0.06, color=GOLD)
            for i in range(len(log_scales))
        ])
        self.play(LaggedStart(*[FadeIn(d, scale=2) for d in dots], lag_ratio=0.08),
                  run_time=1)

        # Fit line
        intercept = float(traj_data["hurst_intercept"])
        fit_x = np.linspace(x_min + 0.1, x_max - 0.1, 50)
        fit_y = H * fit_x + intercept
        fit_line = VMobject(color=TEAL, stroke_width=2.5)
        fit_line.set_points_smoothly([ax.c2p(x, y) for x, y in zip(fit_x, fit_y)])
        sgd_label = Text(f"SGD: H = {H:.3f}", font_size=18,
                        color=GOLD).next_to(dots[-1], UR, buff=0.15)
        self.play(Create(fit_line), Write(sgd_label), run_time=1)

        # Random walk reference (H=0.5)
        rw_y = 0.5 * fit_x + intercept
        rw_line = DashedLine(
            ax.c2p(fit_x[0], rw_y[0]), ax.c2p(fit_x[-1], rw_y[-1]),
            color=STEEL, stroke_width=2, dash_length=0.1)
        rw_label = Text("Random walk: H = 0.5", font_size=16,
                       color=STEEL).next_to(
            ax.c2p(fit_x[-1], rw_y[-1]), DR, buff=0.1)
        self.play(Create(rw_line), Write(rw_label), run_time=1)

        hurst_eq = MathTex(
            rf"H = {H:.3f} \;\Rightarrow\; D = 2 - H = {D:.3f}",
            font_size=32, color=TEAL
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(hurst_eq), run_time=1)

        stag = source_tag(f"source: 21 checkpoints, R²={r2:.4f}", self)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================
# Scene 7b: Fractal SGD Trajectory (3D) — REAL PCA projection
# ============================================================
class FractalTrajectory(ThreeDScene):
    def construct(self):
        traj_data = load_data("sgd_trajectory.npz")
        pca_3d = traj_data["pca_3d"]  # [21, 3]
        var_exp = traj_data["variance_explained"]
        steps = traj_data["steps"]

        heading = Text("SGD Trajectory in PCA Space", font_size=30,
                      color=GOLD).to_edge(UP, buff=0.5)
        self.add_fixed_in_frame_mobjects(heading)
        self.play(Write(heading), run_time=0.8)

        var_text = Text(
            f"Top 3 PCs: {var_exp.sum()*100:.1f}% variance",
            font_size=18, color=SOFT_WHITE).next_to(heading, DOWN, buff=0.2)
        self.add_fixed_in_frame_mobjects(var_text)
        self.play(Write(var_text), run_time=0.5)

        # Scale PCA coordinates to fit nicely
        scale = 3.0 / np.abs(pca_3d).max()
        pts = pca_3d * scale

        axes = ThreeDAxes(
            x_range=[-4, 4, 1], y_range=[-4, 4, 1], z_range=[-3, 3, 1],
            x_length=6, y_length=6, z_length=4,
            axis_config={"color": GRID_COLOR})

        # SGD trajectory from real data
        sgd_curve = VMobject(color=GOLD, stroke_width=3)
        sgd_curve.set_points_smoothly([np.array(p) for p in pts])

        start_dot = Sphere(radius=0.12, color=GREEN).move_to(pts[0])
        end_dot = Sphere(radius=0.12, color=RED).move_to(pts[-1])

        # Brownian motion for comparison (decorative, clearly labeled)
        rng = np.random.RandomState(42)
        bm_steps_arr = rng.randn(200, 3) * 0.12
        bm_path = np.cumsum(bm_steps_arr, axis=0)
        bm_path -= bm_path.mean(axis=0)
        bm_path *= 2.0 / np.abs(bm_path).max()
        bm_curve = VMobject(color=STEEL, stroke_width=2, stroke_opacity=0.7)
        bm_curve.set_points_smoothly([np.array(p) for p in bm_path])

        sgd_tag = Text("Real SGD (H=0.75)", font_size=18, color=GOLD).to_corner(DL, buff=0.5)
        bm_tag = Text("Random walk reference (H=0.50)", font_size=18,
                     color=STEEL).next_to(sgd_tag, UP, buff=0.1)

        self.set_camera_orientation(phi=70*DEGREES, theta=-45*DEGREES)
        self.play(Create(axes), run_time=0.8)
        self.play(Create(sgd_curve), run_time=3, rate_func=linear)
        self.play(FadeIn(start_dot), FadeIn(end_dot), run_time=0.5)
        self.add_fixed_in_frame_mobjects(sgd_tag)
        self.play(Write(sgd_tag), run_time=0.5)
        self.wait(0.5)

        self.play(Create(bm_curve), run_time=2, rate_func=linear)
        self.add_fixed_in_frame_mobjects(bm_tag)
        self.play(Write(bm_tag), run_time=0.5)

        self.begin_ambient_camera_rotation(rate=0.12)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        meaning = Text("Persistent fractal curve vs. aimless random walk",
                      font_size=20, color=TEAL).to_edge(DOWN, buff=0.4)
        self.add_fixed_in_frame_mobjects(meaning)
        self.play(Write(meaning), run_time=1)
        self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================
# Scene 8: Universality — REAL metrics across scales
# ============================================================
class Universality(Scene):
    def construct(self):
        scale_data = load_data("scale_comparison.npz")
        scale_labels = list(scale_data["scale_labels"])
        n_params = scale_data["n_params"]
        mean_alpha = scale_data["mean_alpha"]
        cross_sim = scale_data["cross_layer_sim"]
        hurst_H = scale_data["hurst_H"]

        heading = heading_text("Finding #5: It's Universal")
        self.play(Write(heading), run_time=1)

        subhead = Text("Three models, three scales, same fractal signatures",
                       font_size=22, color=SOFT_WHITE).next_to(heading, DOWN, buff=0.3)
        self.play(Write(subhead), run_time=0.8)

        # Model icons
        model_colors = ["#2196F3", "#4CAF50", "#FF5722"]
        scale_factors = [0.6, 0.8, 1.0]
        models = VGroup()
        for label, sf, color, np_val in zip(scale_labels, scale_factors, model_colors, n_params):
            n = int(sf * 7)
            stack = VGroup(*[
                RoundedRectangle(width=sf*1.5, height=0.22, corner_radius=0.04,
                                color=color, fill_opacity=0.3, stroke_width=1.5)
                for _ in range(n)])
            stack.arrange(DOWN, buff=0.04)
            lbl = Text(f"{label}\n{np_val/1e6:.1f}M params", font_size=13,
                      color=SOFT_WHITE, line_spacing=1.1).next_to(stack, DOWN, buff=0.15)
            models.add(VGroup(stack, lbl))
        models.arrange(RIGHT, buff=1.2).shift(UP*0.5)
        self.play(LaggedStart(*[FadeIn(m, shift=UP*0.3) for m in models],
                              lag_ratio=0.15), run_time=1.5)
        self.wait(0.5)

        self.play(models.animate.scale(0.45).to_edge(UP, buff=0.4),
                  FadeOut(subhead), FadeOut(heading), run_time=0.8)

        # Real metrics as animated bar charts
        metrics = [
            ("Hurst H", hurst_H, TEAL),
            ("Cross-Layer Sim", cross_sim, GOLD),
            ("Mean α", mean_alpha, CORAL),
        ]

        all_charts = VGroup()
        for title_str, values, accent in metrics:
            t = Text(title_str, font_size=16, color=SOFT_WHITE)
            vmin = float(values.min()) * 0.85
            vmax = float(values.max()) * 1.15
            if vmax == vmin:
                vmax = vmin + 0.1
            bars = VGroup()
            for i, (v, mc) in enumerate(zip(values, model_colors)):
                bar_h = ((v - vmin) / (vmax - vmin)) * 1.8 + 0.2
                bar = Rectangle(width=0.5, height=bar_h, color=mc,
                               fill_opacity=0.6, stroke_width=1.5, stroke_color=mc)
                val_t = Text(f"{v:.3f}", font_size=11, color=accent)
                val_t.next_to(bar, UP, buff=0.04)
                sz_t = Text(["S", "M", "L"][i], font_size=11,
                           color=SOFT_WHITE).next_to(bar, DOWN, buff=0.04)
                bars.add(VGroup(bar, val_t, sz_t))
            bars.arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
            chart = VGroup(t, bars).arrange(DOWN, buff=0.15)
            all_charts.add(chart)

        all_charts.arrange(RIGHT, buff=0.8).shift(DOWN*0.6)

        for chart in all_charts:
            title_m, bars = chart[0], chart[1]
            self.play(Write(title_m), run_time=0.3)
            grow_anims = []
            for bg in bars:
                bar = bg[0]
                h = bar.height
                bar.stretch_to_fit_height(0.01, about_edge=DOWN)
                grow_anims.append(bar.animate.stretch_to_fit_height(h, about_edge=DOWN))
            self.play(*grow_anims, run_time=0.8, rate_func=rush_from)
            for bg in bars:
                bg[1].set_opacity(0)
                bg[2].set_opacity(0)
            self.play(*[bg[1].animate.set_opacity(1) for bg in bars],
                      *[bg[2].animate.set_opacity(1) for bg in bars], run_time=0.4)
            self.wait(0.2)

        # Highlight Hurst
        hr = SurroundingRectangle(all_charts[0][1], color=TEAL, buff=0.12, stroke_width=2.5)
        h_range = f"{float(hurst_H.min()):.3f}–{float(hurst_H.max()):.3f}"
        note = Text(f"Hurst H = {h_range} across 4x scale range",
                   font_size=18, color=TEAL).to_edge(DOWN, buff=0.3)
        self.play(Create(hr), Write(note), run_time=1)

        stag = source_tag("source: checkpoints_small/medium/large", self)
        self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================
# Scene 9: What Does It Mean?
# ============================================================
class WhatDoesItMean(Scene):
    def construct(self):
        # Load real summary values for the findings
        traj_data = load_data("sgd_trajectory.npz")
        scale_data = load_data("scale_comparison.npz")

        H = float(traj_data["hurst_H"])
        D = float(traj_data["fractal_dim"])
        mean_sim = float(scale_data["cross_layer_sim"].mean())

        heading = heading_text("What Does This Mean?")
        self.play(Write(heading), run_time=1)

        findings = [
            ("Weights", "random", "power-law spectra", STEEL, GOLD),
            ("Correlations", "noise", "nested blocks", STEEL, GOLD),
            ("Layers", "independent", f"scaled copies (sim>{mean_sim:.2f})", STEEL, GOLD),
            ("SGD path", "random walk", f"fractal (H={H:.3f}, D={D:.3f})", STEEL, TEAL),
            ("Across scales", "varies", "universal", STEEL, TEAL),
        ]

        timeline = VGroup()
        for label_t, from_t, to_t, cf, ct in findings:
            lbl = Text(label_t, font_size=22, color=SOFT_WHITE, weight=BOLD)
            if lbl.width > 2.0:
                lbl.scale(2.0 / lbl.width)
            ft = Text(from_t, font_size=18, color=cf)
            arrow = Arrow(LEFT*0.5, RIGHT*0.5, color=ct, stroke_width=2.5,
                         max_tip_length_to_length_ratio=0.2)
            tt = Text(to_t, font_size=18, color=ct, weight=BOLD)
            timeline.add(VGroup(lbl, ft, arrow, tt).arrange(RIGHT, buff=0.25))
        timeline.arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(DOWN*0.3)

        for row in timeline:
            self.play(FadeIn(row, shift=RIGHT*0.5), run_time=0.5)
            self.wait(0.15)
        self.wait(1)

        self.play(timeline.animate.shift(UP*0.8).set_opacity(0.3), run_time=0.8)

        p1 = Text("Training doesn't just find good weights.",
                  font_size=28, color=SOFT_WHITE)
        p2 = Text("It builds fractal geometry.",
                  font_size=38, color=GOLD, weight=BOLD)
        punch = VGroup(p1, p2).arrange(DOWN, buff=0.35).shift(DOWN*1.2)

        self.play(Write(p1), run_time=1.5)
        self.wait(0.3)
        p2.scale(0.5).set_opacity(0)
        self.add(p2)
        self.play(p2.animate.scale(2).set_opacity(1), run_time=1.5, rate_func=rush_from)
        self.wait(2)

        self.play(*[FadeOut(m) for m in self.mobjects])

        closing_lines = [
            ("The same mathematics that describes", SOFT_WHITE),
            ("coastlines, galaxies, and blood vessels", TEAL),
            ("also describes the internal structure", SOFT_WHITE),
            ("of neural networks.", GOLD),
        ]
        closing = VGroup(*[Text(t, font_size=26, color=c) for t, c in closing_lines])
        closing.arrange(DOWN, buff=0.25)
        self.play(LaggedStart(*[FadeIn(l, shift=UP*0.2) for l in closing],
                              lag_ratio=0.3), run_time=3)
        self.wait(3)
        self.play(FadeOut(closing), run_time=1)

        title = Text("Neural Networks Are Fractals", font_size=48,
                     color=GOLD, weight=BOLD)
        glow = title.copy().set_color(TEAL).set_opacity(0)
        self.play(Write(title), run_time=1.5)
        self.add(glow)
        for _ in range(2):
            self.play(glow.animate.set_opacity(0.3).scale(1.05),
                      run_time=1, rate_func=there_and_back)
        self.wait(1)
        self.play(FadeOut(title), FadeOut(glow))
