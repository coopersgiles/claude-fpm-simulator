"""Fourier Ptychography Teaching App.

A 5-tab interactive teaching tool for Fourier ptychographic microscopy (FPM).
Built with marimo; runs client-side via WASM. See fpm_teaching_app_spec.md.

FFT convention (single source of truth):
- Object lives in real space, indexed (y, x) with origin at array center after fftshift.
- Spectrum = fftshift(fft2(ifftshift(object))). DC at center of the array.
- Inverse: object = fftshift(ifft2(ifftshift(spectrum))).
- Pupil is defined in the centered Fourier plane (DC at center).
- A tilted illumination with frequency shift (ky, kx) (in pixels) shifts the
  object's spectrum by (ky, kx); the fixed pupil then samples it.
- np.roll(spec, shift=(ky, kx), axis=(0, 1)) shifts content toward +y, +x.
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


# ---------------------------------------------------------------------------
# Imports & global defaults
# ---------------------------------------------------------------------------


@app.cell
def _imports():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    return Circle, mo, np, plt


@app.cell
def _defaults():
    # Physical defaults (labeled; absolute units are normalized for the sim).
    WAVELENGTH_UM = 0.5            # green light
    PIXEL_PITCH_UM = 0.2           # object-plane pixel pitch
    OBJ_NA_DEFAULT = 0.10          # low-NA objective => room to grow with FPM
    ILLUM_NA_MAX = 0.40            # outermost LED's NA
    # Grid sizes per spec.
    N_LIVE = 256                   # real-time tabs (1-3)
    N_RECON = 128                  # reconstruction tabs (4-5)
    return (
        ILLUM_NA_MAX,
        N_LIVE,
        N_RECON,
        OBJ_NA_DEFAULT,
        PIXEL_PITCH_UM,
        WAVELENGTH_UM,
    )


# ---------------------------------------------------------------------------
# FFT utilities + forward model (single source of truth, reused everywhere)
# ---------------------------------------------------------------------------


@app.cell
def _fft_utils(np):
    def ft(x):
        """Centered forward 2D FFT. Input/output have DC at array center."""
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

    def ift(X):
        """Centered inverse 2D FFT. Input/output have DC at array center."""
        return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

    def freq_grid(N, pitch=1.0):
        """Centered frequency-coordinate grids (cycles per length unit)."""
        f = np.fft.fftshift(np.fft.fftfreq(N, d=pitch))
        fy, fx = np.meshgrid(f, f, indexing="ij")
        return fy, fx

    def make_pupil(N, cutoff_px, aberration_phase=None):
        """Disk pupil of radius `cutoff_px` centered on the (N,N) Fourier grid.

        cutoff_px: radius in pixels (= NA / lambda mapped to grid).
        aberration_phase: optional real array, same shape, added inside the disk.
        Returns complex pupil P = disk * exp(i * phase).
        """
        cy, cx = N // 2, N // 2
        yy, xx = np.ogrid[:N, :N]
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        disk = (r2 <= cutoff_px ** 2).astype(np.float64)
        if aberration_phase is None:
            return disk.astype(np.complex128)
        return disk * np.exp(1j * aberration_phase)

    def na_to_cutoff_px(NA, wavelength_um, pixel_pitch_um, N):
        """Map a numerical aperture to a pupil-cutoff radius in pixels.

        Spatial frequency at the cutoff: f_c = NA / lambda  (cycles / length).
        The DFT bin spacing on an N-by-N grid with pitch p is 1/(N*p).
        So cutoff in pixels = f_c * (N * p) = NA * N * p / lambda.
        """
        return NA * N * pixel_pitch_um / wavelength_um

    def simulate_intensity(object_field, pupil, k_shift_px):
        """Forward model. See module docstring for sign conventions.

        object_field : complex 2D array (transmittance A*exp(i*phi)).
        pupil        : complex 2D array, centered, same shape.
        k_shift_px   : (ky, kx) integer pixel shift of the object spectrum.
        returns      : real 2D intensity image of the same shape.
        """
        S = ft(object_field)
        ky, kx = int(round(k_shift_px[0])), int(round(k_shift_px[1]))
        S_shifted = np.roll(S, shift=(ky, kx), axis=(0, 1))
        low_res_field = ift(pupil * S_shifted)
        return np.abs(low_res_field) ** 2

    def low_res_field(object_field, pupil, k_shift_px):
        """Same as simulate_intensity but returns the complex low-res field."""
        S = ft(object_field)
        ky, kx = int(round(k_shift_px[0])), int(round(k_shift_px[1]))
        S_shifted = np.roll(S, shift=(ky, kx), axis=(0, 1))
        return ift(pupil * S_shifted)

    return (
        freq_grid,
        ft,
        ift,
        low_res_field,
        make_pupil,
        na_to_cutoff_px,
        simulate_intensity,
    )


# ---------------------------------------------------------------------------
# Plotting helper (two-pane real / Fourier, phase masked by magnitude)
# ---------------------------------------------------------------------------


@app.cell
def _plotting(Circle, mo, np, plt):
    def _phase_with_alpha(field, gamma=0.6):
        """Render complex field's phase, alpha-masked by normalized magnitude."""
        mag = np.abs(field)
        m = mag / (mag.max() + 1e-12)
        alpha = np.clip(m ** gamma, 0.0, 1.0)
        return np.angle(field), alpha

    def _show_complex(ax, field, mode, *, title="", cmap_mag="gray",
                      cmap_phase="twilight", log_mag=False, vmin=None, vmax=None,
                      extent=None):
        """Draw one pane showing a complex field as magnitude OR phase."""
        if mode == "magnitude":
            mag = np.abs(field)
            if log_mag:
                mag = np.log1p(mag)
            ax.imshow(mag, cmap=cmap_mag, vmin=vmin, vmax=vmax,
                      interpolation="nearest", extent=extent, origin="upper")
        elif mode == "phase":
            phi, alpha = _phase_with_alpha(field)
            # imshow doesn't natively alpha per-pixel from a separate array, so
            # render via RGBA composite.
            import matplotlib.cm as cm
            cmap = plt.get_cmap(cmap_phase)
            normed = (phi + np.pi) / (2 * np.pi)
            rgba = cmap(normed)
            rgba[..., 3] = alpha
            ax.imshow(rgba, interpolation="nearest", extent=extent,
                      origin="upper")
        elif mode == "real":
            ax.imshow(field.real, cmap=cmap_mag, vmin=vmin, vmax=vmax,
                      interpolation="nearest", extent=extent, origin="upper")
        else:
            raise ValueError(f"Unknown mode {mode!r}")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    def two_pane(left_field, right_field, *,
                 left_mode="magnitude", right_mode="magnitude",
                 left_title="object", right_title="Fourier plane",
                 pupil_radius_px=None, log_right=True,
                 figsize=(7.5, 3.7)):
        """Standard two-pane plot. Pupil circle is drawn on the right if given."""
        fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
        _show_complex(axes[0], left_field, left_mode, title=left_title)
        _show_complex(axes[1], right_field, right_mode, title=right_title,
                      log_mag=log_right)
        if pupil_radius_px is not None:
            N = right_field.shape[0]
            ring = Circle((N / 2 - 0.5, N / 2 - 0.5), pupil_radius_px,
                          fill=False, edgecolor="cyan", linewidth=1.2,
                          linestyle="--", alpha=0.9)
            axes[1].add_patch(ring)
        return fig

    def three_pane(left_field, middle_field, right_field, *,
                   modes=("magnitude", "magnitude", "magnitude"),
                   titles=("object", "pupil-sampled spectrum", "image"),
                   pupil_radius_px=None, log_middle=True,
                   figsize=(10.5, 3.7)):
        fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        _show_complex(axes[0], left_field, modes[0], title=titles[0])
        _show_complex(axes[1], middle_field, modes[1], title=titles[1],
                      log_mag=log_middle)
        _show_complex(axes[2], right_field, modes[2], title=titles[2])
        if pupil_radius_px is not None:
            N = middle_field.shape[0]
            ring = Circle((N / 2 - 0.5, N / 2 - 0.5), pupil_radius_px,
                          fill=False, edgecolor="cyan", linewidth=1.2,
                          linestyle="--", alpha=0.9)
            axes[1].add_patch(ring)
        return fig

    def show_fig(fig):
        """Render a matplotlib figure as a static PNG and release it.

        Static rendering is far lighter than mo.mpl.interactive in the
        WASM build, and closing the figure keeps figures from piling up
        across interactions (a real per-session memory/perf leak).
        """
        html = mo.as_html(fig)
        plt.close(fig)
        return html

    return show_fig, three_pane, two_pane


# ---------------------------------------------------------------------------
# Object library: presets in both real-space and as "amplitude / phase / both"
# ---------------------------------------------------------------------------


@app.cell
def _object_library(np):
    def _coords(N):
        y = np.arange(N) - N // 2
        yy, xx = np.meshgrid(y, y, indexing="ij")
        return yy, xx

    def preset_object(name, N, *, tx=0, ty=0, scale=1.0, rotation_deg=0.0):
        """Return a real-valued 2D array on an NxN grid for shape-based presets.

        Transformations (translate / scale / rotate) are applied to the
        underlying coordinate frame so the math is exact.
        """
        yy, xx = _coords(N)
        th = np.deg2rad(rotation_deg)
        c, s = np.cos(th), np.sin(th)
        # inverse map: take output (yy,xx), apply inverse transform.
        x = (c * (xx - tx) + s * (yy - ty)) / max(scale, 1e-6)
        y = (-s * (xx - tx) + c * (yy - ty)) / max(scale, 1e-6)
        if name == "point":
            obj = np.zeros((N, N))
            yi, xi = N // 2 + int(round(ty)), N // 2 + int(round(tx))
            yi = np.clip(yi, 0, N - 1)
            xi = np.clip(xi, 0, N - 1)
            obj[yi, xi] = 1.0
            return obj
        if name == "two points":
            obj = np.zeros((N, N))
            d = max(4, int(round(8 * scale)))
            for (dy, dx) in [(-d // 2, -d // 2), (d // 2, d // 2)]:
                yi = np.clip(N // 2 + int(round(ty)) + dy, 0, N - 1)
                xi = np.clip(N // 2 + int(round(tx)) + dx, 0, N - 1)
                obj[yi, xi] = 1.0
            return obj
        if name == "rectangle":
            w = 30 * scale
            h = 18 * scale
            return ((np.abs(x) <= w) & (np.abs(y) <= h)).astype(np.float64)
        if name == "slit":
            w = 4 * scale
            h = 50 * scale
            return ((np.abs(x) <= w) & (np.abs(y) <= h)).astype(np.float64)
        if name == "disk":
            r = 24 * scale
            return ((x * x + y * y) <= r * r).astype(np.float64)
        if name == "circle (ring)":
            r = 24 * scale
            return (np.abs(np.sqrt(x * x + y * y) - r) <= 1.5).astype(np.float64)
        if name == "grating":
            period = max(4, int(round(8 * scale)))
            return (0.5 + 0.5 * np.cos(2 * np.pi * x / period)).astype(np.float64)
        if name == "cross":
            w = 4 * scale
            arm = 30 * scale
            a = (np.abs(x) <= arm) & (np.abs(y) <= w)
            b = (np.abs(y) <= arm) & (np.abs(x) <= w)
            return (a | b).astype(np.float64)
        if name == "letter A":
            # A simple "A": two diagonal legs + a crossbar.
            scale_eff = scale
            h = 40 * scale_eff
            mask = np.zeros_like(x, dtype=bool)
            # legs
            slope = 0.5
            for sign in (-1, +1):
                line = np.abs(x - sign * slope * (y + h)) <= 2.0
                mask |= line & (np.abs(y) <= h)
            crossbar = (np.abs(y - 4 * scale_eff) <= 2.0) & (np.abs(x) <= 12 * scale_eff)
            mask |= crossbar
            return mask.astype(np.float64)
        raise ValueError(f"Unknown preset {name!r}")

    PRESETS = [
        "point", "two points", "rectangle", "slit", "disk",
        "circle (ring)", "grating", "cross", "letter A",
    ]

    def complex_object(amplitude=None, phase=None, *, N=None):
        """Combine optional amplitude (in [0,1]) and phase (radians) maps."""
        if amplitude is None and phase is None:
            raise ValueError("need at least one of amplitude or phase")
        if amplitude is None:
            amplitude = np.ones_like(phase)
        if phase is None:
            phase = np.zeros_like(amplitude)
        return amplitude.astype(np.float64) * np.exp(1j * phase.astype(np.float64))

    def cell_like_phase_object(N, *, amp_floor=0.95):
        """A 'cell-like' transparent object: ~uniform amplitude, structured phase."""
        yy, xx = _coords(N)
        # Three Gaussian phase bumps (smooth, low-frequency-ish).
        centers = [(0, 0, 18.0, 1.2),
                   (-12, 14, 10.0, -0.8),
                   (15, -10, 8.0, 0.9)]
        phi = np.zeros((N, N))
        for (cy, cx, sigma, weight) in centers:
            phi += weight * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        amp = amp_floor + (1 - amp_floor) * (1 - np.clip(np.abs(phi) / 1.5, 0, 1))
        return amp, phi

    def absorption_object(N):
        """A pure-absorption USAF-ish blob (no phase)."""
        yy, xx = _coords(N)
        amp = np.ones((N, N))
        # carve features
        amp -= 0.85 * ((np.abs(xx) <= 18) & (np.abs(yy) <= 4))
        amp -= 0.85 * ((np.abs(yy) <= 18) & (np.abs(xx) <= 4))
        amp -= 0.6 * (((xx - 22) ** 2 + (yy - 22) ** 2) <= 40)
        amp -= 0.6 * (((xx + 22) ** 2 + (yy + 22) ** 2) <= 40)
        return np.clip(amp, 0.0, 1.0)

    def mixed_object(N):
        """Mixed absorption + phase, for the 'real samples are both' point."""
        amp = absorption_object(N)
        _, phi = cell_like_phase_object(N, amp_floor=1.0)  # just the phase part
        return amp, 0.7 * phi

    return (
        PRESETS,
        absorption_object,
        cell_like_phase_object,
        complex_object,
        mixed_object,
        preset_object,
    )


# ---------------------------------------------------------------------------
# Zernike polynomials (low order) for aberration toggles (Tab 2E, Tab 5)
# ---------------------------------------------------------------------------


@app.cell
def _zernike(np):
    def _rho_theta(N, R):
        cy, cx = N // 2, N // 2
        yy, xx = np.ogrid[:N, :N]
        y = (yy - cy) / R
        x = (xx - cx) / R
        rho = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)
        return rho, theta

    def zernike_phase(N, R, coeffs):
        """Build a phase map from low-order Zernike coefficients (waves -> rad).

        coeffs keys (each a coefficient, in radians peak):
          'defocus', 'astig_x', 'astig_y', 'coma_x', 'coma_y', 'spherical'.
        Only defined inside the pupil disk (rho<=1); zero outside.
        """
        rho, theta = _rho_theta(N, max(R, 1.0))
        in_disk = rho <= 1.0
        phi = np.zeros((N, N))
        c = coeffs
        if c.get("defocus", 0.0):
            phi += c["defocus"] * (2 * rho ** 2 - 1)
        if c.get("astig_x", 0.0):
            phi += c["astig_x"] * rho ** 2 * np.cos(2 * theta)
        if c.get("astig_y", 0.0):
            phi += c["astig_y"] * rho ** 2 * np.sin(2 * theta)
        if c.get("coma_x", 0.0):
            phi += c["coma_x"] * (3 * rho ** 3 - 2 * rho) * np.cos(theta)
        if c.get("coma_y", 0.0):
            phi += c["coma_y"] * (3 * rho ** 3 - 2 * rho) * np.sin(theta)
        if c.get("spherical", 0.0):
            phi += c["spherical"] * (6 * rho ** 4 - 6 * rho ** 2 + 1)
        return phi * in_disk

    return (zernike_phase,)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


@app.cell
def _header(mo):
    header = mo.md(
        r"""
        # Fourier Ptychography — Teaching Sim

        A guided tour through the physics and math of **Fourier ptychographic
        microscopy (FPM)**. Five tabs, in order:

        1. **FT properties** — what the Fourier transform does, by doing it.
        2. **The central dogma** — image = inverse transform of (object × pupil).
        3. **FPM premise** — tilted illumination shifts the spectrum.
        4. **The inverse problem** — recovering phase from intensity-only data.
        5. **Synthesis** — the same pipeline under realistic perturbations.

        Throughout: **real-space on the left, Fourier on the right**. Phase is
        rendered with a cyclic colormap and alpha-masked by magnitude so empty
        regions don't show random-phase confetti.
        """
    )
    return (header,)


# ===========================================================================
# TAB 1  —  Widget 1: dual-pane FT explorer with predict-then-reveal
# ===========================================================================


@app.cell
def _t1w1_controls(PRESETS, mo):
    t1w1_preset = mo.ui.dropdown(
        options=PRESETS, value="rectangle", label="object A",
    )
    t1w1_tx = mo.ui.slider(-60, 60, value=0, label="translate x (px)")
    t1w1_ty = mo.ui.slider(-60, 60, value=0, label="translate y (px)")
    t1w1_scale = mo.ui.slider(0.3, 2.5, step=0.05, value=1.0, label="scale")
    t1w1_rot = mo.ui.slider(-90, 90, step=1, value=0, label="rotation (deg)")
    t1w1_add = mo.ui.switch(value=False, label="add second object")
    t1w1_preset_b = mo.ui.dropdown(
        options=PRESETS, value="disk", label="object B",
    )
    t1w1_view = mo.ui.dropdown(
        options=["magnitude (log)", "phase"],
        value="magnitude (log)", label="FT view",
    )
    t1w1_controls = mo.vstack([
        mo.md("**Widget 1 — Dual-pane FT explorer.**"),
        mo.hstack([t1w1_preset, t1w1_view]),
        mo.hstack([t1w1_tx, t1w1_ty]),
        mo.hstack([t1w1_scale, t1w1_rot]),
        mo.hstack([t1w1_add, t1w1_preset_b]),
    ])
    return (
        t1w1_add,
        t1w1_controls,
        t1w1_preset,
        t1w1_preset_b,
        t1w1_rot,
        t1w1_scale,
        t1w1_tx,
        t1w1_ty,
        t1w1_view,
    )


@app.cell
def _t1w1_plot(
    N_LIVE,
    ft,
    mo,
    np,
    preset_object,
    t1w1_add,
    t1w1_preset,
    t1w1_preset_b,
    t1w1_rot,
    t1w1_scale,
    t1w1_tx,
    t1w1_ty,
    t1w1_view,
    two_pane,
):
    _N = N_LIVE
    _obj_a = preset_object(
        t1w1_preset.value, _N,
        tx=t1w1_tx.value, ty=t1w1_ty.value,
        scale=t1w1_scale.value, rotation_deg=t1w1_rot.value,
    )
    if t1w1_add.value:
        _obj_b = preset_object(t1w1_preset_b.value, _N)
        _obj = _obj_a + _obj_b
        _title = f"{t1w1_preset.value} + {t1w1_preset_b.value}"
    else:
        _obj = _obj_a
        _title = t1w1_preset.value
    _obj_c = _obj.astype(np.complex128)
    _spec = ft(_obj_c)
    _mode = "magnitude" if t1w1_view.value.startswith("magnitude") else "phase"
    _fig = two_pane(
        _obj_c, _spec,
        left_mode="magnitude", right_mode=_mode,
        left_title=f"object: {_title}",
        right_title=("FT magnitude (log)"
                     if _mode == "magnitude" else "FT phase (cyclic)"),
        log_right=(_mode == "magnitude"),
    )
    t1w1_plot = show_fig(_fig)
    return (t1w1_plot,)


@app.cell
def _t1w1_prompts(mo):
    t1w1_prompts = mo.vstack([
        mo.md(
            "### Predict, then reveal\n\n"
            "Hold each prediction in your head before you peek — the answers "
            "are the FT properties you'll lean on for the rest of the app."
        ),
        mo.accordion({
            "1. Translate the object — what changes in **magnitude**? in **phase**?": mo.md(
                "**Shift theorem.** The magnitude is **invariant** under "
                "translation — moving the object only adds a **linear phase ramp** "
                "to its Fourier transform. Switch FT view to *phase* and slide "
                "translate x or y: you'll see the ramp rotate. This is why "
                "phase carries position information."
            ),
            "2. Toggle 'add second object' — predict the FT of (A + B)": mo.md(
                "**Linearity.** $\\mathcal{F}\\{A+B\\} = \\mathcal{F}\\{A\\} + "
                "\\mathcal{F}\\{B\\}$. The transform of a sum is the sum of "
                "transforms — but the *magnitude* of a sum is **not** the sum "
                "of magnitudes (interference!)."
            ),
            "3. Scale the object larger — what happens to its FT?": mo.md(
                "**Inverse scaling.** Stretching the object by a factor $s$ "
                "compresses its FT by the same factor: "
                "$\\mathcal{F}\\{f(x/s)\\}(k) = s\\,\\mathcal{F}\\{f\\}(s\\,k)$. "
                "Tight features in real space ⇒ broad features in frequency, "
                "and vice versa."
            ),
            "4. Rotate the object — what happens to its FT?": mo.md(
                "**Rotational equivariance.** The FT rotates with the object. "
                "Whatever frame you pick in real space, the same frame applies "
                "in Fourier space."
            ),
            "5. Pick a real-valued object — look at the FT magnitude": mo.md(
                "**Hermitian symmetry.** Real objects have FTs that satisfy "
                "$F(-k) = \\overline{F(k)}$, so the **magnitude is symmetric "
                "through the origin**. Pure phase objects don't share this "
                "symmetry — a hint of why phase objects are hard, and why "
                "FPM works (Tab 4)."
            ),
        }),
    ])
    return (t1w1_prompts,)


# ===========================================================================
# TAB 1  —  Widget 2: phase-vs-magnitude swap
# ===========================================================================


@app.cell
def _t1w2_controls(PRESETS, mo):
    t1w2_a = mo.ui.dropdown(
        options=PRESETS, value="cross", label="image A",
    )
    t1w2_b = mo.ui.dropdown(
        options=PRESETS, value="letter A", label="image B",
    )
    t1w2_controls = mo.vstack([
        mo.md(
            "**Widget 2 — Phase-vs-magnitude swap.**  "
            "Take two objects A and B. Compute their FTs. Build a new "
            "frequency-domain field by **gluing A's magnitude onto B's "
            "phase** (and vice versa), then inverse-transform. Predict "
            "before you reveal: which input does the reconstruction "
            "look like — the one whose **magnitude** was used, or the one "
            "whose **phase** was used?"
        ),
        mo.hstack([t1w2_a, t1w2_b]),
    ])
    return t1w2_a, t1w2_b, t1w2_controls


@app.cell
def _t1w2_plot(
    Circle,
    N_LIVE,
    ft,
    ift,
    mo,
    np,
    plt,
    preset_object,
    t1w2_a,
    t1w2_b,
):
    _N = N_LIVE
    _A = preset_object(t1w2_a.value, _N).astype(np.complex128)
    _B = preset_object(t1w2_b.value, _N).astype(np.complex128)
    _FA = ft(_A)
    _FB = ft(_B)
    _eps = 1e-12
    # |A| * phase(B)
    _hyb1 = np.abs(_FA) * np.exp(1j * np.angle(_FB))
    # |B| * phase(A)
    _hyb2 = np.abs(_FB) * np.exp(1j * np.angle(_FA))
    _rec1 = ift(_hyb1).real
    _rec2 = ift(_hyb2).real

    _fig, _ax = plt.subplots(2, 2, figsize=(7.5, 7.0),
                             constrained_layout=True)
    _ax[0, 0].imshow(np.abs(_A), cmap="gray")
    _ax[0, 0].set_title(f"A: {t1w2_a.value}", fontsize=10)
    _ax[0, 1].imshow(np.abs(_B), cmap="gray")
    _ax[0, 1].set_title(f"B: {t1w2_b.value}", fontsize=10)
    _ax[1, 0].imshow(_rec1, cmap="gray")
    _ax[1, 0].set_title("inv FT of  |F(A)| × exp(i·phase(F(B)))",
                        fontsize=10)
    _ax[1, 1].imshow(_rec2, cmap="gray")
    _ax[1, 1].set_title("inv FT of  |F(B)| × exp(i·phase(F(A)))",
                        fontsize=10)
    for _a in _ax.ravel():
        _a.set_xticks([])
        _a.set_yticks([])
    t1w2_plot = show_fig(_fig)
    t1w2_explain = mo.accordion({
        "Reveal the takeaway": mo.md(
            "The reconstruction follows the **phase**, not the magnitude. "
            "Each reconstruction below resembles the object whose *phase* "
            "we kept, not the one whose magnitude we kept.\n\n"
            "**This is why throwing away phase is catastrophic** — phase "
            "carries most of the structural information about an object. "
            "And it's exactly why Tab 4 has to work so hard: an intensity "
            "image throws phase away, but FPM gets it back."
        ),
    })
    return t1w2_explain, t1w2_plot


# ===========================================================================
# TAB 1  —  Widget 3: convolution theorem (light touch)
# ===========================================================================


@app.cell
def _t1w3_controls(PRESETS, mo):
    t1w3_obj = mo.ui.dropdown(
        options=PRESETS, value="letter A", label="object",
    )
    t1w3_kernel = mo.ui.dropdown(
        options=["gaussian blur", "box blur", "edge (Laplacian-ish)",
                 "horizontal smear"],
        value="gaussian blur", label="kernel",
    )
    t1w3_radius = mo.ui.slider(1, 12, step=1, value=4, label="kernel size (px)")
    t1w3_controls = mo.vstack([
        mo.md("**Widget 3 — Convolution theorem (light touch).**"),
        mo.hstack([t1w3_obj, t1w3_kernel, t1w3_radius]),
    ])
    return t1w3_controls, t1w3_kernel, t1w3_obj, t1w3_radius


@app.cell
def _t1w3_plot(
    N_LIVE, ft, ift, mo, np, plt, preset_object,
    t1w3_kernel, t1w3_obj, t1w3_radius,
):
    _N = N_LIVE
    _obj = preset_object(t1w3_obj.value, _N).astype(np.complex128)
    _r = int(t1w3_radius.value)
    _yy, _xx = np.meshgrid(
        np.arange(_N) - _N // 2, np.arange(_N) - _N // 2, indexing="ij",
    )
    _kname = t1w3_kernel.value
    if _kname == "gaussian blur":
        _k = np.exp(-(_yy ** 2 + _xx ** 2) / (2 * _r ** 2))
    elif _kname == "box blur":
        _k = ((np.abs(_yy) <= _r) & (np.abs(_xx) <= _r)).astype(np.float64)
    elif _kname == "edge (Laplacian-ish)":
        _k = np.zeros((_N, _N))
        _cy, _cx = _N // 2, _N // 2
        _k[_cy, _cx] = 4.0
        for _dy, _dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            _k[_cy + _dy, _cx + _dx] = -1.0
    else:  # horizontal smear
        _k = (np.abs(_yy) <= 1) & (np.abs(_xx) <= _r)
        _k = _k.astype(np.float64)
    _k = _k / (_k.sum() + 1e-12) if _kname != "edge (Laplacian-ish)" else _k
    _Fobj = ft(_obj)
    _Fk = ft(_k.astype(np.complex128))
    _Fconv = _Fobj * _Fk
    _conv = ift(_Fconv).real

    _fig, _ax = plt.subplots(2, 3, figsize=(10.5, 6.6),
                             constrained_layout=True)
    _ax[0, 0].imshow(_obj.real, cmap="gray")
    _ax[0, 0].set_title("object", fontsize=10)
    _ax[0, 1].imshow(_k, cmap="gray")
    _ax[0, 1].set_title(f"kernel: {_kname}", fontsize=10)
    _ax[0, 2].imshow(_conv, cmap="gray")
    _ax[0, 2].set_title("object * kernel  (space)", fontsize=10)
    _ax[1, 0].imshow(np.log1p(np.abs(_Fobj)), cmap="gray")
    _ax[1, 0].set_title("|F(object)|  (log)", fontsize=10)
    _ax[1, 1].imshow(np.log1p(np.abs(_Fk)), cmap="gray")
    _ax[1, 1].set_title("|F(kernel)|  (log)", fontsize=10)
    _ax[1, 2].imshow(np.log1p(np.abs(_Fconv)), cmap="gray")
    _ax[1, 2].set_title("F(object) × F(kernel)  (log)", fontsize=10)
    for _a in _ax.ravel():
        _a.set_xticks([])
        _a.set_yticks([])
    t1w3_plot = show_fig(_fig)
    t1w3_caption = mo.md(
        r"""
        **Convolution in space = multiplication in frequency.** The bottom row
        shows this directly: the rightmost panel is exactly the product of
        the two panels to its left. *Imaging through a pupil*, which we build
        next tab, will turn out to be exactly this: multiplying the object's
        spectrum by the pupil (a frequency-space mask) — equivalent to
        convolving the object with the pupil's inverse transform (the PSF).
        """
    )
    return t1w3_caption, t1w3_plot


# ===========================================================================
# TAB 1 assembly
# ===========================================================================


@app.cell
def _t1_assembly(
    mo,
    t1w1_controls, t1w1_plot, t1w1_prompts,
    t1w2_controls, t1w2_plot, t1w2_explain,
    t1w3_controls, t1w3_plot, t1w3_caption,
):
    tab1 = mo.vstack([
        mo.md(
            "## Tab 1 — Fourier transform: discover the properties\n\n"
            "We're going to *use* the Fourier transform until its properties "
            "become obvious. The key bits to internalize: **position lives in "
            "phase** (shift theorem), the transform is **linear**, and "
            "**phase is the precious part** — three facts we'll lean on for "
            "the rest of the app."
        ),
        t1w1_controls, t1w1_plot, t1w1_prompts,
        mo.md("---"),
        t1w2_controls, t1w2_plot, t1w2_explain,
        mo.md("---"),
        t1w3_controls, t1w3_plot, t1w3_caption,
    ])
    return (tab1,)


# ===========================================================================
# TAB 2  —  Stage A: One point → its far field (linear phase ramp)
# ===========================================================================


@app.cell
def _t2a_controls(mo):
    t2a_x = mo.ui.slider(-60, 60, value=12, label="point x (px)")
    t2a_y = mo.ui.slider(-60, 60, value=-8, label="point y (px)")
    t2a_view = mo.ui.dropdown(
        options=["phase (ramp)", "magnitude"],
        value="phase (ramp)",
        label="far-field view",
    )
    t2a_controls = mo.vstack([
        mo.md("**Stage A — One object point → its far field.**"),
        mo.hstack([t2a_x, t2a_y, t2a_view]),
    ])
    return t2a_controls, t2a_view, t2a_x, t2a_y


@app.cell
def _t2a_plot(N_LIVE, ft, mo, np, t2a_view, t2a_x, t2a_y, two_pane):
    _N = N_LIVE
    _obj = np.zeros((_N, _N), dtype=np.complex128)
    _yi = np.clip(_N // 2 + int(t2a_y.value), 0, _N - 1)
    _xi = np.clip(_N // 2 + int(t2a_x.value), 0, _N - 1)
    _obj[_yi, _xi] = 1.0
    _spec = ft(_obj)
    _right_mode = "phase" if t2a_view.value.startswith("phase") else "magnitude"
    _fig = two_pane(
        _obj, _spec,
        left_mode="magnitude", right_mode=_right_mode,
        left_title=f"object: point at (x={t2a_x.value}, y={t2a_y.value})",
        right_title=("far-field phase (linear ramp)"
                     if _right_mode == "phase"
                     else "far-field |F| (uniform)"),
        log_right=False,
    )
    t2a_plot = show_fig(_fig)
    t2a_caption = mo.md(
        r"""
        **What to notice.** The far-field **magnitude is uniform** — a point's
        spectrum is flat. The **phase is a linear ramp**, and its slope
        rotates as you move the point. That is the **shift theorem** at its
        most elemental: *position in real space ↔ a phase slope in Fourier
        space*. Tab 3 will run this in reverse: tilted illumination puts a
        ramp on the *object*, which shifts the *spectrum*.
        """
    )
    return t2a_caption, t2a_plot


# ===========================================================================
# TAB 2  —  Stage B: Many points, complex weights → far field by linearity
# ===========================================================================


@app.cell
def _t2b_controls(mo):
    t2b_n = mo.ui.slider(1, 6, value=3, label="number of points")
    t2b_sep = mo.ui.slider(4, 40, value=14, label="spacing (px)")
    t2b_phase_spread = mo.ui.slider(
        0.0, 3.1416, step=0.1, value=0.0,
        label="random phase spread (rad)",
    )
    t2b_amp_spread = mo.ui.slider(
        0.0, 1.0, step=0.05, value=0.0, label="amplitude variation",
    )
    t2b_view = mo.ui.dropdown(
        options=["magnitude (interference)", "phase"],
        value="magnitude (interference)",
        label="far-field view",
    )
    t2b_controls = mo.vstack([
        mo.md("**Stage B — Linearity builds an object from points.**"),
        mo.hstack([t2b_n, t2b_sep, t2b_view]),
        mo.hstack([t2b_amp_spread, t2b_phase_spread]),
    ])
    return (
        t2b_amp_spread,
        t2b_controls,
        t2b_n,
        t2b_phase_spread,
        t2b_sep,
        t2b_view,
    )


@app.cell
def _t2b_plot(
    N_LIVE,
    ft,
    mo,
    np,
    t2b_amp_spread,
    t2b_n,
    t2b_phase_spread,
    t2b_sep,
    t2b_view,
    two_pane,
):
    _N = N_LIVE
    _obj = np.zeros((_N, _N), dtype=np.complex128)
    _rng = np.random.default_rng(seed=42)
    _n = int(t2b_n.value)
    _d = int(t2b_sep.value)
    # arrange n points on a small circle, deterministically
    _angles = np.linspace(0, 2 * np.pi, _n, endpoint=False)
    _amps_var = t2b_amp_spread.value * _rng.standard_normal(_n)
    _phs_var = t2b_phase_spread.value * _rng.standard_normal(_n)
    for _k in range(_n):
        _yi = int(round(_N // 2 + _d * np.sin(_angles[_k])))
        _xi = int(round(_N // 2 + _d * np.cos(_angles[_k])))
        _yi = np.clip(_yi, 0, _N - 1)
        _xi = np.clip(_xi, 0, _N - 1)
        _amp = float(np.clip(1.0 + _amps_var[_k], 0.1, 2.0))
        _ph = float(_phs_var[_k])
        _obj[_yi, _xi] += _amp * np.exp(1j * _ph)
    _spec = ft(_obj)
    _mode = "magnitude" if t2b_view.value.startswith("magnitude") else "phase"
    _fig = two_pane(
        _obj, _spec,
        left_mode="magnitude", right_mode=_mode,
        left_title=f"{_n} points (spacing {_d}px)",
        right_title=("|F| — fringes from interference"
                     if _mode == "magnitude" else "phase"),
        log_right=False,
    )
    t2b_plot = show_fig(_fig)
    t2b_caption = mo.md(
        r"""
        **What to notice.** Each point contributes a ramp; their **sum** is the
        far field. Two equal-amplitude points give classic intensity fringes
        whose spacing varies **inversely** with the points' separation. Add a
        few points with varying amplitude and phase: the far field is just the
        superposition. That is **linearity**:
        $\,\mathcal{F}\{A+B\} = \mathcal{F}\{A\} + \mathcal{F}\{B\}$. Any
        object can be thought of as a sum of points, so its far field is the
        sum of those ramps — i.e. **the far field is the Fourier transform of
        the object**.
        """
    )
    return t2b_caption, t2b_plot


# ===========================================================================
# TAB 2  —  Stage C: A lens with a pupil → imaging; resolution from cutoff
# ===========================================================================


@app.cell
def _t2c_controls(mo):
    t2c_preset = mo.ui.dropdown(
        options=["cross", "letter A", "rectangle", "disk",
                 "grating", "two points"],
        value="cross", label="object",
    )
    t2c_NA = mo.ui.slider(
        0.02, 0.40, step=0.01, value=0.10, label="objective NA",
    )
    t2c_controls = mo.vstack([
        mo.md(
            "**Stage C — A lens with a pupil → imaging.**  "
            "Object spectrum forms at the pupil plane, gets multiplied by the "
            "pupil disk, then re-transforms to the image. Lower NA = smaller "
            "pupil = high frequencies clipped = blur."
        ),
        mo.hstack([t2c_preset, t2c_NA]),
    ])
    return t2c_NA, t2c_controls, t2c_preset


@app.cell
def _t2c_plot(
    N_LIVE,
    PIXEL_PITCH_UM,
    WAVELENGTH_UM,
    ft,
    ift,
    make_pupil,
    mo,
    na_to_cutoff_px,
    np,
    preset_object,
    t2c_NA,
    t2c_preset,
    three_pane,
):
    _N = N_LIVE
    _obj_real = preset_object(t2c_preset.value, _N)
    _obj = _obj_real.astype(np.complex128)
    _S = ft(_obj)
    _cutoff_px = na_to_cutoff_px(
        t2c_NA.value, WAVELENGTH_UM, PIXEL_PITCH_UM, _N,
    )
    _pupil = make_pupil(_N, _cutoff_px)
    _sampled = _pupil * _S
    _img = ift(_sampled)
    _fig = three_pane(
        _obj, _sampled, _img,
        modes=("magnitude", "magnitude", "magnitude"),
        titles=(
            f"object transmittance t = A·exp(iϕ)",
            f"pupil P · F{{t}}  (cutoff ≈ {_cutoff_px:.1f}px)",
            f"image  |F⁻¹{{P · F{{t}}}}|",
        ),
        pupil_radius_px=_cutoff_px,
        log_middle=True,
    )
    t2c_plot = show_fig(_fig)
    t2c_caption = mo.md(
        rf"""
        **What to notice.** As NA shrinks the cyan pupil ring tightens,
        clipping high frequencies; sharp edges blur. As NA grows you get more
        of the spectrum back and the image sharpens. **The resolution limit
        is the pupil radius**, not an asserted formula: $f_c = \mathrm{{NA}}/\lambda$.

        Two distinct complex functions live in this picture — keep them
        separate in your head:

        - **object transmittance** $t = A\,e^{{i\phi}}$ — lives in the *object plane*.
        - **lens pupil** $P$ — a disk (the aperture) times an optional aberration
          phase; lives in the *Fourier plane*.

        At the current setting the pupil cutoff is about
        **{na_to_cutoff_px(t2c_NA.value, WAVELENGTH_UM, PIXEL_PITCH_UM, _N):.1f}
        pixels** of the {_N}×{_N} spectrum.
        """
    )
    return t2c_caption, t2c_plot


# ===========================================================================
# TAB 3  —  Widget 1: tilted illumination shifts the spectrum
# ===========================================================================


@app.cell
def _t3w1_controls(ILLUM_NA_MAX, mo):
    t3w1_obj = mo.ui.dropdown(
        options=["letter A", "cross", "disk", "rectangle", "grating"],
        value="letter A", label="object",
    )
    t3w1_NA = mo.ui.slider(
        0.05, 0.30, step=0.01, value=0.10, label="objective NA",
    )
    t3w1_NAill_x = mo.ui.slider(
        -ILLUM_NA_MAX, ILLUM_NA_MAX, step=0.02, value=0.0,
        label="illumination NA · x",
    )
    t3w1_NAill_y = mo.ui.slider(
        -ILLUM_NA_MAX, ILLUM_NA_MAX, step=0.02, value=0.0,
        label="illumination NA · y",
    )
    t3w1_view = mo.ui.dropdown(
        options=["low-res image (intensity)", "low-res field amplitude"],
        value="low-res image (intensity)", label="left pane",
    )
    t3w1_controls = mo.vstack([
        mo.md(
            "**Widget 1 — Tilted illumination → spectral shift.**  "
            "Tilting the illumination multiplies the object by a plane wave "
            "(a phase ramp), which **shifts** the object's spectrum across "
            "the fixed pupil. The Tab-2 single-point result, running backwards."
        ),
        mo.hstack([t3w1_obj, t3w1_NA, t3w1_view]),
        mo.hstack([t3w1_NAill_x, t3w1_NAill_y]),
    ])
    return (
        t3w1_NA,
        t3w1_NAill_x,
        t3w1_NAill_y,
        t3w1_controls,
        t3w1_obj,
        t3w1_view,
    )


@app.cell
def _t3w1_plot(
    N_LIVE,
    PIXEL_PITCH_UM,
    WAVELENGTH_UM,
    ft,
    low_res_field,
    make_pupil,
    mo,
    na_to_cutoff_px,
    np,
    preset_object,
    t3w1_NA,
    t3w1_NAill_x,
    t3w1_NAill_y,
    t3w1_obj,
    t3w1_view,
    three_pane,
):
    _N = N_LIVE
    _obj = preset_object(t3w1_obj.value, _N).astype(np.complex128)
    _cutoff_px = na_to_cutoff_px(
        t3w1_NA.value, WAVELENGTH_UM, PIXEL_PITCH_UM, _N,
    )
    _pupil = make_pupil(_N, _cutoff_px)
    # The illumination tilt maps to a frequency shift of NA/λ (cycles/length).
    # In pixels on this grid: shift_px = NA * N * pitch / lambda.
    _kx_px = na_to_cutoff_px(
        t3w1_NAill_x.value, WAVELENGTH_UM, PIXEL_PITCH_UM, _N,
    )
    _ky_px = na_to_cutoff_px(
        t3w1_NAill_y.value, WAVELENGTH_UM, PIXEL_PITCH_UM, _N,
    )
    _k_shift = (int(round(_ky_px)), int(round(_kx_px)))
    _S = ft(_obj)
    _S_shifted = np.roll(_S, shift=_k_shift, axis=(0, 1))
    _pupil_sampled = _pupil * _S_shifted
    _lr_field = low_res_field(_obj, _pupil, _k_shift)
    if t3w1_view.value.startswith("low-res image"):
        _left_field = (np.abs(_lr_field) ** 2).astype(np.complex128)
        _left_title = "low-res intensity (camera sees)"
    else:
        _left_field = _lr_field
        _left_title = "low-res complex field |·|"

    _fig = three_pane(
        _left_field, _pupil_sampled, _obj,
        modes=("magnitude", "magnitude", "magnitude"),
        titles=(
            _left_title,
            f"object spectrum (shifted), pupil overlaid",
            f"object: {t3w1_obj.value}",
        ),
        pupil_radius_px=_cutoff_px,
        log_middle=True,
    )
    t3w1_plot = show_fig(_fig)
    t3w1_caption = mo.md(
        rf"""
        **What to notice.** As you slide illumination NA: the object spectrum
        **slides under the stationary pupil**. Frequencies that were outside
        the pupil at NA=0 can be brought inside — that's how a low-NA
        objective can "see" higher spatial frequencies one tilt at a time.

        At this setting the spectrum is shifted by **(Δky, Δkx) = ({_k_shift[0]}, {_k_shift[1]})
        px** on a {_N}×{_N} grid, while the pupil cutoff stays fixed at
        **{_cutoff_px:.1f} px**. The math is identical to Tab-2 Stage A — only
        the roles swap: in Tab 2 a *moving point* put a *phase ramp* on the
        Fourier side; here a *phase ramp* on the object side moves the
        spectrum.
        """
    )
    return t3w1_caption, t3w1_plot


# ===========================================================================
# TAB 3  —  Widget 2: synthetic aperture / k-space tiling
# ===========================================================================


@app.cell
def _t3w2_controls(mo):
    t3w2_obj = mo.ui.dropdown(
        options=["letter A", "cross", "grating", "disk", "rectangle"],
        value="letter A", label="object",
    )
    t3w2_NA = mo.ui.slider(
        0.05, 0.20, step=0.01, value=0.08, label="objective NA",
    )
    t3w2_NA_ill_max = mo.ui.slider(
        0.05, 0.40, step=0.02, value=0.30, label="max illumination NA",
    )
    t3w2_grid_side = mo.ui.slider(
        3, 11, step=2, value=7, label="LED grid side (n×n)",
    )
    t3w2_r_active = mo.ui.slider(
        0, 7, step=1, value=3, label="LEDs lit out to radius (steps)",
    )
    t3w2_controls = mo.vstack([
        mo.md(
            "**Widget 2 — Synthetic aperture.**  "
            "Each LED contributes a pupil-sized disk to an accumulating "
            "*synthesized* aperture. Slide the radius up: coverage grows, "
            "preview sharpens — **field of view stays fixed.** This is the "
            "FPM premise."
        ),
        mo.hstack([t3w2_obj, t3w2_NA, t3w2_NA_ill_max]),
        mo.hstack([t3w2_grid_side, t3w2_r_active]),
    ])
    return (
        t3w2_NA,
        t3w2_NA_ill_max,
        t3w2_controls,
        t3w2_grid_side,
        t3w2_obj,
        t3w2_r_active,
    )


@app.cell
def _t3w2_plot(
    Circle,
    N_LIVE,
    PIXEL_PITCH_UM,
    WAVELENGTH_UM,
    ft,
    ift,
    make_pupil,
    mo,
    na_to_cutoff_px,
    np,
    plt,
    preset_object,
    t3w2_NA,
    t3w2_NA_ill_max,
    t3w2_grid_side,
    t3w2_obj,
    t3w2_r_active,
):
    _N = N_LIVE
    _obj = preset_object(t3w2_obj.value, _N).astype(np.complex128)
    _S = ft(_obj)
    _cutoff_px = na_to_cutoff_px(
        t3w2_NA.value, WAVELENGTH_UM, PIXEL_PITCH_UM, _N,
    )
    _NA_max = float(t3w2_NA_ill_max.value)
    _side = int(t3w2_grid_side.value)
    _r_act = int(t3w2_r_active.value)

    # LED grid in NA units: -NA_max .. +NA_max
    _na_vals = np.linspace(-_NA_max, _NA_max, _side)
    _leds = []  # (NA_x, NA_y, k-shift_px_y, k-shift_px_x, dist_steps)
    _center = _side // 2
    for _i, _ny in enumerate(_na_vals):
        for _j, _nx in enumerate(_na_vals):
            _kx_px = na_to_cutoff_px(_nx, WAVELENGTH_UM, PIXEL_PITCH_UM, _N)
            _ky_px = na_to_cutoff_px(_ny, WAVELENGTH_UM, PIXEL_PITCH_UM, _N)
            _r = max(abs(_i - _center), abs(_j - _center))
            _leds.append((_nx, _ny, _ky_px, _kx_px, _r))

    # Build the synthesized-aperture mask (union of pupil disks at each lit LED).
    _agg_mask = np.zeros((_N, _N), dtype=bool)
    _cy, _cx = _N // 2, _N // 2
    _yy, _xx = np.ogrid[:_N, :_N]
    _lit_count = 0
    for (_nx, _ny, _ky, _kxp, _step) in _leds:
        if _step <= _r_act:
            _disk = ((_yy - (_cy + int(round(_ky)))) ** 2
                     + (_xx - (_cx + int(round(_kxp)))) ** 2) <= _cutoff_px ** 2
            _agg_mask |= _disk
            _lit_count += 1

    # "Best achievable" preview: ground-truth spectrum × synthesized mask.
    _S_synth = _S * _agg_mask
    _preview = np.abs(ift(_S_synth))

    # Plot: LED grid, synthesized aperture, reconstruction preview.
    _fig, _ax = plt.subplots(1, 3, figsize=(11.5, 4.0),
                             constrained_layout=True)
    # LED grid
    _ax[0].set_title("LED grid (lit = blue)", fontsize=10)
    for (_nx, _ny, _ky, _kxp, _step) in _leds:
        _is_lit = _step <= _r_act
        _ax[0].plot(_nx, _ny, "o",
                    color=("tab:blue" if _is_lit else "lightgray"),
                    markersize=8)
    # mark BF circle vs DF outside
    _bf_circle = Circle((0, 0), t3w2_NA.value, fill=False,
                        edgecolor="green", linestyle="--",
                        label=f"BF: |NA|≤{t3w2_NA.value}")
    _ax[0].add_patch(_bf_circle)
    _ax[0].set_xlim(-_NA_max * 1.15, _NA_max * 1.15)
    _ax[0].set_ylim(-_NA_max * 1.15, _NA_max * 1.15)
    _ax[0].set_xlabel("NA (x)")
    _ax[0].set_ylabel("NA (y)")
    _ax[0].set_aspect("equal")
    _ax[0].legend(loc="upper right", fontsize=8)
    # Synthesized aperture overlaid on the *log* spectrum, for context
    _ax[1].imshow(np.log1p(np.abs(_S)), cmap="gray")
    _ax[1].imshow(_agg_mask, cmap="Blues", alpha=0.35)
    _ax[1].set_title(f"synthesized aperture ({_lit_count} LEDs)", fontsize=10)
    _ax[1].set_xticks([])
    _ax[1].set_yticks([])
    # Reconstruction preview
    _ax[2].imshow(_preview, cmap="gray")
    _ax[2].set_title("achievable resolution preview", fontsize=10)
    _ax[2].set_xticks([])
    _ax[2].set_yticks([])
    t3w2_plot = show_fig(_fig)

    # Synthetic NA estimate
    _syn_NA = t3w2_NA.value + (_r_act / max(_center, 1)) * _NA_max
    t3w2_caption = mo.md(
        rf"""
        **What to notice.** Each LED lights up a *pupil-sized* patch of the
        synthesized spectrum. Slide the *LEDs lit out to radius* up: coverage
        grows outward, and the right-hand preview gains resolution — but the
        **field of view (the image size) doesn't change.** That's the FPM
        bargain.

        **Synthetic NA ≈ objective NA + max illumination NA.**
        At the current setting that's about
        **{_syn_NA:.2f}**, vs. the bare objective's
        **{t3w2_NA.value:.2f}** — a ~**{_syn_NA / max(t3w2_NA.value, 1e-6):.1f}×**
        resolution gain in principle.

        The catch (set up here, paid for in Tab 5): bigger or more peripheral
        LEDs push acquisitions into the dim *darkfield* regime where SNR is
        a struggle.
        """
    )
    return t3w2_caption, t3w2_plot


# ===========================================================================
# TAB 3 assembly
# ===========================================================================


@app.cell
def _t3_assembly(
    mo,
    t3w1_controls, t3w1_plot, t3w1_caption,
    t3w2_controls, t3w2_plot, t3w2_caption,
):
    tab3 = mo.vstack([
        mo.md(
            "## Tab 3 — FPM premise: tilted illumination shifts the spectrum\n\n"
            "Take everything from Tabs 1–2 and notice one re-framing: "
            "**tilting the illumination = multiplying the object by a phase "
            "ramp = shifting its spectrum** across the (fixed) pupil. Now "
            "the same pupil can sample frequencies that were originally out "
            "of reach — and many tilts together synthesize a much larger "
            "effective aperture."
        ),
        t3w1_controls, t3w1_plot, t3w1_caption,
        mo.md("---"),
        t3w2_controls, t3w2_plot, t3w2_caption,
    ])
    return (tab3,)


# ===========================================================================
# TAB 4 / TAB 5 shared setup  —  ground truth + reconstruction geometry
# ===========================================================================


@app.cell
def _t4_groundtruth(N_RECON, np):
    def make_ground_truth(N=N_RECON):
        """A small complex object: amplitude letters + phase blob."""
        y = np.arange(N) - N // 2
        yy, xx = np.meshgrid(y, y, indexing="ij")
        # Amplitude: a "+" shape, smoothed
        amp = np.ones((N, N))
        amp -= 0.6 * ((np.abs(xx) <= 10) & (np.abs(yy) <= 2))
        amp -= 0.6 * ((np.abs(yy) <= 10) & (np.abs(xx) <= 2))
        # Phase: smooth Gaussian blob
        phi = 1.2 * np.exp(-((yy - 4) ** 2 + (xx + 4) ** 2) / (2 * 6.0 ** 2))
        phi -= 0.9 * np.exp(-((yy + 6) ** 2 + (xx - 6) ** 2) / (2 * 5.0 ** 2))
        return amp.astype(np.float64), phi.astype(np.float64)
    return (make_ground_truth,)


@app.cell
def _t4_geometry(N_RECON):
    # Geometry chosen so a 3x3 LED grid gives meaningful overlap at N=128.
    # The objective cutoff radius (in pixels of the N x N spectrum) and the
    # LED-grid spacing (also pixels) together determine the overlap.
    PUPIL_CUTOFF_PX = 16            # objective NA in spectrum pixels
    LED_SPACING_PX_DEFAULT = 10     # adjacent LED ⇒ ~60% pupil overlap
    LED_GRID_SIDE = 3               # 3 x 3 by spec
    return LED_GRID_SIDE, LED_SPACING_PX_DEFAULT, PUPIL_CUTOFF_PX


# ===========================================================================
# TAB 4  —  Why overlap buys resolution (stateless / reactive)
#   The camera measures intensity (phase discarded). Overlapping Fourier
#   windows over-determine the problem just enough to solve for that missing
#   phase, and the recovered phase is what synthesizes a larger aperture =
#   resolution. One slider (LED array density) drives the whole story.
# ===========================================================================


@app.cell
def _t4_config():
    # Tab-4 geometry (independent of the Tab-5 reconstruction constants).
    # Low-NA objective (small pupil) so the bare image is blurry; a fixed
    # synthetic aperture (the resolution ceiling) is reached by the LED array.
    T4_R_OBJ = 11             # objective pupil cutoff radius (px in k-space)
    T4_R_MAX = 26             # outermost LED shift (px) -> fixed synthetic NA
    T4_R_SYNTH = T4_R_OBJ + T4_R_MAX
    T4_ITERS = 10             # reconstruction iterations
    T4_PHOTONS = 1.5e3        # mild shot noise so weak overlap visibly struggles
    return T4_ITERS, T4_PHOTONS, T4_R_MAX, T4_R_OBJ, T4_R_SYNTH


@app.cell
def _t4_target(N_RECON, np):
    def _star(N, spokes, r_out, rot=0.0, r_in=3):
        y = np.arange(N) - N // 2
        yy, xx = np.meshgrid(y, y, indexing="ij")
        th = np.arctan2(yy, xx) - rot
        rho = np.hypot(yy, xx)
        return ((0.5 + 0.5 * np.sign(np.cos(spokes * th)))
                * ((rho < r_out) & (rho > r_in)))

    def mixed_target(N=N_RECON):
        """A compact mixed object: a Siemens-star *resolution* target in
        amplitude (absorption), plus a rotated phase Siemens star. Amplitude
        carries the resolution story; phase is the harder, hidden channel that
        a conventional intensity image never sees."""
        y = np.arange(N) - N // 2
        yy, xx = np.meshgrid(y, y, indexing="ij")
        body = (yy ** 2 + xx ** 2) <= 50 ** 2
        amp = body * (1.0 - 0.6 * _star(N, 16, 50))
        phi = body * (1.2 * _star(N, 12, 50, rot=np.pi / 12))
        return amp.astype(np.float64), phi.astype(np.float64)

    t4_amp_gt, t4_phi_gt = mixed_target()
    t4_obj = t4_amp_gt * np.exp(1j * t4_phi_gt)
    return t4_amp_gt, t4_obj, t4_phi_gt


@app.cell
def _t4_controls(mo):
    t4_nside = mo.ui.slider(
        3, 7, step=1, value=3,
        label="LEDs across the array  (more LEDs = more Fourier-window overlap)",
    )
    t4_controls = mo.vstack([
        mo.md("**Turn up the overlap.** Start sparse (left), predict what the "
              "reconstruction can recover, then add LEDs and watch."),
        t4_nside,
    ])
    return t4_controls, t4_nside


@app.cell
def _t4_setup(
    T4_PHOTONS,
    T4_R_MAX,
    T4_R_OBJ,
    T4_R_SYNTH,
    make_pupil,
    np,
    simulate_intensity,
    t4_nside,
    t4_obj,
):
    _N = t4_obj.shape[0]
    _n = int(t4_nside.value)
    # Fixed synthetic aperture: n x n LEDs spanning [-R_MAX, R_MAX].
    _pos = np.linspace(-T4_R_MAX, T4_R_MAX, _n)
    t4_leds = [(int(round(_ky)), int(round(_kx))) for _ky in _pos for _kx in _pos]
    _spacing = (2 * T4_R_MAX / (_n - 1)) if _n > 1 else 0.0
    t4_overlap = max(0.0, 1.0 - _spacing / (2 * T4_R_OBJ))

    t4_pupil = make_pupil(_N, T4_R_OBJ)

    # Measurement stack, with mild deterministic shot noise.
    _rng = np.random.default_rng(0)
    _stack = []
    for _k in t4_leds:
        _img = simulate_intensity(t4_obj, t4_pupil, _k)
        _sc = _img.max() + 1e-12
        _img = _rng.poisson(_img / _sc * T4_PHOTONS) / T4_PHOTONS * _sc
        _stack.append(_img)
    t4_measurements = np.stack(_stack, axis=0)

    # Redundancy map: how many LED windows measure each spatial frequency.
    _mask = (np.abs(t4_pupil) > 0).astype(np.float64)
    _cov = np.zeros((_N, _N))
    for (_ky, _kx) in t4_leds:
        _cov += np.roll(_mask, shift=(-_ky, -_kx), axis=(0, 1))
    t4_coverage = _cov

    # Back-of-envelope constraint count (the 'why').
    _yy, _xx = np.ogrid[:_N, :_N]
    _r2 = (_yy - _N // 2) ** 2 + (_xx - _N // 2) ** 2
    _n_freq = int((_r2 <= T4_R_SYNTH ** 2).sum())     # frequencies to synthesize
    _pix_per_img = int((_r2 <= T4_R_OBJ ** 2).sum())  # measured values per image
    t4_counts = {
        "unknowns_mag": _n_freq,
        "unknowns_phase": 2 * _n_freq,
        "measurements": len(t4_leds) * _pix_per_img,
        "n_leds": len(t4_leds),
    }
    return t4_counts, t4_coverage, t4_leds, t4_measurements, t4_overlap, t4_pupil


@app.cell
def _t4_recon(T4_ITERS, ft, ift, np, t4_leds, t4_measurements, t4_pupil):
    """Alternating-projection recovery (reactive: reruns when the slider moves)."""
    _mask = np.abs(t4_pupil) > 0
    _order = sorted(range(len(t4_leds)),
                    key=lambda i: t4_leds[i][0] ** 2 + t4_leds[i][1] ** 2)
    _spec = ft(np.sqrt(np.clip(t4_measurements[_order[0]], 0.0, None)
                       ).astype(np.complex128))
    for _ in range(T4_ITERS):
        for _i in _order:
            _ky, _kx = t4_leds[_i]
            _S = np.roll(_spec, shift=(_ky, _kx), axis=(0, 1))
            _low = ift(t4_pupil * _S)
            _newlow = (np.sqrt(np.clip(t4_measurements[_i], 0.0, None))
                       * np.exp(1j * np.angle(_low)))
            _Sn = _S.copy()
            _Sn[_mask] = ft(_newlow)[_mask]
            _spec = np.roll(_Sn, shift=(-_ky, -_kx), axis=(0, 1))
    t4_recovered = ift(_spec)
    return (t4_recovered,)


@app.cell
def _t4_why_plot(
    Circle,
    T4_R_OBJ,
    T4_R_SYNTH,
    mo,
    plt,
    show_fig,
    t4_counts,
    t4_coverage,
    t4_overlap,
):
    _N = t4_coverage.shape[0]
    _fig, _ax = plt.subplots(1, 2, figsize=(7.6, 3.7), constrained_layout=True)

    _im = _ax[0].imshow(t4_coverage, cmap="magma")
    _ax[0].set_title(f"k-space redundancy — LEDs measuring\n"
                     f"each frequency (overlap ≈ {t4_overlap:.0%})",
                     fontsize=9)
    _fig.colorbar(_im, ax=_ax[0], fraction=0.046, shrink=0.85,
                  label="× measured")
    for _r, _col in [(T4_R_SYNTH, "cyan"), (T4_R_OBJ, "white")]:
        _ax[0].add_patch(Circle((_N / 2 - 0.5, _N / 2 - 0.5), _r, fill=False,
                                edgecolor=_col, lw=1.0, ls="--"))
    _ax[0].set_xticks([])
    _ax[0].set_yticks([])

    _u = t4_counts["unknowns_phase"]
    _m = t4_counts["measurements"]
    _ax[1].barh([2, 1], [t4_counts["unknowns_mag"], _u],
                color=["#9aa7ff", "#3344cc"])
    _ax[1].barh([0], [_m], color=("#2ca02c" if _m >= _u else "#d62728"))
    _ax[1].set_yticks([2, 1, 0])
    _ax[1].set_yticklabels(["unknowns\n(ignore phase)", "unknowns\n(+ phase!)",
                            "measurements"], fontsize=8)
    _ax[1].set_xlabel("count (real numbers)", fontsize=8)
    _ax[1].set_title("counting the constraints", fontsize=9)
    t4_why_fig = show_fig(_fig)

    _verdict = (
        "**over-determined** — on paper there are now more measurements than "
        "unknowns. That's *necessary* for recovery, but (as the picture below "
        "shows) it's a floor, not a promise — the phase needs comfortably more "
        "overlap than the bare count to settle"
        if _m >= _u else
        "**under-determined** — fewer measurements than unknowns; the phase "
        "cannot be pinned down at all"
    )
    t4_why = mo.vstack([
        t4_why_fig,
        mo.md(
            f"Intensity = |field|², so each of the "
            f"**{t4_counts['unknowns_mag']:,}** frequencies you want is really "
            f"**two** unknowns — a magnitude *and a phase* "
            f"(**{_u:,}** real unknowns). Overlapping windows re-measure shared "
            f"frequencies ({t4_counts['n_leds']} LEDs → **{_m:,}** "
            f"measurements); that redundancy is what supplies the missing "
            f"phase.\n\n{_verdict}."
        ),
    ])
    return (t4_why,)


@app.cell
def _t4_payoff_plot(
    T4_R_OBJ,
    ft,
    ift,
    make_pupil,
    np,
    plt,
    show_fig,
    t4_amp_gt,
    t4_obj,
    t4_overlap,
    t4_phi_gt,
    t4_recovered,
):
    _N = t4_amp_gt.shape[0]
    _bare = np.abs(ift(make_pupil(_N, T4_R_OBJ) * ft(t4_obj)))
    # Align recovered global phase to the truth for a fair phase comparison.
    _rec = t4_recovered * np.exp(
        -1j * np.angle(np.sum(t4_recovered * np.conj(t4_obj))))

    def _phase_rgba(field):
        _mag = np.abs(field)
        _cmap = plt.get_cmap("twilight")
        _rgba = _cmap((np.angle(field) + np.pi) / (2 * np.pi))
        _rgba[..., 3] = np.clip((_mag / (_mag.max() + 1e-12)) ** 0.6, 0, 1)
        return _rgba

    _fig, _ax = plt.subplots(2, 3, figsize=(9.6, 6.5), constrained_layout=True)
    _ax[0, 0].imshow(t4_amp_gt, cmap="gray", vmin=0, vmax=1.1)
    _ax[0, 1].imshow(np.abs(_rec), cmap="gray", vmin=0, vmax=1.1)
    _ax[0, 2].imshow(_bare, cmap="gray")
    _ax[1, 0].imshow(_phase_rgba(t4_obj))
    _ax[1, 1].imshow(_phase_rgba(_rec))
    _ax[1, 2].text(0.5, 0.5,
                   "a conventional microscope\nrecords no phase at all —\n"
                   "recovering this channel is\nthe whole point of FPM",
                   ha="center", va="center", fontsize=8.5, style="italic",
                   transform=_ax[1, 2].transAxes)

    for _j, _t in enumerate(["ground truth",
                             f"FPM recovered  (overlap ≈ {t4_overlap:.0%})",
                             "conventional microscope"]):
        _ax[0, _j].set_title(_t, fontsize=9)
    _ax[0, 0].set_ylabel("amplitude\n(resolution)", fontsize=9)
    _ax[1, 0].set_ylabel("phase\n(the hard part)", fontsize=9)
    for _a in _ax.ravel():
        _a.set_xticks([])
        _a.set_yticks([])
    _ax[1, 2].axis("off")
    t4_payoff = show_fig(_fig)
    return (t4_payoff,)


@app.cell
def _t4_assembly(mo, t4_controls, t4_payoff, t4_why):
    tab4 = mo.vstack([
        mo.md(
            "## Tab 4 — Why overlap buys resolution\n\n"
            "Your camera records only **intensity** — it throws away the "
            "**phase** of the light (Tab 1, Widget 2: scramble an image's phase "
            "and its structure is destroyed — phase is where the information "
            "lives).\n\n"
            "So FPM has a puzzle. Each photo captures a small disk of the "
            "object's spectrum, but *without its phase*. Lay those disks side by "
            "side and you still can't stitch them — the phase that aligns one "
            "disk to the next is missing. The fix is **overlap**: make "
            "neighbouring measurements share spectral territory. Those shared, "
            "redundant measurements **over-determine** the problem enough to "
            "**solve for the phase you never recorded** — and the recovered "
            "phase is what fuses the disks into one large synthetic aperture. "
            "That is where the extra **resolution** comes from.\n\n"
            "The target below is a mixed object — a resolution star in "
            "**amplitude** and a second star hidden in **phase**, like the real "
            "samples the lab images."
        ),
        t4_controls,
        mo.md("### Why it works — the redundancy that pays for phase"),
        t4_why,
        mo.md("### What it buys — resolution (and the phase you couldn't see)"),
        t4_payoff,
        mo.md(
            "**Predict, then slide — and watch the two channels separately.** "
            "With the fewest LEDs the windows barely overlap: the problem is "
            "under-determined and both channels are junk. Add LEDs and the "
            "**amplitude** sharpens first — the spokes resolve toward the "
            "centre, the resolution gain your microscope was after. The "
            "**phase** lags: right around where the counter flips to "
            "*over-determined* (~40% overlap) it's only a faint, unstable "
            "ghost, and it doesn't settle until the overlap is comfortably "
            "higher (~60%). That gap is the lesson: **more knowns than unknowns "
            "is necessary, not sufficient** — phase retrieval is a hard, "
            "non-convex problem, and overlap has to buy not just coverage but "
            "the redundancy that makes solving for phase *stable*. Same field "
            "of view, more resolution, and a whole channel of information a "
            "conventional microscope never records — all bought with overlap."
        ),
    ])
    return (tab4,)


# ===========================================================================
# Tab assembly
# ===========================================================================


# ===========================================================================
# TAB 2  —  Stage D: Object library (amplitude / phase / mixed)
# ===========================================================================


@app.cell
def _t2d_controls(mo):
    t2d_kind = mo.ui.dropdown(
        options=["pure absorption", "pure phase (cell-like)", "mixed"],
        value="pure phase (cell-like)", label="object kind",
    )
    t2d_NA = mo.ui.slider(0.05, 0.40, step=0.01, value=0.15,
                          label="objective NA")
    t2d_controls = mo.vstack([
        mo.md(
            "**Stage D — Object library.**  "
            "Real specimens come in three flavors: **absorbing** (varying A, "
            "ϕ=0), **phase-only** (A≈1, varying ϕ — most cells), and **mixed**. "
            "The teaser: in a plain intensity image a pure phase object is "
            "nearly *invisible* — even though its spectrum is rich."
        ),
        mo.hstack([t2d_kind, t2d_NA]),
    ])
    return t2d_NA, t2d_controls, t2d_kind


@app.cell
def _t2d_plot(
    N_LIVE,
    PIXEL_PITCH_UM,
    WAVELENGTH_UM,
    absorption_object,
    cell_like_phase_object,
    complex_object,
    ft,
    ift,
    make_pupil,
    mixed_object,
    mo,
    na_to_cutoff_px,
    np,
    plt,
    t2d_NA,
    t2d_kind,
):
    _N = N_LIVE
    _kind = t2d_kind.value
    if _kind == "pure absorption":
        _amp = absorption_object(_N)
        _phi = np.zeros_like(_amp)
        _label = "pure absorption: A varies, ϕ = 0"
    elif _kind == "pure phase (cell-like)":
        _amp, _phi = cell_like_phase_object(_N)
        _label = "pure phase: A ≈ 1, ϕ varies"
    else:
        _amp, _phi = mixed_object(_N)
        _label = "mixed: A and ϕ both vary"
    _obj = complex_object(amplitude=_amp, phase=_phi)
    _cutoff_px = na_to_cutoff_px(
        t2d_NA.value, WAVELENGTH_UM, PIXEL_PITCH_UM, _N,
    )
    _pupil = make_pupil(_N, _cutoff_px)
    _S = ft(_obj)
    _img_field = ift(_pupil * _S)
    _intensity = np.abs(_img_field) ** 2

    _fig, _ax = plt.subplots(1, 3, figsize=(10.5, 3.7),
                             constrained_layout=True)
    _ax[0].imshow(_amp, cmap="gray", vmin=0, vmax=1.05)
    _ax[0].set_title("object amplitude A", fontsize=10)
    _ax[1].imshow(_phi, cmap="twilight", vmin=-2.0, vmax=2.0)
    _ax[1].set_title("object phase ϕ (rad)", fontsize=10)
    _ax[2].imshow(_intensity, cmap="gray")
    _ax[2].set_title("camera intensity = |F⁻¹{P·F{t}}|²", fontsize=10)
    for _a in _ax:
        _a.set_xticks([])
        _a.set_yticks([])
    t2d_plot = show_fig(_fig)
    if _kind == "pure phase (cell-like)":
        _msg = (
            "**Look at the right pane.** A blob of phase variation — but the "
            "intensity image is *nearly uniform*. Phase delays don't change "
            "how much light reaches the sensor. **Recovering this needs phase** "
            "— see Tab 4."
        )
    elif _kind == "pure absorption":
        _msg = (
            "An absorption object behaves intuitively: dark regions stay "
            "dark in the intensity image. Here the camera *does* see the "
            "structure directly."
        )
    else:
        _msg = (
            "Mixed objects are typical. The amplitude features show up in "
            "intensity; the phase features are mostly hidden until we "
            "recover them with FPM."
        )
    t2d_caption = mo.md(_label + ". " + _msg)
    return t2d_caption, t2d_plot


# ===========================================================================
# TAB 2  —  Stage E: Aberrations (Zernike sliders, off by default)
# ===========================================================================


@app.cell
def _t2e_controls(mo):
    t2e_on = mo.ui.switch(value=False, label="enable aberration sliders")
    t2e_defocus = mo.ui.slider(-2.5, 2.5, step=0.1, value=0.0,
                               label="defocus")
    t2e_astig_x = mo.ui.slider(-2.5, 2.5, step=0.1, value=0.0,
                               label="astigmatism (x)")
    t2e_astig_y = mo.ui.slider(-2.5, 2.5, step=0.1, value=0.0,
                               label="astigmatism (y)")
    t2e_coma_x = mo.ui.slider(-2.5, 2.5, step=0.1, value=0.0,
                              label="coma (x)")
    t2e_coma_y = mo.ui.slider(-2.5, 2.5, step=0.1, value=0.0,
                              label="coma (y)")
    t2e_spherical = mo.ui.slider(-2.5, 2.5, step=0.1, value=0.0,
                                 label="spherical")
    t2e_NA = mo.ui.slider(0.05, 0.30, step=0.01, value=0.15,
                          label="objective NA")
    t2e_controls = mo.vstack([
        mo.md(
            "**Stage E — Aberrations (off by default).**  "
            "Cheap lenses live here. Toggle on and dial in low-order "
            "Zernike phase terms; watch the PSF widen and the image "
            "degrade. This sets up Tab 5's **EPRY** toggle, which estimates "
            "and corrects the pupil from data."
        ),
        t2e_on,
        mo.hstack([t2e_NA, t2e_defocus, t2e_spherical]),
        mo.hstack([t2e_astig_x, t2e_astig_y]),
        mo.hstack([t2e_coma_x, t2e_coma_y]),
    ])
    return (
        t2e_NA,
        t2e_astig_x,
        t2e_astig_y,
        t2e_coma_x,
        t2e_coma_y,
        t2e_controls,
        t2e_defocus,
        t2e_on,
        t2e_spherical,
    )


@app.cell
def _t2e_plot(
    N_LIVE,
    PIXEL_PITCH_UM,
    WAVELENGTH_UM,
    absorption_object,
    complex_object,
    ft,
    ift,
    make_pupil,
    mo,
    na_to_cutoff_px,
    np,
    plt,
    t2e_NA,
    t2e_astig_x,
    t2e_astig_y,
    t2e_coma_x,
    t2e_coma_y,
    t2e_defocus,
    t2e_on,
    t2e_spherical,
    zernike_phase,
):
    _N = N_LIVE
    _cutoff_px = na_to_cutoff_px(
        t2e_NA.value, WAVELENGTH_UM, PIXEL_PITCH_UM, _N,
    )
    if t2e_on.value:
        _coefs = {
            "defocus":   float(t2e_defocus.value),
            "astig_x":   float(t2e_astig_x.value),
            "astig_y":   float(t2e_astig_y.value),
            "coma_x":    float(t2e_coma_x.value),
            "coma_y":    float(t2e_coma_y.value),
            "spherical": float(t2e_spherical.value),
        }
        _ab_phase = zernike_phase(_N, _cutoff_px, _coefs)
    else:
        _ab_phase = None
    _pupil = make_pupil(_N, _cutoff_px, aberration_phase=_ab_phase)
    _psf = np.abs(ift(_pupil)) ** 2
    _amp = absorption_object(_N)
    _obj = complex_object(amplitude=_amp, phase=np.zeros_like(_amp))
    _img = np.abs(ift(_pupil * ft(_obj))) ** 2

    _fig, _ax = plt.subplots(1, 3, figsize=(10.5, 3.7),
                             constrained_layout=True)
    if t2e_on.value:
        _phase_img = np.angle(_pupil)
        _mask = np.abs(_pupil) > 0
        _phase_img = np.where(_mask, _phase_img, np.nan)
        _ax[0].imshow(_phase_img, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        _ax[0].set_title("pupil phase (Zernike)", fontsize=10)
    else:
        _ax[0].imshow(np.abs(_pupil), cmap="gray")
        _ax[0].set_title("pupil (ideal)", fontsize=10)
    _cy, _cx = _N // 2, _N // 2
    _hw = max(int(2 * _cutoff_px) + 8, 24)
    _psf_crop = _psf[_cy - _hw:_cy + _hw, _cx - _hw:_cx + _hw]
    _ax[1].imshow(np.log1p(_psf_crop / (_psf_crop.max() + 1e-9) * 1e3),
                  cmap="hot")
    _ax[1].set_title("PSF (log, central crop)", fontsize=10)
    _ax[2].imshow(_img, cmap="gray")
    _ax[2].set_title("image with aberration", fontsize=10)
    for _a in _ax:
        _a.set_xticks([])
        _a.set_yticks([])
    t2e_plot = show_fig(_fig)
    t2e_caption = mo.md(
        "**What to notice.** Defocus → a soft ring-like PSF, blurring. "
        "Astigmatism → asymmetric (line-like) PSF. Coma → tail trailing off "
        "the PSF in one direction. Spherical → halos. Each scrambles the "
        "image in a characteristic way — and each can be *estimated* from "
        "data, which is what Tab 5's pupil-recovery toggle does."
    )
    return t2e_caption, t2e_plot


@app.cell
def _t2_assembly(
    mo,
    t2a_controls, t2a_plot, t2a_caption,
    t2b_controls, t2b_plot, t2b_caption,
    t2c_controls, t2c_plot, t2c_caption,
    t2d_controls, t2d_plot, t2d_caption,
    t2e_controls, t2e_plot, t2e_caption,
):
    tab2 = mo.vstack([
        mo.md(
            "## Tab 2 — The central dogma, built from one point\n\n"
            "We're going to construct, by superposition, the central result of "
            "imaging: **the image is the inverse transform of (object spectrum × "
            "lens pupil)**, and the camera records its modulus squared. Nothing "
            "below is asserted — each stage builds the next."
        ),
        t2a_controls, t2a_plot, t2a_caption,
        mo.md("---"),
        t2b_controls, t2b_plot, t2b_caption,
        mo.md("---"),
        t2c_controls, t2c_plot, t2c_caption,
        mo.md("---"),
        t2d_controls, t2d_plot, t2d_caption,
        mo.md("---"),
        t2e_controls, t2e_plot, t2e_caption,
    ])
    return (tab2,)


# ===========================================================================
# TAB 5  —  Synthesis: the principles in work clothes
# ===========================================================================


@app.cell
def _t5_controls(mo):
    t5_noise_exposure = mo.ui.slider(
        0.05, 5.0, step=0.05, value=1.0, label="exposure × (lower ⇒ noisier)",
    )
    t5_read_noise = mo.ui.slider(
        0.0, 0.5, step=0.01, value=0.05, label="read-noise σ",
    )
    t5_aberration = mo.ui.slider(
        0.0, 2.0, step=0.05, value=0.0, label="aberration (rad, defocus+coma)",
    )
    t5_epry = mo.ui.switch(value=False, label="enable pupil recovery (EPRY)")
    t5_led_miss = mo.ui.slider(
        0.0, 4.0, step=0.1, value=0.0, label="LED misposition σ (px)",
    )
    t5_partial_coh = mo.ui.slider(
        0.0, 3.0, step=0.1, value=0.0,
        label="partial coherence (pupil-edge blur σ)",
    )
    t5_iters = mo.ui.slider(2, 40, step=1, value=15, label="iterations")
    t5_run = mo.ui.button(label="Run reconstruction")
    t5_controls = mo.vstack([
        mo.md("**Perturbation toggles.** Each is labeled with the earlier "
              "principle it stresses."),
        mo.hstack([t5_noise_exposure, t5_read_noise]),
        mo.hstack([t5_aberration, t5_epry]),
        mo.hstack([t5_led_miss, t5_partial_coh]),
        mo.hstack([t5_iters, t5_run]),
    ])
    return (
        t5_aberration,
        t5_controls,
        t5_epry,
        t5_iters,
        t5_led_miss,
        t5_noise_exposure,
        t5_partial_coh,
        t5_read_noise,
        t5_run,
    )


@app.cell
def _t5_setup(
    LED_GRID_SIDE,
    PUPIL_CUTOFF_PX,
    complex_object,
    ft,
    make_ground_truth,
    make_pupil,
    np,
    simulate_intensity,
    t5_aberration,
    t5_led_miss,
    t5_partial_coh,
    t5_noise_exposure,
    t5_read_noise,
    zernike_phase,
):
    """Rebuild ground truth + perturbed measurements when controls change."""
    amp_gt5, phi_gt5 = make_ground_truth()
    obj_gt5 = complex_object(amplitude=amp_gt5, phase=phi_gt5)
    _N = obj_gt5.shape[0]

    # True pupil with hidden aberration (the simulator's reality).
    _ab = float(t5_aberration.value)
    _ab_phase = zernike_phase(
        _N, PUPIL_CUTOFF_PX,
        {"defocus": 0.7 * _ab, "coma_x": 0.5 * _ab},
    )
    pupil_true = make_pupil(_N, PUPIL_CUTOFF_PX, aberration_phase=_ab_phase)

    # Partial-coherence approximation: smooth the pupil's hard edge slightly,
    # which damps frequencies near the cutoff.
    _pc = float(t5_partial_coh.value)
    if _pc > 0.0:
        _yy, _xx = np.meshgrid(
            np.arange(_N) - _N // 2, np.arange(_N) - _N // 2, indexing="ij",
        )
        _k = np.exp(-(_yy ** 2 + _xx ** 2) / (2 * _pc ** 2))
        _k /= _k.sum()
        # Convolve |pupil| with the Gaussian
        _mag_f = np.fft.fft2(np.fft.fftshift(np.abs(pupil_true)))
        _k_f = np.fft.fft2(np.fft.fftshift(_k))
        _smoothed = np.real(np.fft.ifftshift(np.fft.ifft2(_mag_f * _k_f)))
        pupil_true = _smoothed * np.exp(1j * np.angle(pupil_true))

    # LED grid (3 x 3), spacing 10 px to match Tab 4 geometry
    _side = LED_GRID_SIDE
    _coords1d = np.arange(-(_side // 2), _side // 2 + 1) * 10
    led_positions5 = []
    _ctr = _side // 2
    for _i, _ky in enumerate(_coords1d):
        for _j, _kx in enumerate(_coords1d):
            _step = max(abs(_i - _ctr), abs(_j - _ctr))
            led_positions5.append((_ky, _kx, _step))
    led_positions5.sort(key=lambda t: (t[2], abs(t[0]) + abs(t[1])))

    # Hidden ground-truth LED positions: assumed grid plus Gaussian jitter.
    _rng = np.random.default_rng(7)
    _miss_sigma = float(t5_led_miss.value)
    led_true5 = []
    for (ky, kx, step) in led_positions5:
        _dy = _rng.standard_normal() * _miss_sigma if _miss_sigma > 0 else 0.0
        _dx = _rng.standard_normal() * _miss_sigma if _miss_sigma > 0 else 0.0
        led_true5.append((ky + _dy, kx + _dx, step))

    # Forward measurements with noise. 'exposure' scales the photon budget.
    _exposure = float(t5_noise_exposure.value)
    _photons = 5e3 * _exposure
    _read_sigma = float(t5_read_noise.value)
    meas5_clean = np.stack([
        simulate_intensity(
            obj_gt5, pupil_true,
            (int(round(ky)), int(round(kx))),
        ) for (ky, kx, _) in led_true5
    ])
    _scale = meas5_clean.max() + 1e-9
    _signal = meas5_clean / _scale * _photons
    _noisy = _rng.poisson(np.clip(_signal, 0, None)).astype(np.float64)
    _noisy += _rng.standard_normal(_noisy.shape) * (_read_sigma * _photons)
    measurements5 = np.clip(_noisy, 0.0, None) * (_scale / _photons)

    # Initialize reconstructor with central-LED amplitude and ideal pupil.
    init_field5 = np.sqrt(
        np.clip(measurements5[0], 0.0, None)
    ).astype(np.complex128)
    init_spectrum5 = ft(init_field5)
    pupil_assumed = make_pupil(_N, PUPIL_CUTOFF_PX)
    return (
        amp_gt5,
        init_spectrum5,
        led_positions5,
        measurements5,
        obj_gt5,
        phi_gt5,
        pupil_assumed,
    )


@app.cell
def _t5_run_recon(
    PUPIL_CUTOFF_PX,
    ft,
    ift,
    init_spectrum5,
    led_positions5,
    measurements5,
    np,
    pupil_assumed,
    t5_epry,
    t5_iters,
    t5_run,
):
    """Full reconstruction sweep. Re-runs when any input changes."""
    _ = t5_run.value
    _spec = init_spectrum5.copy()
    _pupil = pupil_assumed.copy()
    _epry_on = bool(t5_epry.value)
    _mask = np.abs(pupil_assumed) > 0
    _alpha, _beta = 1.0, 1.0
    _err_hist5 = []
    _n_leds = len(led_positions5)
    for _it in range(int(t5_iters.value)):
        for _idx, (_ky_raw, _kx_raw, _step) in enumerate(led_positions5):
            _ky = int(round(_ky_raw))
            _kx = int(round(_kx_raw))
            _S_shifted = np.roll(_spec, shift=(_ky, _kx), axis=(0, 1))
            _cropped = _pupil * _S_shifted
            _low_res = ift(_cropped)
            _meas_amp = np.sqrt(np.clip(measurements5[_idx], 0.0, None))
            _new_low_res = _meas_amp * np.exp(1j * np.angle(_low_res))
            _new_cropped = ft(_new_low_res)
            _delta = _new_cropped - _cropped
            _P_norm = (np.abs(_pupil) ** 2).max() + 1e-9
            _O_norm = (np.abs(_S_shifted) ** 2).max() + 1e-9
            _S_new = _S_shifted + _alpha * np.conj(_pupil) / _P_norm * _delta
            if _epry_on:
                _pupil = (_pupil
                          + _beta * np.conj(_S_shifted) / _O_norm * _delta)
                _pupil = _pupil * _mask
            _spec = np.roll(_S_new, shift=(-_ky, -_kx), axis=(0, 1))
        _err = 0.0
        for _idx, (_ky_raw, _kx_raw, _step) in enumerate(led_positions5):
            _ky = int(round(_ky_raw))
            _kx = int(round(_kx_raw))
            _Ss = np.roll(_spec, shift=(_ky, _kx), axis=(0, 1))
            _lr = ift(_pupil * _Ss)
            _err += float(np.mean(
                (np.abs(_lr)
                 - np.sqrt(np.clip(measurements5[_idx], 0, None))) ** 2
            ))
        _err_hist5.append(_err / _n_leds)
    spec5_final = _spec
    pupil5_final = _pupil
    err_hist5 = _err_hist5
    return err_hist5, pupil5_final, spec5_final


@app.cell
def _t5_display(
    Circle,
    PUPIL_CUTOFF_PX,
    amp_gt5,
    err_hist5,
    ift,
    led_positions5,
    mo,
    np,
    phi_gt5,
    plt,
    pupil5_final,
    spec5_final,
    t5_aberration,
    t5_epry,
):
    _field = ift(spec5_final)
    _amp = np.abs(_field)
    _phi = np.angle(_field)
    _gt = amp_gt5 * np.exp(1j * phi_gt5)
    if np.abs(np.sum(_field * np.conj(_gt))) > 1e-9:
        _off = np.angle(np.sum(_field * np.conj(_gt)))
        _phi = np.angle(_field * np.exp(-1j * _off))
    _N = spec5_final.shape[0]
    _cy, _cx = _N // 2, _N // 2
    _fig, _ax = plt.subplots(2, 3, figsize=(11.5, 7.2),
                             constrained_layout=True)
    _ax[0, 0].imshow(amp_gt5, cmap="gray", vmin=0, vmax=1.05)
    _ax[0, 0].set_title("ground-truth amplitude", fontsize=10)
    _ax[0, 1].imshow(phi_gt5, cmap="twilight", vmin=-2.0, vmax=2.0)
    _ax[0, 1].set_title("ground-truth phase (rad)", fontsize=10)
    _ax[0, 2].imshow(np.log1p(np.abs(spec5_final)), cmap="gray")
    _ax[0, 2].set_title("|recovered spectrum| (log)", fontsize=10)
    for (_ky_led, _kx_led, _) in led_positions5:
        _ring = Circle((_cx - _kx_led, _cy - _ky_led), PUPIL_CUTOFF_PX,
                       fill=False, edgecolor="cyan", linewidth=0.6,
                       linestyle="--", alpha=0.7)
        _ax[0, 2].add_patch(_ring)
    _ax[1, 0].imshow(_amp, cmap="gray", vmin=0, vmax=1.05)
    _ax[1, 0].set_title("recovered amplitude", fontsize=10)
    _ax[1, 1].imshow(_phi, cmap="twilight", vmin=-2.0, vmax=2.0)
    _ax[1, 1].set_title("recovered phase (rad)", fontsize=10)
    if err_hist5:
        _ax[1, 2].plot(range(1, len(err_hist5) + 1), err_hist5, "-o",
                       markersize=3)
        _ax[1, 2].set_yscale("log")
        _ax[1, 2].set_xlabel("iteration")
        _ax[1, 2].set_ylabel("mean measurement residual")
        _ax[1, 2].set_title("error per iteration", fontsize=10)
        _ax[1, 2].grid(True, alpha=0.3)
    for _r in _ax[:2]:
        for _a in _r[:2]:
            _a.set_xticks([])
            _a.set_yticks([])
    _ax[0, 2].set_xticks([])
    _ax[0, 2].set_yticks([])
    t5_plot = show_fig(_fig)

    if t5_epry.value and t5_aberration.value > 0:
        _figp, _axp = plt.subplots(1, 2, figsize=(6.5, 3.3),
                                   constrained_layout=True)
        _axp[0].imshow(np.angle(pupil5_final), cmap="twilight",
                       vmin=-np.pi, vmax=np.pi)
        _axp[0].set_title("recovered pupil phase", fontsize=10)
        _axp[1].imshow(np.abs(pupil5_final), cmap="gray")
        _axp[1].set_title("recovered pupil magnitude", fontsize=10)
        for _a in _axp:
            _a.set_xticks([])
            _a.set_yticks([])
        t5_pupil_plot = show_fig(_figp)
    else:
        t5_pupil_plot = mo.md("")
    return t5_plot, t5_pupil_plot


@app.cell
def _t5_prompts(mo):
    t5_prompts = mo.accordion({
        "Noise / exposure — which Tab gave us this?": mo.md(
            "**Tab 3 — the bandwidth / SNR trade.** Outer (high-angle) LEDs "
            "illuminate weakly-scattered, high-spatial-frequency content "
            "in the darkfield regime — dim signal vs fixed read noise = "
            "worse SNR exactly where the resolution gain comes from. Drop "
            "exposure and watch the outer spectrum get noisier and the "
            "recovered amplitude get speckle-y."
        ),
        "Aberration — which Tab gave us this?": mo.md(
            "**Tab 2 Stage E (pupils).** Real lenses have a non-ideal pupil. "
            "If the reconstructor assumes an ideal pupil but the camera "
            "saw an aberrated one, the alternating projection fights "
            "itself. Turn **EPRY** on: the algorithm jointly recovers "
            "object **and** pupil. This is what keeps cheap optics usable."
        ),
        "LED misposition — which Tab gave us this?": mo.md(
            "**Tab 2/3 — the angle ↔ frequency map.** The reconstructor "
            "places each measurement at an *assumed* k-shift; if reality "
            "differs, every window lands in the wrong place and the overlaps "
            "stop agreeing. Real FPM adds a position-search step (not "
            "implemented here) — but you can see the failure clearly."
        ),
        "Partial coherence — which Tab gave us this?": mo.md(
            "**Tab 2 — the single-plane-wave idealization softening.** A "
            "real LED has finite size and bandwidth; its illumination is "
            "many plane waves, which blurs the spectral-window's edge and "
            "caps high-frequency recovery. **Hardware-fundamental** — smaller "
            "pinholes / narrower filters help, but you can't iterate it away."
        ),
    })
    return (t5_prompts,)


@app.cell
def _t5_assembly(mo, t5_controls, t5_plot, t5_prompts, t5_pupil_plot):
    tab5 = mo.vstack([
        mo.md(
            "## Tab 5 — Synthesis: the principles in work clothes\n\n"
            "Nothing new — same pipeline as Tab 4, with the perturbations "
            "real FPM systems contend with. **Predict before you flip the "
            "switch.** If Tabs 1–4 landed, you can call each effect."
        ),
        t5_controls,
        t5_plot,
        t5_pupil_plot,
        mo.md("---"),
        t5_prompts,
    ])
    return (tab5,)


# ===========================================================================
# Tab assembly
# ===========================================================================


@app.cell
def _tabs(header, mo, tab1, tab2, tab3, tab4, tab5):
    tabs = mo.ui.tabs({
        "Tab 1 · FT properties": tab1,
        "Tab 2 · Central dogma": tab2,
        "Tab 3 · FPM premise": tab3,
        "Tab 4 · Inverse problem": tab4,
        "Tab 5 · Synthesis": tab5,
    })
    mo.vstack([header, tabs])
    return (tabs,)


if __name__ == "__main__":
    app.run()
