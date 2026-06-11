# Fourier Ptychography Teaching App — Build Spec

## Purpose & audience

An interactive, browser-hosted teaching tool that builds **operational understanding of the physics and math** behind Fourier ptychographic microscopy (FPM), for undergraduates who will build, operate, and possibly modify a small 3D-printed FPM. Assumed background: proof-based linear algebra, multivariable calculus, light ODE/PDE exposure, basic ray optics, and an operational (not rigorous) grasp of the Fourier transform.

Five tabs, Steps 1–5. Steps 1–4 are the core (physics + math). Step 5 is **synthesis**: it introduces no new physics, only the earlier principles wearing "work clothes."

The guiding idea for the whole app: **a teaching sim should reveal what a real microscope hides** — the complex field, its phase, the Fourier-plane spectrum, and the iterative reconstruction state. Lean into showing the invisible middle.

---

## Global / shared decisions (read first)

**Stack.** Python + [marimo](https://marimo.io), exported to WASM (`marimo export html-wasm ... --mode run`) and hosted on GitHub Pages. Runs fully client-side (Pyodide); no backend; interactions stay snappy because there are no server round-trips. Keep dependencies minimal — **numpy only** if possible (use `numpy.fft`, avoid scipy to keep the cold-load bundle small). Plotting via matplotlib (Pyodide-supported).

**Tabs.** Use `mo.ui.tabs(...)` (or stacked sections) for the five steps. Each tab interleaves short markdown explanation with reactive widgets.

**Grid sizes.** Object/field grids: **256×256** for Tabs 1–3 (one FFT per interaction, real-time). Tab 4–5 reconstruction: **128×128** to keep the iterative loop responsive.

**Visual grammar (apply identically in every tab).**
- Real-space pane on the **left**, frequency/Fourier-plane pane on the **right**.
- Magnitude: `gray` or `viridis`. Phase: a **cyclic** colormap (`twilight` or `hsv`).
- **Always alpha-mask phase by normalized magnitude** so near-zero-magnitude regions don't render random-phase confetti.
- The lens pupil is always drawn as the same circle overlaid on the Fourier pane.
- A reusable two-pane plotting helper should be written once and called everywhere.

**Shared forward model (write once, reuse in Tabs 3–5).** All of FPM, the inverse problem, and the practical nuances are one model viewed three ways. Implement a single function:

```python
def simulate_intensity(object_field, pupil, k_shift_px):
    """
    object_field : complex 2D array (transmittance A*exp(i*phi))
    pupil        : complex 2D array (disk mask * exp(i*aberration_phase)),
                   centered, radius = cutoff in spatial-frequency pixels
    k_shift_px   : (dy, dx) integer shift of the object spectrum (tilted illumination)
    returns      : real 2D intensity image = |IFFT( pupil * roll(FFT(object), k_shift) )|^2
    """
```

Use `numpy.fft.fft2/ifft2` with `fftshift` consistently; document the sign/orientation convention in one place (a flipped axis here is the classic silent bug).

**Physical parameters (expose as labeled, tunable defaults).** wavelength λ (default 0.5 µm), object-plane pixel pitch, objective NA (default low, e.g. 0.1), illumination NA / max LED angle. Pupil cutoff radius in frequency pixels = NA/λ mapped to the grid. Absolute units can be normalized; keep them labeled so Tab 5 can connect to hardware.

---

## Tab 1 — Fourier transform (discover the properties)

**Goal.** Let students *discover* the FT properties they'll need, especially that **position lives in phase** (shift theorem) and that the transform is linear. Seed the idea that **phase is precious** (motivates Tab 4).

**Widget 1 — Dual-pane FT explorer with predict-then-reveal.**
- Left: object (pick from presets: point, two points, rectangle/slit, circle/disk, simple shape, grating; option to upload a grayscale image). Right: its FT, with a toggle for **magnitude (log)** vs **phase (cyclic colormap, magnitude-masked)**.
- Controls: x/y translation sliders; scale slider; rotation slider; a second object selector + an "**add the two objects**" toggle (for linearity).
- Interaction pattern: each manipulation is framed as a challenge with a **"Predict"** prompt (markdown) and a **"Reveal"** button/toggle that performs it. Discovery prompts to implement:
  - *Translate the object.* What changes in the magnitude? In the phase? (→ shift theorem: magnitude invariant, phase gains a linear ramp.)
  - *Add object A + object B.* Predict the FT before revealing. (→ linearity: FT(A+B)=FT(A)+FT(B).)
  - *Scale the object larger.* What happens to the FT? (→ inverse scaling.)
  - *Rotate the object.* (→ FT rotates with it.)
  - *Use a purely real object.* Notice the magnitude's symmetry. (→ Hermitian symmetry; foreshadows why phase objects differ.)

**Widget 2 — Phase-vs-magnitude swap.**
- Two preset images. A button swaps their FT phases (keeping each one's magnitude), then inverse-transforms.
- Reveal: the reconstruction follows the **phase**, not the magnitude. One-line takeaway tying to Tab 4: "this is why throwing away phase is catastrophic, and why we'll work to get it back."

**Widget 3 — Convolution theorem (light touch).**
- Object on the left, a selectable kernel; show convolution-in-space equals product-in-frequency. One paragraph: this is what imaging-through-a-pupil will turn out to be.

**Computation.** One `fft2`/`ifft2` per interaction at 256². Trivial, fully live.

---

## Tab 2 — The central dogma, built from one point (the linchpin)

**Goal.** Construct, by superposition, the result that *the image is the inverse transform of (object spectrum × lens pupil)*, and detected intensity is its modulus squared. Build it in stages so nothing is asserted. This is the most important tab; spend the most design care here.

**Stage A — One object point → its far field.**
- A single point at position (x₀, y₀) in the object plane (draggable or via sliders). Right pane shows its far-field pattern: uniform magnitude, a **linear phase ramp**.
- Move the point: the ramp's slope changes. State explicitly: the far field of a displaced point is a tilting phase ramp — **this is the shift theorem in its most elemental form** (position ↔ phase slope). (Tab 3 will run this backwards.)

**Stage B — Two points, then many → linearity builds an object.**
- Add a second point; show the far-field **intensity** fringes (interference), spacing inversely proportional to point separation.
- Let the student place several points with adjustable complex weights (amplitude + phase). The far field is the sum of their ramps. Takeaway: **by linearity, an arbitrary object's far field is the superposition of its points' far fields — i.e. the far field is the Fourier transform of the object.**

**Stage C — Add a lens with a pupil → imaging.**
- Now the object's far field forms at the **pupil plane**; a finite **pupil** (a disk; ideal = unit phase) multiplies it; a second transform re-images. Show three linked things: object | spectrum-at-pupil-with-pupil-circle | reconstructed image.
- Shrink the pupil (lower NA): high frequencies are clipped, the image blurs. The **resolution limit emerges** from the pupil radius (cutoff = NA/λ) rather than being asserted.
- **Distinguish the two complex functions clearly in the UI labels:** the *object transmittance* `t = A·exp(iφ)` (amplitude/absorbance A∈[0,1] and phase delay φ, in the object plane) vs the *lens pupil* `P` (aperture + aberration, in the Fourier plane).

**Stage D — Object library (amplitude / phase / mixed).**
- Presets: pure **absorption** object (varying A, φ=0), pure **phase** object (A≈1, varying φ — e.g. a cell-like blob), and mixed. Optional upload (interpret brightness as amplitude, or offer an "as phase" toggle).
- Key reveal with the phase-object preset: in a plain intensity image it is nearly **invisible**, even though its spectrum is rich. Explicit teaser: "recovering this needs phase — see Tab 4."

**Stage E — Aberrations (extension, off by default).**
- A toggle reveals **Zernike sliders** that add aberration phase to the pupil: defocus, astigmatism, coma, spherical (low-order). Watch the PSF and image degrade. Frame as: "real (cheap) lenses live here," connecting forward to Tab 5's pupil-recovery.

**Computation.** Stage A–B can be drawn analytically (no FFT needed for single components). Stage C–E: one `fft2` (object) → multiply by pupil → `ifft2`. Real-time at 256².

---

## Tab 3 — Introducing FPM (make the shift theorem explicit; state the premise)

**Goal.** Explicitly connect Tab 2's position↔phase-ramp result to FPM, and establish the premise: **more illumination angles → higher spatial frequencies captured → higher resolution, at fixed field of view.**

**Widget 1 — Tilted illumination → spectral shift.**
- Controls: illumination angle (θx, θy) sliders, or a clickable LED-grid position. Multiplying the object by the corresponding plane wave **shifts its spectrum** across the fixed pupil.
- Display: object spectrum sliding under the stationary pupil circle (right) and the resulting low-resolution image (left).
- Explicit text: tilting the illumination = multiplying by a phase ramp = shifting the spectrum (the Tab-2 single-point relationship, reversed). Frequencies that were outside the pupil are now inside it.

**Widget 2 — Synthetic aperture / k-space tiling.**
- A small LED grid; clicking/sweeping LEDs each drops a pupil-sized disk into an accumulating synthesized spectrum, with a live reconstructed-resolution preview that sharpens as coverage grows while the **field of view stays fixed**.
- Side note panel: synthetic NA ≈ objective NA + max illumination NA; bigger/closer LED array buys resolution but pushes acquisitions into dim darkfield (sets up Tab 5).

**Computation.** Spectrum shift = `numpy.roll` (or phase ramp), then mask + `ifft2`. Live at 256².

---

## Tab 4 — The inverse problem (recover phase from intensity)

**Goal.** Show *why* phase must be recovered and *how* the alternating-projection algorithm does it. Keep it minimal: a **3×3 LED grid** and a deliberately simple ground-truth target.

**Widget 1 — Reconstruction stepper.**
- Setup: a known simple complex ground-truth object (small amplitude + phase features). Precompute the 9 low-resolution intensity images via `simulate_intensity` (one per LED k-shift). Cache them.
- Algorithm (alternating projection / Gerchberg–Saxton style):
  1. Initialize high-res spectrum estimate (upsampled brightfield, flat phase).
  2. For each LED: extract its shifted spectral window → ×pupil → `ifft2` → low-res complex field; **replace the amplitude with √(measured intensity), keep the phase**; `fft2` back; update that window of the high-res spectrum.
  3. Repeat for a set number of iterations.
- Controls: **Step one LED**, **Step one full iteration**, **Run/Pause**, **Reset**; a slider for total iterations.
- Displays (the "invisible middle"): current recovered **amplitude**, recovered **phase**, the **filling k-space** spectrum, and an **error-vs-iteration** curve.
- Let students sabotage it: a "**bad initialization**" toggle and a "**use fewer LEDs**" control, to see failure/convergence behavior.

**Widget 2 — Overlap slider.**
- Vary the Fourier-space overlap between adjacent LED windows; watch convergence quality change and degrade below a usable threshold. Text: the overlap is shared spectral content that over-determines the missing phase — that redundancy is what makes recovery possible.

**Honest framing to include.** With only 9 LEDs the resolution gain is modest by design; the lesson is **phase emerging from intensity-only data and the spectrum filling in**, not a large resolution leap. Keep the target simple enough that 9 windows suffice.

**Computation.** Heaviest tab: precompute the 9-image stack once; run a **capped** number of iterations per click at 128². Keep the pupil fixed here (no joint pupil recovery — that's a Tab-5 toggle).

---

## Tab 5 — Synthesis (the principles in work clothes)

**Goal.** No new physics. Reuse the Tab-4 reconstructor with perturbation toggles, each **labeled by the earlier principle it stresses**. A student who understood Tabs 1–4 should be able to *predict* each effect before flipping the switch — that prediction is the built-in assessment.

Perturbation toggles (all layered onto the Tab-4 pipeline):
- **Noise / exposure** — Poisson + read noise, adjustable exposure. Reveals the brightfield-bright / darkfield-dim SNR gap. (← Tab 3 bandwidth meets weak high-angle scattering; this *is* the "low spatial frequency problem.")
- **Aberration** — add a hidden pupil aberration; optional **pupil recovery (EPRY)** toggle that estimates and corrects it. (← Tab 2 pupil being non-ideal; this is what makes builds reproducible unit-to-unit.)
- **LED misposition** — perturb assumed LED positions; optional **position-correction** toggle. (← Tab 2/3 frequency-angle map being miscalibrated.)
- **Partial coherence** — finite LED size/bandwidth blurs the spectral-window edges, capping high-frequency recovery. (← Tab 2's single-plane-wave idealization softening; a fundamental hardware limit, not a fixable bug.)

For each toggle: a one-line "which principle is this?" prompt and a "predict, then observe" structure mirroring Tab 1.

---

## Suggested build order for the agent

1. Shared infrastructure: the two-pane plotting helper (with phase masking + cyclic colormap), `simulate_intensity`, FFT convention/utilities, parameter defaults.
2. Tab 2 Stages A–C (the linchpin) — get the single-point → linearity → pupil-imaging chain right first; everything leans on it.
3. Tab 1 (FT explorer + predict/reveal + phase swap).
4. Tab 3 (reuses `simulate_intensity`).
5. Tab 4 reconstruction stepper (reuses `simulate_intensity`; add caching).
6. Tab 5 (wraps Tab 4 with perturbation toggles).
7. Tab 2 Stages D–E (object library + Zernike extension).
8. WASM export + GitHub Pages deploy; verify cold-load time and trim dependencies.

## Notes
- Prefer sliders/discrete controls over free dragging where either works; if a drag interaction is wanted (e.g. moving the Tab-2 point), client-side WASM handles it fine.
- Keep one consistent sign/orientation convention for all FFTs and document it.
- Apply sound visual-design judgment (clear labels, restrained palette, consistent layout) when implementing the UI.
