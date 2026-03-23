# Neural Networks Are Fractals

## Overview
- **Topic**: Fractal structure spontaneously emerges in neural network weights during training
- **Hook**: "What do coastlines and neural networks have in common?"
- **Target Audience**: Knows linear algebra (eigenvalues, SVD) and basic ML (what a transformer is, what training means)
- **Estimated Length**: 12-15 minutes
- **Key Insight**: Training doesn't just find good weights — it builds fractal geometry. Every layer becomes a scaled copy of every other layer, the optimization path is a fractal curve, and this happens universally across model sizes.

## Narrative Arc
We start with a mystery — why do completely different neural networks develop eerily similar internal structure? We train a small GPT from scratch and watch its weights transform from random noise into organized fractal patterns. Each finding builds on the last: first the weights become power-law, then we zoom out and see layers mirroring each other, then we discover the training path itself is fractal. The punchline: this isn't a quirk of one model — it's universal.

---

## Scene 1: The Hook
**Duration**: ~45 seconds
**Purpose**: Grab attention, set up the mystery

### Visual Elements
- Mandelbrot set zoom animation (classic fractal imagery)
- Crossfade to a neural network weight matrix heatmap
- Side-by-side: coastline outline | weight matrix correlation pattern

### Content
Open on a slow Mandelbrot zoom — the classic "infinite complexity" shot. But instead of zooming forever, we freeze and pull back to reveal it's actually a neural network weight matrix. Quick montage: coastline fractal, stock price chart, fern leaf, then — a weight matrix correlation plot showing the same nested self-similar structure.

Title card: **"Neural Networks Are Fractals"**

### Narration Notes
"Fractals are everywhere — coastlines, trees, stock markets. But here's something nobody expected: the weights inside a neural network are fractal too. Not metaphorically. Literally. Let me show you."

### Technical Notes
- Use `ImageMobject` for the actual plots from our experiments
- Mandelbrot can be a pre-rendered zoom or a simple parametric plot
- Smooth crossfade with `Transform` between fractal and weight matrix

---

## Scene 2: What Makes Something Fractal?
**Duration**: ~90 seconds
**Purpose**: Quick refresher on fractals and power laws for the LA/ML audience

### Visual Elements
- Koch snowflake construction (3 iterations, animated)
- Log-log plot appearing beside it — straight line = power law
- Equation: `P(x) ~ x^{-\alpha}` with alpha highlighted
- Marchenko-Pastur distribution curve (labeled "random") vs heavy-tailed curve (labeled "structured")

### Content
Brief, visual definition of fractals: **self-similarity across scales** and **power-law statistics**. Build a Koch snowflake step by step — at each zoom level, the same pattern. Then pivot to the mathematical signature: on a log-log plot, fractal systems produce straight lines. This is a power law.

Show the Marchenko-Pastur distribution — "this is what random matrices look like." Then morph it into a heavy-tailed distribution — "this is what trained neural networks look like." The gap between these two distributions IS the learned structure.

### Narration Notes
"A fractal has two signatures. First: self-similarity — zoom in and you see the same pattern. Second: power laws — on a log-log plot, you get a straight line. Random matrices follow a well-known distribution called Marchenko-Pastur. But something happens when you train a neural network..."

### Technical Notes
- Koch snowflake: recursive `Line` construction with `Create` animations
- MP distribution: plot with `Axes` + parametric curve
- Morph MP → heavy-tail with `Transform`
- Keep equations minimal — focus on visual intuition

---

## Scene 3: The Experiment
**Duration**: ~60 seconds
**Purpose**: Set up what we did — train a GPT, save checkpoints, analyze

### Visual Elements
- Simple GPT architecture diagram (embedding → 6 transformer blocks → output)
- Training loss curve animating downward
- Checkpoint icons appearing at regular intervals along the curve
- Text overlay: "12.3M parameters | Tiny Shakespeare | 10,000 steps"

### Content
"Here's what we did." Show the model architecture as a clean vertical stack of transformer blocks. Animate the training process — loss curve dropping from ~7 to ~1. At each checkpoint (every 500 steps), a little snapshot icon appears. "We saved 21 snapshots of every weight in the network. Then we asked: what geometric structure is hiding in these matrices?"

### Narration Notes
Keep this fast and light. The audience knows what a transformer is. Don't over-explain — just establish the experimental setup.

### Technical Notes
- Architecture diagram: `VGroup` of `RoundedRectangle` blocks with labels
- Loss curve: `Axes` with animated `line` using `Create`
- Checkpoint icons: small circles or squares appearing with `FadeIn`

---

## Scene 4: Power Laws Emerge
**Duration**: ~120 seconds
**Purpose**: The first major finding — singular value spectra go from random to power-law

### Visual Elements
- Log-log axes (rank vs singular value)
- Gray curve: initialization (follows MP, curves)
- Blue curve: trained weights (straight line = power law)
- Animation: gray curve morphing into blue curve as "training step" counter ticks up
- Inset: the actual `singular_values_loglog.png` plot from our experiments
- Alpha exponent appearing: `\alpha \approx 1.1 - 4.0`

### Content
This is the core visual. Start with a log-log plot showing the singular values of a weight matrix at initialization — it follows the curved Marchenko-Pastur shape. Then animate training: as the step counter increases, the curve slowly straightens into a line. A straight line on log-log = power law.

Show this happening for multiple layers simultaneously — they ALL develop power laws, but at different rates. Attention layers get steeper (larger alpha) than MLP layers.

Zoom into the `power_law_evolution.png` — show the exponents climbing over training steps.

### Narration Notes
"At initialization, weight matrices are random. Their singular values follow the Marchenko-Pastur distribution — this curved shape. But watch what happens during training... [pause as curve straightens] ...a straight line on a log-log plot. That's a power law. The same mathematical signature as coastlines and earthquake magnitudes. And it happens in every single layer."

### Technical Notes
- `Axes` with log scale on both axes
- Animate curve morph: interpolate between MP shape and power-law line
- Use `ValueTracker` for the training step counter
- Multiple curves with different colors for different layer types
- `SurroundingRectangle` or `Indicate` to highlight the straight-line region

---

## Scene 5: Fractal Attention
**Duration**: ~120 seconds
**Purpose**: The most visually striking finding — nested block-diagonal structure in attention correlations

### Visual Elements
- 192x192 correlation matrix heatmap, animated
- Init state: uniform random noise (no structure)
- Trained state: clear block-diagonal with nested sub-blocks
- Zoom sequence: zoom into a block, find smaller blocks inside, zoom into those
- Side-by-side: Sierpinski triangle | attention correlation matrix

### Content
"But the most beautiful result is what happens when we look at the *correlations* between neurons." Show the attention weight correlation matrix at initialization — random noise, no pattern. Then dissolve to the trained version — dramatic reveal of nested block-diagonal structure.

Now zoom in. Inside each block, there are smaller blocks. Inside those, even smaller blocks. This is self-similarity — the hallmark of a fractal. Quick side-by-side with a Sierpinski triangle to drive the point home.

Show this for multiple layers — L2 through L5 all develop the same pattern.

### Narration Notes
"Now here's where it gets beautiful. This is the correlation matrix of an attention layer — which neurons fire together. At initialization: noise. After training... [dramatic pause, reveal] ...blocks within blocks within blocks. The same pattern at every scale. That's not just structure — that's a fractal."

### Technical Notes
- Use `ImageMobject` for the actual `attention_correlations.png` heatmap
- Zoom effect: `self.camera.frame.animate.move_to().set(width=)`
- Or crop and scale regions of the image progressively
- Sierpinski triangle: recursive `Polygon` construction
- This is the "wow" moment — spend time on the reveal animation

---

## Scene 6: Layers as Scaled Copies
**Duration**: ~90 seconds
**Purpose**: Cross-layer self-similarity — every layer has the same spectral shape

### Visual Elements
- 7x7 similarity matrix (embed + 6 layers), values >0.99 everywhere
- Overlay of all layer spectra on one log-log plot — nearly identical curves
- Animation: take one layer's spectrum, scale it, overlay on another — they match
- Visual metaphor: Russian nesting dolls or fractal zoom

### Content
"If each layer is fractal internally, what about across layers?" Show the cross-layer similarity matrix — all values above 0.99. Every layer's singular value spectrum has the same shape. Overlay them on one plot — they collapse onto the same curve.

Animate: take Layer 0's spectrum, scale it (multiply by a constant), and it lands exactly on Layer 5's spectrum. The layers aren't just similar — they're scaled copies. Like looking at the same fractal at different magnifications.

### Narration Notes
"Every layer develops the same spectral shape. Not approximately — the cosine similarity is above 0.99. Layer 0 and Layer 5 are scaled copies of each other. The same pattern, repeated at every depth. That's literally the definition of self-similarity."

### Technical Notes
- Similarity matrix: `MobjectTable` or `ImageMobject` from `cross_layer_similarity.png`
- Spectra overlay: multiple curves on same `Axes` with `Create` animations
- Scale animation: `curve.animate.scale()` to show alignment
- Color-code layers with a gradient (viridis-style)

---

## Scene 7: The Fractal Path of Learning
**Duration**: ~120 seconds
**Purpose**: SGD trajectory is fractal — the training path through weight space has fractal dimension

### Visual Elements
- 3D PCA trajectory (from `sgd_trajectory_3d.png`), animated as a path being drawn
- Green dot (start) → red dot (end), path traces through 3D space
- Log-log plot: displacement vs scale, straight line, H=0.753 labeled
- Comparison: Brownian motion path (jagged, random) vs our trajectory (smooth arc but fractal)
- Fractal dimension equation: `D = 2 - H = 1.247`

### Content
"So the weights are fractal. But what about the process of getting there?" Show the 12.3M-dimensional weight space projected into 3D via PCA. Animate the training trajectory — a green dot starts moving, tracing a smooth but complex arc through space.

Now measure it: displacement scales as a power law with Hurst exponent H=0.753. That means the trajectory has fractal dimension 1.247. It's not a random walk (H=0.5) — it's persistent, trending, with structure at every scale. Show a Brownian motion path for comparison — much more jagged and undirected.

### Narration Notes
"The path that SGD takes through 12 million dimensions of weight space... is a fractal curve. Its Hurst exponent is 0.753 — significantly above the random walk value of 0.5. Training isn't wandering randomly. It's carving a fractal path with structure at every scale."

### Technical Notes
- 3D trajectory: `ThreeDScene` with `ParametricFunction` or point-by-point line
- Animate path drawing with `Create` or custom updater
- Use `self.move_camera` for a slow rotation to show 3D depth
- Brownian motion: pre-generated random walk for visual comparison
- Hurst scaling plot: 2D `Axes` inset or separate shot

---

## Scene 8: Universality
**Duration**: ~90 seconds
**Purpose**: The capstone — all of this holds across model scales

### Visual Elements
- Three model silhouettes: small (3L), medium (6L), large (8L) with parameter counts
- Bar charts morphing to show same metrics across all three scales
- Hurst exponents: 0.755, 0.757, 0.763 — nearly identical bars
- The `scale_comparison.png` plot, or a cleaner animated version

### Content
"But is this just a quirk of one model?" Reveal three models at different scales. Show the key metrics side by side — power-law exponents, kurtosis, cross-layer similarity, Hurst exponent. They all tell the same story.

The killer number: Hurst exponent is 0.755, 0.757, 0.763 across three very different architectures. The fractal dimension of learning is essentially a constant.

Cross-layer similarity actually *increases* with scale (0.85 → 0.92) — bigger models are *more* fractal.

### Narration Notes
"We trained three models — 5 million, 12 million, and 19 million parameters. Every single fractal signature appears in all of them. The Hurst exponent barely changes: 0.755, 0.757, 0.763. The fractal dimension of learning is... a universal constant. And bigger models are actually *more* self-similar than smaller ones."

### Technical Notes
- Three model icons: `VGroup` stacks of rectangles at different sizes
- Animated bar charts: `BarChart` with `animate` to change values
- Highlight the Hurst values with `SurroundingRectangle` and a pulsing glow
- Use the actual `scale_comparison.png` or recreate key panels

---

## Scene 9: What Does It Mean?
**Duration**: ~90 seconds
**Purpose**: Synthesis — why this matters, what it tells us about learning

### Visual Elements
- Timeline: init (random/noise) → trained (fractal/structured), with icons for each finding
- The "training builds fractals" visual: weight matrix heatmap morphing from noise to structured
- Closing montage: Mandelbrot zoom but this time it's clear it's a weight matrix
- Final text: key paper citations

### Content
Pull it all together. "Training is not just optimization — it's the spontaneous emergence of fractal geometry." Show a visual timeline of everything we found:
- Weights: random → power-law
- Correlations: noise → nested blocks
- Layers: independent → scaled copies
- SGD path: → fractal curve
- Across scales: → universal

End with the philosophical note: the same mathematics that describes coastlines, galaxies, and blood vessels also describes the internal structure of neural networks. Maybe that's not a coincidence.

### Narration Notes
"So here's the picture. Training isn't just finding good weights. It's building fractal geometry — power laws in the spectra, self-similarity across layers, fractal paths through weight space. And it happens the same way whether the model has 5 million or 19 million parameters. The mathematics of fractals — the same math that describes coastlines and galaxies — is the mathematics of learning."

### Technical Notes
- Timeline: horizontal sequence of icons with `FadeIn` animations
- Weight matrix morph: interpolate between noise texture and structured heatmap
- Final Mandelbrot zoom: reverse the opening — start wide, zoom reveals weight structure
- Keep citations subtle — small text at bottom

---

## Transitions & Flow
- **Scene 1→2**: "But what does 'fractal' actually mean?" (curiosity bridge)
- **Scene 2→3**: "Let's find out." (action bridge)
- **Scene 3→4**: "Here's the first thing we found." (discovery)
- **Scene 4→5**: "But the weights are just the beginning." (escalation)
- **Scene 5→6**: "It goes deeper." (escalation)
- **Scene 6→7**: "And it's not just the destination — it's the journey." (pivot)
- **Scene 7→8**: "But is this universal?" (question)
- **Scene 8→9**: "Yes. And here's what that means." (resolution)

## Color Palette
- **Background**: `#1a1a2e` (deep midnight navy — clean, cinematic)
- **Primary (structure/trained)**: `#e6a817` (warm gold/amber — "fractal gold")
- **Secondary (random/init)**: `#4a6fa5` (cool steel blue — "random blue")
- **Accent (highlight)**: `#00d4aa` (bright teal — for key numbers and reveals)
- **Accent 2 (attention)**: `#ff6b6b` (coral red — for attention-specific visuals)
- **Text**: `#e8e8e8` (soft white)
- **Grid/axes**: `#3a3a5c` (muted purple-gray)

## Mathematical Content
- Marchenko-Pastur density: `\rho_{MP}(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi\sigma^2\lambda}`
- Power law: `P(x) \sim x^{-\alpha}`
- Hurst exponent definition: `\langle |X(t+\tau) - X(t)| \rangle \sim \tau^H`
- Fractal dimension: `D = 2 - H`
- Cosine similarity: `\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}||\mathbf{b}|}`
- Power-law exponent range: `\alpha \approx 1.1 - 4.0, \quad R^2 > 0.9`

## Implementation Order
1. **Scene 4** (Power Laws Emerge) — core visual, establishes the main technique
2. **Scene 5** (Fractal Attention) — most visually striking, uses real plot images
3. **Scene 7** (SGD Trajectory) — 3D scene, technically interesting
4. **Scene 2** (What is Fractal) — foundational, can reuse elements from 4
5. **Scene 6** (Cross-Layer) — builds on visuals from 4
6. **Scene 8** (Universality) — bar charts, relatively straightforward
7. **Scene 3** (Experiment) — simple diagram scene
8. **Scene 1** (Hook) — needs final polish, references other scenes
9. **Scene 9** (Conclusion) — synthesis, best done last

## ManimCE Notes
- Use ManimCE (community edition) — better documented, stable API
- `config.background_color = "#1a1a2e"` for the dark theme
- Use `ImageMobject` liberally for our actual experimental plots — they're the real evidence
- `ThreeDScene` only for Scene 7 (SGD trajectory)
- Prefer `Transform`/`ReplacementTransform` over `FadeOut`/`FadeIn` for continuity
- Use `ValueTracker` + `always_redraw` for animated plots (training step counter, evolving curves)
