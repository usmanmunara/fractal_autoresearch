# Neural Networks Are Fractals — Voiceover Script

This script is aligned to the current scene order in `video/scenes.py`.
Audience: basic algebra + beginner ML.
Delivery target: calm, curious, precise.

Suggested voice settings for TTS:
- Pace: 0.95x to 1.0x
- Stability: medium-high
- Style exaggeration: low-medium
- Pauses: keep short pauses at punctuation, longer pauses after key reveals

---

## Scene 1 — TheHook (~45s)

"What if the same mathematics that describes coastlines and snowflakes also appears inside neural networks?

At first, this image looks like a classic fractal pattern.
Now watch this crossfade.

This is not computer art.
It is a real correlation map from a trained transformer attention layer.

Notice the blocks.
Then blocks inside those blocks.
Then structure inside those smaller blocks.

That repeating pattern across scales is the signature we are chasing.

So here is our question:
Are neural networks, in a geometric sense, fractal?"

---

## Scene 2 — WhatIsFractal (~65s)

"Before jumping into deep learning, let’s make fractals concrete.

A fractal has self-similarity:
zoom in, and you keep seeing a related pattern.

The Koch snowflake is a classic example.
Each refinement adds detail, but the rule stays the same.

There is also a quantitative fingerprint:
power laws.
On log-log axes, power laws look like straight lines.

Now compare singular-value spectra from one real model layer:
random initialization versus trained weights.

After training, the spectrum aligns much more closely with a straight-line trend.

That is our first hint:
training may be organizing weights into scale-free structure."

---

## Scene 3 — TheExperiment (~55s)

"Here is the setup.
We train a small transformer on Tiny Shakespeare.
Six layers, about twelve million parameters, and checkpoints saved throughout training.

Instead of only looking at final accuracy, we track geometric structure over time.

Each point here is a real saved checkpoint.
The plotted value is a fitted power-law exponent, alpha.

The point is not that alpha is magic.
The point is that structure appears systematically, not randomly.

So the experiment is simple:
save many snapshots,
measure geometry,
and ask whether fractal signatures emerge consistently."

---

## Scene 4 — PowerLawsEmerge (~80s)

"Finding one:
power laws emerge during training.

This curve starts at random initialization.
As optimization proceeds, the spectrum morphs smoothly.
By the end, a clean power-law regime appears.

That dashed line is the fitted slope.
Its exponent, alpha, gives a compact summary of scale behavior.

Now we repeat the same analysis across layers.
Different layers can have different alpha values,
but the same broad pattern appears again and again:
trained spectra are far from random and often close to power-law.

So this is not a one-layer accident.
It is a repeated structural effect."

---

## Scene 5 — FractalAttention (~85s)

"Finding two:
attention correlations become fractal-like.

Left is initialization.
Right is the trained model.
Same layer type, same visualization scale.

In the trained map, large coherent blocks appear.
When we zoom in, those blocks split into smaller blocks.
And zooming again reveals finer structure.

This is the qualitative hallmark of self-similarity:
organization that persists across magnification levels.

To ground the intuition, we compare with a Sierpinski triangle.
The neural map is not literally the same object,
but the geometric principle is similar:
pattern within pattern within pattern."

---

## Scene 6 — ScaledCopies (~80s)

"Finding three:
layers look like scaled copies.

First, this matrix shows pairwise similarity between layer spectra.
Values are high across many layer pairs.

Now look at the raw spectra together.
They are not identical, but they have a strongly related shape.

Here is the key test:
normalize each layer by its top singular value.

After normalization, the curves collapse much closer together.
Different magnitude, similar profile.

That means many layers seem to share a common template,
scaled up or down.

This is exactly the kind of self-similar organization we expect in fractal systems."

---

## Scene 7 — HurstExponent + FractalTrajectory (~95s)

"Finding four:
the training path itself is fractal.

We analyze how parameter displacement grows with scale.
On log-log axes, the slope gives the Hurst exponent, H.

If H equals one-half, behavior resembles an ordinary random walk.
Here, H is clearly above one-half.
That indicates persistence:
motion has directional memory across scales.

In 3D PCA space, we can visualize the trajectory.
Gold is the real SGD path.
Blue is a random-walk reference.

The real path is not just noise.
It has longer-range geometric coherence,
consistent with fractal-like dynamics in optimization."

---

## Scene 8 — Universality (~65s)

"Finding five:
the signatures persist across scale.

We compare small, medium, and larger model variants.
Different parameter counts, same style of measurements.

Across scales, key metrics stay in a similar regime:
Hurst behavior, cross-layer similarity, and mean power-law exponents.

So the effect is not tied to one exact model size.

That supports a stronger claim:
fractal organization may be a general property of how these networks train,
not a one-off artifact."

---

## Scene 9 — WhatDoesItMean (~70s)

"Put the pieces together.

Weights move from random toward power-law structure.
Correlations move from noise toward nested blocks.
Layers behave less like isolated parts and more like scaled copies.
Training trajectories show persistent, fractal-like motion.
And these patterns recur across model sizes.

So the conclusion is not just poetic.
Training appears to build geometry.

A useful mental model is this:
gradient descent is not only fitting data.
It is also sculpting a multi-scale structure inside parameter space.

And that structure often looks fractal."

---

## Optional Closing Tag (~12s)

"The same mathematics behind natural fractals
may also describe how neural networks organize themselves during learning."

