Morpheus - bio-inspired voxel cell morpohology simulator
========================================================

![](./docs/screenshot.png)

<!-- 

I want to be able to program cells to grow into custom 3D shapes (morphology)

This is really hard. Every cell can only run the same code. How can only local computation result in global form?

Here's a bio-inspired Lisp runtime + voxel cell simulator where it happens!


A single seed cell self-assembles (divides and replicates), grows outwards and upwards, and then knows when to stop at a certain height. 

This is what the code looks like. Every cell runs this exact code. It figures out where it is in the morphology, where to build, and what signals to send to coordinate with others.

It's meant to mimic real biology, where signalling molecules create gradient fields, cells inherit state from parents, and GRN circuit motifs allow for Turing-ish computation 
 -->

## About.

I want to program cells to grow into 3D shapes. Unlike human design, you cannot design parts and glue them together. The cell is both something which prints the material (it divides and replicates) as well as something that arranges it into geometry. 

Every cell runs the same code. We call this the local program. And yet the code somehow generates 3D structure.

Morpheus is a simulator for morpheological development programs. It runs a voxel cell model, where every cell runs a Lisp program, which gives it a small amount of internal state, turing-complete operations, the ability to send/receive signals (hormones), and naturally - the ability to divide and replicate. 

## Demo.

A simple example is the cylinder. How do we write a cell program that grows an organoid shaped like a cylinder?

```sh
cargo run --release -- cylinder.local
```

Cell program:
![](./docs/cylinder_code.png)

Organoid growth:
![](./docs/cylinder.gif)

Code explainer:
![](./docs/explainer.png)

## Other shapes.

```sh
cargo run --release -- tree.local      # trunk + spherical canopy
cargo run --release -- cone.local      # tapered cone shell
cargo run --release -- smiley.local    # flat smiley face (uses 4-anchor trilateration so cells decode their own (x,y))
```

## Headless mode.

For iterating on a `.local` program without launching the GUI: run the simulation to fixed-point (or a step budget) and dump orthographic projections to a PNG.

```sh
cargo run --release -- --headless tree.local
cargo run --release -- --headless --steps 200 cone.local
# writes /tmp/morpheus.png and prints the path
```

## Gun mode.

Once an organoid is grown, you can blast it and watch it heal. The simulation keeps ticking, so any program that uses `(neighbor-exists +z)` (rather than a one-shot `has-grown` latch) will re-extend through the crater.

- **left-click** in the 3D view: fire a spherical blast at the cell under the cursor
- **blast** slider in the toolbar: radius (1–20 voxels)
- **red wire sphere** at the cursor previews the destruction zone; cells inside it are tinted red live so you see exactly what'll go before you click
- **right-drag** / **middle-drag** / **shift + left-drag**: orbit camera
- **scroll**: zoom

Each blast also throws off a particle burst proportional to the number of cells destroyed.

