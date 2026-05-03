
## Cell substrate.


## 3D voxel world.

A bounded 3D integer lattice (x, y, z) ∈ ℤ³, voxel scale 1.0. Each voxel is either empty or contains exactly one cell. Cells have 6-connected neighbors (±x, ±y, ±z). 

Assumptions: base centered at the origin, axis along +z, voxel scale 1.0, single seed at (0, 0, 0) with optional polarization (a unit vector indicating the cell's "up").

**Global programs**. Every global program is a predicate `P(x, y, z)` over absolute voxel coordinates, and rendering is "fill every voxel where P is true.".

**Local programs**. Every local program is a single deterministic procedure that runs identically inside every cell, every tick.

Time advances in discrete synchronous ticks: every cell reads its inputs, computes, and writes its outputs in lockstep. The simulation runs from a single seed cell at the origin until either a fixed-point is reached (no cell changes state, no division occurs) or a step budget is exhausted.


## Local program runtime.

- State. Each cell has named local variables of types Boolean, integer, and float, with assignment.
- Control flow. if, begin, function definitions, comparisons.
- Gradients. (emit-gradient name value) and (read-gradient name). The substrate handles propagation between cells.
- Replication. (replicate-toward direction) where direction is one of six axis-aligned options. The substrate creates a new cell in that voxel if empty, ignores if full. New cells inherit the program but start with default state.
- Identity / initial conditions. Some way to mark the initial seed cell as different. Could be is-seed set as initial state.

Gradients. A cell can (emit-gradient name) to become a source of the named gradient. The substrate computes a continuous concentration field by summing contributions from all sources: g(c) = Σ_s f(||c - s||) where f is a fixed isotropic falloff function (e.g., exp(-d/λ)). The cell reads (read-gradient name) to get the field value at its own location. Cells have named gradients with separate fields per name. Anisotropic.


## `cylinder.global`

```clojure
cylinder(radius=1.5,height=100)

def cylinder(r,h): 
    circle(r) * h

def circle(r): 
    (x-h)^2 + (y-k)^2 = r^2
```

Concrete predicate:

```
is_in_cylinder(x, y, z) := 
    x² + y² < 1.5²  ∧  0 ≤ z < 100
```

## `cylinder.local`

```clojure
;; CYLINDER local program
;; Substrate: emit-gradient, read-gradient, replicate-toward, internal state, if/comparators
;; Initial seed at origin with is-seed=#t, polarized along +z

(define is-seed #f)
(define is-axis #f)
(define inside #f)
(define has-grown-axis #f)
(define has-grown-radial #f)

;; Axis: seed and axis cells emit a marker, replicate +z up to height h
(if is-seed (set! is-axis #t))

(if is-axis
    (emit-gradient g-axis 0))

(if (and is-axis
         (< (read-gradient g-axis-length) 100)
         (not has-grown-axis))
    (begin (replicate-toward +z)
           (set! has-grown-axis #t)))

;; New +z child detects axis membership via g-axis from -z neighbor
(if (and (not is-seed)
         (= (read-gradient g-axis) 0))
    (set! is-axis #t))

;; Radial: axis cells source g-radial, others fill by propagation
(if is-axis
    (emit-gradient g-radial 0))

(if (< (read-gradient g-radial) 1.5)
    (set! inside #t))

;; Inside cells replicate laterally; outside cells (boundary scaffolding) stop
(if (and inside (not has-grown-radial))
    (begin (replicate-toward +x)
           (replicate-toward -x)
           (replicate-toward +y)
           (replicate-toward -y)
           (set! has-grown-radial #t)))
```




### Notes on local program runtime.

The cell can compute: sums, differences, thresholds against constants, Boolean combinations. It cannot compute: products of variables, squares of variables, square roots, division by variables.

Useful additions that aren't strictly minimum but make programs cleaner

- Directional gradients. Already discussed — without this, anything involving directional structure (axes, polarities) is very awkward.
- Self-destruct. (die) — cell removes itself from the substrate. Cleans up scaffolding.
- Neighbor occupancy sensing. (neighbor-occupied? direction) — needed to avoid pointless replication attempts and to detect "boundary" conditions.
- Step counter. A globally synchronized step number that all cells can read. Useful for phasing — "do axis growth in steps 0–100, then radial fill in steps 100–200." Without this, phase ordering has to be encoded with gradient propagation as a clock, which works but is awkward.
