

## `cylinder.global`

```clojure
cylinder(radius=1.5,height=100)

def cylinder(r,h): 
    circle(r) * h

def circle(r): 
    (x-h)^2 + (y-k)^2 = r^2
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