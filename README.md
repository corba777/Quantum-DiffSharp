# Quantum-DiffSharp

Automatic differentiation of **variational quantum circuits** in F#, by grafting the
[parameter-shift rule](https://arxiv.org/abs/1811.11184) onto [DiffSharp](https://diffsharp.github.io/)'s
reverse-mode tape, with [Q#](https://learn.microsoft.com/azure/quantum/) / the Microsoft Quantum
Simulator as the circuit backend.

A hybrid classical–quantum cost function — classical pre/post-processing wrapped around a quantum
expectation value `⟨ψ(θ)|A|ψ(θ)⟩` — becomes an ordinary differentiable `DV -> D` function in DiffSharp,
so you can call `grad`, `diff`, nest derivatives, and compose with the rest of the F# AD machinery as
usual. The quantum part is exposed to the tape as a single custom primitive whose Jacobian is supplied
analytically by the parameter-shift rule.

This is a proof of concept / research exploration, not a maintained library. See the write-up:
[*Automatic Differentiation of Variational Quantum Circuits in Q# and F# (DiffSharp)*](https://medium.com/@corba77/automatic-differentiation-variational-quantum-circuits-in-q-and-f-diffsharp-d71b152249).

## Idea

A quantum simulator call is opaque to trace-based AD: you cannot push dual numbers through the unitary
evolution and the sampled measurement. But for a gate `e^{-iθP/2}` whose generator `P` has spectrum
`{-1, +1}`, the expectation `f(θ) = ⟨ψ(θ)|A|ψ(θ)⟩` is a pure sinusoid in `θ`, so its derivative is
recovered *exactly* (zero bias) from two evaluations at a finite shift:

```
∂f/∂θᵢ = ½ ( f(θ + (π/2)·eᵢ) − f(θ − (π/2)·eᵢ) )      (r = 1/2, s = π/2)
```

The derivative is therefore an algebraic identity on finite shifts rather than a limit — no
finite-difference step-size tradeoff. We register the quantum expectation as a custom DiffSharp
primitive with this gradient as its forward (jvp) and reverse (vjp) rule.

## How it works

**`DiffSharp.Quantum/Quantum.AD.Float64.fs`**
- `q_grad : (DV -> D) -> DV -> DV` — the per-parameter parameter-shift gradient above.
- `DVQ.Expval` — wraps the Q# `Expval` operation as a `DV -> D` primitive via DiffSharp's
  `DV.Op_DV_D` extension point:
  - primal: runs the circuit on the simulator and returns the measured expectation;
  - forward rule (jvp): `at · ∇f` (directional derivative against the incoming tangent);
  - reverse rule (vjp): a custom tape node carrying the primal point and `q_grad`.

**Core fork — `DiffSharp/src/DiffSharp/AD.Float64.fs`**
Because DiffSharp's `TraceOp` is a closed discriminated union, the reverse node has to be added to the
core. The entire change is three lines:
- a new case `VarCirq_DV of DV * (DV -> DV)` (the primal point and its gradient function);
- in the reset sweep: recurse into the parent;
- in the reverse sweep: `pushRec ((bxv (dA * f(a.P)) a) :: t)` — i.e. propagate adjoint
  `dA · ∇f(θ)`, the vector–Jacobian product of the quantum primitive.

Everything else lives in the extension module; this is the `custom_vjp` pattern, hand-rolled in F#.

**`VariationalQuantumCircuits/Operations.qs`**
- `Expval(cirq, args, measurement, nQubits)` — `2 · EstimateFrequencyA(…, 1000 shots) − 1`,
  the expectation of a `±1`-valued Pauli observable estimated by sampling.
- `Circuit` — a sample 2-qubit ansatz: `Rz(θ₀, q₀); CNOT(q₀, q₁); Ry(θ₁, q₁)`.

## Example

```fsharp
open DiffSharp.AD.Float64
open Microsoft.Quantum.Simulation.Simulators
open DiffSharp.Quantum.AD.Float64

use qsim = new QuantumSimulator()

let circ (args: DV) : D =
    let cirq = qsim.Get<IAdjointable, Quantum.VariationalQuantumCircuits.Circuit>()
    DVQ.Expval (qsim, cirq, [| Pauli.PauliZ; Pauli.PauliZ |], 2L) args   // ⟨Z⊗Z⟩

let cost (args: DV) : D =
    D.Sin (D.Abs (circ args)) - D 1.0

let args  = DV [| System.Math.PI / 4.0; 0.7 |]
let dcost = grad cost
let dr    = dcost args        // ≈ DV [| 0.0; -0.4596 |]
```

The first component is `0`: `Rz` before a `Z`-basis measurement leaves `⟨Z⊗Z⟩` invariant, so that
gradient is identically zero, and the parameter-shift node returns it exactly. Because the observable
is sampled (1000 shots), the nonzero component carries shot noise and will vary slightly between runs.

## Scope and caveats

- **Spectral assumption.** `q_grad` hard-codes the two-term rule (`r = 1/2`, `s = π/2`), which is exact
  *only* for gates whose generators have spectrum `{-1, +1}` (`Rx`, `Ry`, `Rz`, …). For generators with
  a richer spectrum — e.g. the continuous-variable Squeezing and Displacement gates — this rule is
  silently biased and needs the corresponding multi-term / generalized shift rule. The current code
  covers Phase Rotation and Beamsplitter; Squeezing/Displacement are not yet handled.
- **Bias vs. variance.** Parameter-shift removes the *bias* of the gradient (no truncation, no
  step-size tradeoff). It does nothing about *variance*: each evaluation is a 1000-shot Monte-Carlo
  estimate, so the returned gradient is unbiased but noisy. On real (NISQ) hardware the shot budget,
  not the differentiation rule, is the limiting factor.
- Forward and reverse rules are implemented for the scalar-output (`DV -> D`) case used here.

## Build

- .NET / F#, `netstandard2.1`.
- Microsoft Quantum Development Kit `0.10.1912.501` (`Microsoft.Quantum.Development.Kit`).
- A vendored DiffSharp fork (in `DiffSharp/`) with the `VarCirq_DV` tape node added; OpenBLAS backend.
- Projects: `DiffSharp.Quantum` (AD glue) → references the DiffSharp fork and
  `VariationalQuantumCircuits` (Q#). Run `TestApp` for the example above.

## Background and references

- Parameter-shift rule: Schuld, Bergholm, Gogolin, Izaac, Killoran, *Evaluating analytic gradients on
  quantum hardware*, [arXiv:1811.11184](https://arxiv.org/abs/1811.11184).
- Companion note on the AD/control-flow side, with the SDG reading:
  [*Some Differentiable Programming pitfalls*](https://medium.com/@corba77/some-differentiable-programming-pitfalls-with-examples-in-python-google-jax-julia-zygote-f83c8c9f777b).
- Categorical semantics of AD that motivate the dual-number / tangent view: Ehrhard–Regnier
  (differential λ-calculus), Blute–Cockett–Seely (Cartesian differential categories),
  Cockett–Cruttwell (*Differential structure, tangent structure, and SDG*), Cruttwell–Gallagher–MacAdam
  (tangent categories for differentiable programming). See the
  [DifferentiableProgramming](https://github.com/corba777/DifferentiableProgramming) repo.

## License

DiffSharp under `DiffSharp/` retains its original (BSD) license. New code in `DiffSharp.Quantum`,
`VariationalQuantumCircuits`, and `TestApp` may be used freely; attribution welcome.
