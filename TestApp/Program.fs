// Learn more about F# at http://fsharp.org

open System
open DiffSharp.AD.Float64
open Microsoft.Quantum.Simulation;
open Microsoft.Quantum.Simulation.Core;
open Microsoft.Quantum.Simulation.Common;
open Microsoft.Quantum.Simulation.Simulators;
open DiffSharp.Quantum.AD.Float64

[<EntryPoint>]
let main argv =

    use qsim=new QuantumSimulator()

    let circ(args:DV):D=
        let cirq=qsim.Get<IAdjointable,Quantum.VariationalQuantumCircuits.Circuit>()
        let res= DVQ.Expval(qsim,cirq,[|Pauli.PauliZ;Pauli.PauliZ|],2L) args
        res

    let args=DV [|Math.PI/4.0; 0.7|]
    
    let r= circ(args)

    let cost(args:DV) :D=
        let res= circ args
        D.Sin(D.Abs(res)) - D 1.0            

    let dcost= grad cost

    let dr= dcost (args)
    0 // return an integer exit code
