module DiffSharp.Quantum.AD.Float64


open DiffSharp.AD.Float64
open Microsoft.Quantum.Simulation.Core;

let inline sparseArray (i:int) (n:int) (el:D)=        
    DV.init n (fun j -> if j <> i then D 0.0 else el)

let inline q_grad (f:DV->D) (args:DV) : DV=        
    let n =args.Length
    let r=DV.init n (fun i ->
        let r1,r2=(f(args + (sparseArray i n (D System.Math.PI/2.0))),f(args - (sparseArray i n  (D System.Math.PI/2.0)))) in   D 0.5* (r1 - r2))
    r

type DVQ=
    interface dobj

    static member Expval(q:IOperationFactory,cirq:IAdjointable,measurement:Pauli[],nQubits:int64) (args:DV) :D=
        let ff(a:float[])= Quantum.VariationalQuantumCircuits.Expval.Run(q,cirq,new QArray<float>(a),new QArray<Pauli>(measurement),nQubits).Result
        let fd(a:DV)= DVQ.Expval (q,cirq, measurement,nQubits) a
        let fprime= DVQ.Expval (q,cirq, measurement,nQubits) |> q_grad
        let df(cp,ap:DV,at)=let t = fprime ap in at * t
        let r(a)= TraceOp.VarCirq_DV(a,fprime)
        DV.Op_DV_D(args,ff,fd,df,r)
   

   

