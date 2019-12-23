namespace Quantum.VariationalQuantumCircuits
{
    open Microsoft.Quantum.Characterization;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Canon;
    
	operation Expval(cirq:((Double[],Qubit[])=>Unit is Adj),args:Double[],measurement:Pauli[], nQubits:Int) : Double {
		let rs= EstimateFrequencyA(cirq(args,_),Measure(measurement,_),nQubits,1000);
        return 2.*rs-1.;
    }

    operation Circuit(args:Double[],q:Qubit[]): Unit is Adj{
		Rz(args[0],q[0]);
        CNOT(q[0],q[1]);
        Ry(args[1],q[1]);
	}
}
