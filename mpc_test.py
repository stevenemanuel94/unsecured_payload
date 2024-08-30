import numpy as np
from pydrake.all import (
    Binding,
    MathematicalProgram,
    VPolytope,
    HPolyhedron,
    Solve,
    eq,
)

def compute_MPC(A, B, C, D, dt, N, x_desired, x0, Q, r, u_bounds, constrained):
    #discretize the system dynamics via Euler Metod
    A = np.eye(4) + dt * A
    B = dt * B

    #initialize mathematical program
    prog = MathematicalProgram()

    #create polytopes - this code might be redundant, I first turned the bounds form the model into a Vpolytope
    #second, convert the vPolytope to an Hpolyhedromn to be used as linear constraints
    vRep = VPolytope(u_bounds)
    hRep = HPolyhedron(vRep)

    # Create decision variables
    #2 control inputs - x and y force
    #4 states, x, y, xdot, ydot (in that order)
    u = prog.NewContinuousVariables(2, N - 1, "u")
    x = prog.NewContinuousVariables(4, N, "x")

    #constrain the initial condition
    prog.AddBoundingBoxConstraint(x0, x0, x[:, 0])

    #create constraints for each point in horizon
    for n in range(N - 1):
        #dynamics constraints
        prog.AddConstraint(eq(x[:, n + 1], A.dot(x[:, n]) + B.dot(u[:, n])))

        #add cost for x input
        prog.AddQuadraticCost(r*u[0, n] ** 2, True)

        #add cost for y input
        prog.AddQuadraticCost(r*u[1, n] ** 2, True)

        #activate the force constraints if the experiment is being run w/constraints
        if constrained:
            for m in range(hRep.A().shape[0]):
                #constraints in form Au<=b
                prog.AddLinearConstraint(hRep.A()[m,0] * u[0,n] + hRep.A()[m,1] * u[1,n] <= hRep.b()[m])

    #add constraints for state deviation from desire trajectory
    prog.AddQuadraticErrorCost(Q, x_desired[:, 0].T, x[0, :])
    prog.AddQuadraticErrorCost(Q, x_desired[:, 1].T, x[1, :])

    #solve the mathematical program
    result = Solve(prog)

    #Check if optimization w
    assert result.is_success(), "Optimization failed"

    #get the control trajectory
    u_sol = result.GetSolution(u)

    #return the first control input of the optimized control sequence
    return u_sol[:,0]


