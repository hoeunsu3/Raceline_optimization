import numpy as np
import casadi as cs
import matplotlib.pyplot as plt


def fnc_arclength(path):
    '''Calculate cumulative arclength of track'''
    s = [0.0]
    s.extend(np.cumsum(np.sqrt(np.diff(path[:, 0])**2 + np.diff(path[:, 1])**2)))
    return s

def fnc_interparc(data, arclength, ds):
    '''Interpolate the path by same arc-length 1d'''
    s_ds = np.arange(0, arclength[-1], ds)
    
    return np.interp(s_ds, arclength, data)

def main():
    # ==================== Read Track Data ====================
    data = np.loadtxt('./paths/Catalunya.csv', skiprows=1, delimiter=',')
    
    x = data[:,0] # center line x [m]
    y = data[:,1] # center line y [m]
    twr = data[:,2] # track width right [m]
    twl = data[:,3] # track width left [m]

    # ==================== Interpolate same distance ====================
    ds = 1
    arclength = fnc_arclength(data[:,:2])
    x_interp = fnc_interparc(x, arclength, ds=ds) # ds = segment length [m]
    y_interp = fnc_interparc(y, arclength, ds=ds)
    twr_interp = fnc_interparc(twr, arclength, ds=ds)
    twl_interp = fnc_interparc(twl, arclength, ds=ds)
    
    # ==================== Inner, outer track ====================
    inner = np.zeros((len(x_interp)-1, 2))
    outer = np.zeros((len(x_interp)-1, 2))

    for i in range(len(x_interp)-1):
        dx = x_interp[i+1] - x_interp[i]
        dy = y_interp[i+1] - y_interp[i]
        psi = np.arctan2(dy, dx)

        inner[i,0] = x_interp[i] - twr_interp[i] * np.sin(-psi)
        inner[i,1] = y_interp[i] - twr_interp[i] * np.cos(-psi)

        outer[i,0] = x_interp[i] + twl_interp[i] * np.sin(-psi)
        outer[i,1] = y_interp[i] + twl_interp[i] * np.cos(-psi)

    # ==================== track normal vector ====================
    normal_x = outer[:,0] - inner[:,0]
    normal_y = outer[:,1] - inner[:,1]

    # ==================== Quadratic matrices ====================
    n = len(normal_x) # number of segments
    H = cs.DM.zeros(n,n)
    B = cs.DM.zeros(n)

    for i in range(1, n - 1):
        # First row
        H[i-1,i-1] = H[i-1,i-1] + normal_x[i-1]**2            + normal_y[i-1]**2
        H[i-1,i]   = H[i-1,i]   - 2*normal_x[i-1]*normal_x[i] - 2*normal_y[i-1]*normal_y[i]
        H[i-1,i+1] = H[i-1,i+1] + normal_x[i-1]*normal_x[i+1] + normal_y[i-1]*normal_y[i+1]

        # Second row
        H[i,i-1] = H[i,i-1] - 2*normal_x[i-1]*normal_x[i] - 2*normal_y[i-1]*normal_y[i]
        H[i,i]   = H[i,i]   + 4*normal_x[i]**2            + 4*normal_y[i]**2
        H[i,i+1] = H[i,i+1] - 2*normal_x[i]*normal_x[i+1] - 2*normal_y[i]*normal_y[i+1]
        
        # Third row
        H[i+1,i-1] = H[i+1,i-1] + normal_x[i-1]*normal_x[i+1] + normal_y[i-1]*normal_y[i+1]
        H[i+1,i]   = H[i+1,i]   - 2*normal_x[i]*normal_x[i+1] - 2*normal_y[i]*normal_y[i+1]
        H[i+1,i+1] = H[i+1,i+1] + normal_x[i+1]**2            + normal_y[i+1]**2

    H = 2*H

    for i in range(1, n - 1):
        B[i-1]  = B[i-1] + 2*(inner[i+1,0] + inner[i-1,0] - 2*inner[i,0])*normal_x[i-1] + 2*(inner[i+1,1] + inner[i-1,1] - 2*inner[i,1])*normal_y[i-1]
        B[i]    = B[i]   - 4*(inner[i+1,0] + inner[i-1,0] - 2*inner[i,0])*normal_x[i]   - 4*(inner[i+1,1] + inner[i-1,1] - 2*inner[i,1])*normal_y[i]
        B[i+1]  = B[i+1] + 2*(inner[i+1,0] + inner[i-1,0] - 2*inner[i,0])*normal_x[i+1] + 2*(inner[i+1,1] + inner[i-1,1] - 2*inner[i,1])*normal_y[i+1]

    # ==================== Constraints ====================
    # lb = cs.DM.zeros(n) + 1/12
    # ub = cs.DM.ones(n) - 1/12
    lb = cs.DM.zeros(n)
    ub = cs.DM.ones(n)

    Aeq = cs.DM.zeros(1,n)
    Aeq[0] = 1
    Aeq[-1] = -1 # start alpha = end alpha
    beq = 0
    
    # ==================== QP solver ====================
    qp = {}
    qp['h'] = H.sparsity()
    qp['a'] = Aeq.sparsity()
    S = cs.conic('S', 'qpoases', qp)
    
    res = S(h=H, g=B, a=Aeq, lba=beq, uba=beq, lbx=lb, ubx=ub)
    alpha_opt = res['x']

    # ==================== Plot ====================
    MCP = np.zeros((n-1, 2))

    for i in range(n-1):
        dx = inner[i+1, 0] - inner[i, 0]
        dy = inner[i+1, 1] - inner[i, 1]
        psi = np.arctan2(dy, dx)

        MCP[i,0] = inner[i,0] + alpha_opt[i] * (twr_interp[i] + twl_interp[i]) * np.sin(-psi)
        MCP[i,1] = inner[i,1] + alpha_opt[i] * (twr_interp[i] + twl_interp[i]) * np.cos(-psi)

    plt.plot(inner[:,0], inner[:,1], label='inner')
    plt.plot(outer[:,0], outer[:,1], label='outer')
    plt.plot(MCP[:,0], MCP[:,1], '--', label='MCP')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()