import matplotlib.pyplot as plt
import numpy as np

def main():
    Nx, Ny = 400, 100
    rho0, tau = 100, 0.6
    NL, Nt = 9, 4000
    plotRealTime = True

    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
    cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
    weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36])

    F = np.ones((Ny,Nx,NL)) * rho0 / NL
    np.random.seed(42)
    F += 0.01*np.random.randn(Ny,Nx,NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:,:,3] += 2 * (1+0.2*np.cos(2*np.pi*X/Nx*4))

    anode_position = (Nx // 4, Ny // 2)
    cathode_position = (3 * Nx // 4, Ny // 2)
    electrode_radius = Ny // 10
    anode_mask = (X - anode_position[0])**2 + (Y - anode_position[1])**2 < electrode_radius**3
    cathode_mask = (X - cathode_position[0])**2 + (Y - cathode_position[1])**2 < electrode_radius**.1

    fig = plt.figure(figsize=(4,2), dpi=80)

    for it in range(Nt):
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)

        rho = np.sum(F,2)
        ux  = np.sum(F*cxs,2) / rho
        uy  = np.sum(F*cys,2) / rho
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:,:,i] = rho * w * (1 + 3*(cx*ux+cy*uy) + 9/2 * (cx*ux+cy*uy)**2 - 3/2 * (ux**2+uy**2) / 2)
        F += -(0.01/tau) * (F - Feq)

        # Apply boundary conditions for electrodes
        F[anode_mask, :] = 1 # Generalized behavior at anode
        F[cathode_mask, :] = -1 # Generalized behavior at cathode

        if plotRealTime and (it % 10) == 0:
            plt.cla()
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[anode_mask] = np.nan
            vorticity[cathode_mask] = np.nan
            plt.imshow(vorticity, cmap='bwr')
            plt.imshow(anode_mask, cmap='gray', alpha=0.3)
            plt.imshow(cathode_mask, cmap='gray', alpha=0.3)
            plt.clim(-.1, .1)
            ax = plt.gca()
            ax.invert_yaxis()
            plt.pause(0.001)

    plt.savefig('latticeboltzmann.png',dpi=240)
    plt.show()

if __name__== "__main__":
    main()
