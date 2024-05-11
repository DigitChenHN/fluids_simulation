import numpy as np  
import matplotlib.pyplot as plt  
  
G = np.array([0, 12000 * -9.8])  
GAS_CONST = 2000.  
H = 32  
HSQ = H * H  
MASS = 65.    
DT = 0.008  
 

  
M_PI = np.pi  
POLY6 = 315 / (65. * M_PI * pow(H, 9.))  
SPIKY_GRAD = -45 / (M_PI * pow(H, 6.))  

S_CORR_DELTA_Q = 0.3
S_CORR_K = 0.1
S_CORR_N = 4
S_CORR_CONST = 1 / (POLY6 * (HSQ - HSQ * S_CORR_DELTA_Q * S_CORR_DELTA_Q) ** 3)
LAMBDA_EPSILON = 200
  
REST_DENS = MASS * (POLY6 * (HSQ) ** 3) * 0.5   
EPS = 16.  
BOUND_DAMPING = -0.5  
VIEW_HEIGHT = 800  
VIEW_WIDTH = 800  
  
  
class Particles:  
    def __init__(self, x: np.ndarray):  
        self.x = x  
        self.x_new = np.zeros_like(x)
        self.v = np.zeros_like(x)  
        self.f = np.zeros_like(x)  
        self.rho = np.zeros(len(x))  
        self.lamb = np.zeros(len(x))
        self.dx = np.zeros_like(x) ## delta p in the original paper
  
    def draw_particles(self, ax):  
        particles = self  
        for i, (x, y) in enumerate(particles.x):  
            circ = plt.Circle((x, y), EPS / 2, color='b', alpha=0.5)  
            ax.add_artist(circ)  
            ax.set_aspect("equal")  
            ax.set_xlim([0, 800])  
            ax.set_ylim([0, 800])  
  
    @classmethod  
    def initSPH(cls):  
        y = EPS / 2  
        positions = []  
        while y < VIEW_HEIGHT - EPS * 2.:  
            x = EPS / 2 
            while x <= VIEW_WIDTH:  
                if (x - 400) ** 2 + (y - 400) ** 2 <= 10000:
                # if 2 * np.abs(x - 400) ** 2 - 2 * np.abs(x - 400) * (y - 200) + (y - 200) ** 2 <= 30000:  
                    jitter = np.random.randn()  
                    positions.append([x + jitter, y])  
                    # positions.append([x, y])
                x += EPS 
            y += EPS  
        print(f"Initializing heartbreak with {len(positions)} particles")  
        return cls(x=np.array(positions))
    
    def apply_external_forces(self):
        '''
        The only external forces concerned here is the gravity.
        '''
        particles = self
        particles.f = np.full(particles.f.shape, G)
        for i, pos in enumerate(particles.x):  
            particles.v[i, :] += DT * particles.f[i]
            particles.x_new[i, :] = particles.x[i, :] + DT * particles.v[i, :]  
        
    def computeDensity(self):  
        particles = self  
        for i, particle_i_pos in enumerate(particles.x):  
            particles.rho[i] = 0.  
            for j, particle_j in enumerate(particles.x):  
                rij = particles.x[j, :] - particles.x[i, :]  
                r2 = np.sum(rij * rij)  
                if r2 < HSQ:  
                    particles.rho[i] += MASS * POLY6 * np.power(HSQ - r2, 3.)  
            
    def box_collision(self):  
        particles = self  
        for i, pos in enumerate(particles.x):  

            if pos[0] - EPS < 0.0:  
                # particles.v[i, 0] *= BOUND_DAMPING  
                particles.x_new[i, 0] = EPS  
            if pos[0] + EPS > VIEW_WIDTH:  
                # particles.v[i, 0] *= BOUND_DAMPING  
                particles.x_new[i, 0] = VIEW_WIDTH - EPS  
            if pos[1] - EPS < 0.0:  
                # particles.v[i, 1] *= BOUND_DAMPING    
                particles.x_new[i, 1] = EPS  
            if pos[1] + EPS > VIEW_HEIGHT:  
                # particles.v[i, 1] *= BOUND_DAMPING  
                particles.x_new[i, 1] = VIEW_HEIGHT - EPS  

    def solve_iter(self):
        particles = self
        particles.computeDensity()
        for i, pos in enumerate(particles.x):
            sum_grad_pk_C_sq = 0.
            sum_grad_pi_C = np.array([0., 0.])

            for j, pos in enumerate(particles.x):
                rij = particles.x[j, :] - particles.x[i, :]  
                r2 = np.sum(rij * rij)  
                if r2 < HSQ:
                    r_norm = np.linalg.norm(rij)
                    if r_norm == 0.:
                        continue
                    else:
                        grad = rij / r_norm * SPIKY_GRAD * np.power((H - r_norm), 2) / REST_DENS * MASS
                        sum_grad_pi_C += grad
                        sum_grad_pk_C_sq += np.linalg.norm(grad) ** 2

            C_i = particles.rho[i] / REST_DENS -1
            particles.lamb[i] = -C_i / (sum_grad_pk_C_sq + np.power(np.linalg.norm(sum_grad_pi_C), 2) + LAMBDA_EPSILON)
        
        for i, pos in enumerate(particles.x):
            particles.dx[i] = np.array([0., 0.])
            for j, pos in enumerate(particles.x):
                rij = particles.x[j, :] - particles.x[i, :]  
                r2 = np.sum(rij * rij)
                if r2 < HSQ:
                    r_norm = np.linalg.norm(rij)
                    if r_norm == 0.:
                        continue
                    else:
                        s_corr = -S_CORR_K * (POLY6 * np.power((HSQ - r2), 3) * S_CORR_CONST) ** S_CORR_N
                        particles.dx[i] += (particles.lamb[i] + particles.lamb[j] + s_corr) * SPIKY_GRAD * np.power((H - r_norm), 2) * rij / r_norm
        
        particles.x_new = particles.x_new + particles.dx / REST_DENS
        # for i, pos in enumerate(particles.x):
        #     particles.x_new[i] += particles.dx[i] 
        
        particles.v = (particles.x_new - particles.x) / DT

    def update(self):  
        self.apply_external_forces()
        self.box_collision()
        self.solve_iter()
        self.x = self.x_new


if __name__ == "__main__":

    particles = Particles.initSPH() 
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))  
    fig.set_tight_layout(True)  
    particles.draw_particles(ax=ax)
    plt.savefig(f'./frames_PBF/c_damped_00.png')
    plt.close()

    for i in range(61):  
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))  
        fig.set_tight_layout(True)  
        particles.update()  
        particles.draw_particles(ax=ax)  
        plt.savefig(f'./frames_PBF/c_damped_{i}.png')  
        plt.close()