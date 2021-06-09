import numpy as np
import BinRealGenerator

class HistoryMatching:
    def __init__(self, real_generator, simulator, gamma = 1.0/6, beta = 0.1, max_iter = 1000):
        self.gamma = gamma
        self.best_estimate = 0
        self.best_estimate_val = 1000000
        self.a = 0.5
        self.beta = beta
        self.t = 0
        self.loss_prev = 10000
        self.start_est_l = 100000
        self.start_est = None
        self.realgen = real_generator
        self.sim = simulator
        self.max_iter = max_iter

    def compare_sim_obs(self, sim, obs):
        fopr = sim.numpy_vector("FOPR", report_only=True)
        dsum = np.linalg.norm(fopr - obs[0]) / obs[0].max()
        for x in range(1,5):
            wopr = sim.numpy_vector(f"WOPR:PROD{x}", report_only=True)
            d = np.linalg.norm(wopr - obs[x]) / obs[x].max()
            dsum += d
        return dsum

    def objective_spsa(self, d_obs, ksi, ksi_std):
        reals = []
        for ksi_ind in ksi:
            real = self.realgen.get_real(ksi_ind, gamma=0.8)
            reals.append(real)
        simulated_all = self.sim.run_simulator(reals)
        results = []
        for ksi_curr, simulated in zip(ksi,simulated_all):
            dsum = self.compare_sim_obs(simulated, d_obs)
            results.append(dsum)

        return results

    def spsa_step(self, current_estimate, observed_data, ksi_rand):
        self.a_t = self.a / ((self.t + 1)**self.gamma)
        self.beta_t =  self.beta / ((self.t + 1)**(self.gamma))

        delta = np.random.randint(0,2, current_estimate.shape) * 2 - 1

        self.left = current_estimate + delta * self.beta_t
        self.right = current_estimate - delta * self.beta_t

        loss_plus, loss_minus = self.objective_spsa(observed_data, [self.left, self.right], ksi_rand)
        self.loss_plus = loss_plus
        self.loss_minus = loss_minus
        if(self.start_est is None):
            self.start_est = current_estimate
            self.start_est_l = self.objective_spsa(observed_data, [self.start_est, self.start_est], ksi_rand)[0]

        g_t = (loss_plus - loss_minus) / (2.0 * delta * self.beta_t)

        c = (loss_plus + loss_minus) / 2

        if(self.best_estimate_val > c):
            self.best_estimate = current_estimate
            self.best_estimate_val = c

        current_estimate = current_estimate - self.a_t * g_t

        current_estimate[current_estimate > 10] = 10
        current_estimate[current_estimate < -10] = -10
        
        self.t += 1

        if(((loss_plus > self.start_est_l) and (loss_minus > self.start_est_l))):
            self.a = self.a * 0.5
            current_estimate = self.best_estimate      
            print("resetting to best: ", self.best_estimate_val)

        self.loss_prev = (loss_plus + loss_minus)/2

        return c, current_estimate

    def match_model(self, prior_params, observed_data):
        start_est = prior_params

        cur_est = (10000000, start_est)
        best_est = cur_est

        for iter in range(0,self.max_iter):
            cur_est = self.spsa_step(cur_est[1], observed_data, prior_params) 
            l_plus = self.loss_plus
            l_minus = self.loss_minus
            self.loss_mean = (l_plus + l_minus) / 2
            print("iter: ", iter, ": ", cur_est[0])
            if(self.loss_mean < 1.25):
                    return cur_est, iter
        return cur_est, -1