import numpy as np
from pyfeng import sv_abc as sv
from pyfeng import bsm


#### Use of RN generation spawn:
# 0: simulation of variance (gamma/ncx2/normal)


class GarchMcTimeDisc(sv.SvABC, sv.CondMcBsmABC):
	"""
	Garch model with conditional Monte-Carlo simulation
	The SDE of SV is: dv_t = mr * (theta - v_t) dt + vov * v_t dB_T
	"""

	model_type = "LamSABR"
	vol_process = True
	scheme = 1  #

	def set_num_params(self, n_path=10000, dt=0.05, rn_seed=None, antithetic=True, scheme=1):
		"""
		Set MC parameters

		Args:
			n_path: number of paths
			dt: time step
			rn_seed: random number seed
			antithetic: antithetic
			scheme: 0 for Euler, 1 for Milstein (default)

		References:
			- Andersen L (2008) Simple and efficient simulation of the Heston stochastic volatility model. Journal of Computational Finance 11:1–42. https://doi.org/10.21314/JCF.2008.189
		"""
		super().set_num_params(n_path, dt, rn_seed, antithetic)
		self.scheme = scheme

	def vol_step_euler(self, dt, vol_0, model_replace='original', milstein=True):
		"""
		Euler/Milstein Schemes:
		sigma_(t+dt) = sigma_t + mr * (theta - sigma_t) * dt + vov * sigma_t Z * sqrt(dt) + (vov^2/2) sigma_t (Z^2-1) dt

		Args:
			vol_0: initial variance
			dt: delta t, time step
			model_replace: if 'original': sigma_(t+dt) = sigma_t + mr * (theta - sigma_t) * dt + vov * sigma_t Z * sqrt(dt) + (vov^2/2) sigma_t (Z^2-1) dt
			              if 'exp': theta + (sigma_t - theta) * exp(-mr * dt) + vov * sigma_t Z * sqrt(dt) + (vov^2/2) sigma_t (Z^2-1) dt
			              if 'exact' :  theta * (1-exp(-mr * dt)) + sigma_t * exp(-mr * dt) + vov * sigma_t Z * sqrt(dt) + (vov^2/2) sigma_t (Z^2-1) dt

		Returns: Variance path (time, path) including the value at t=0
		"""

		zz = self.rv_normal(spawn=0)

		mr, theta, vov = self.mr, self.theta, self.vov
		mr_square, theta_square, vov_square = np.square(mr), np.square(theta), np.square(vov)
		vol_0_square = np.square(vol_0)

		# exact mean and variance
		exact_mean = theta * (1 - np.exp(-mr * dt)) + vol_0 * np.exp(mr * dt)
		if mr / vov_square == 1 / 2:
			exact_mean_square = 2 * theta_square * (
					np.exp(-mr * dt) + mr * dt - 1) + 2 * theta * vol_0 * (1 - np.exp(-mr * dt)) + vol_0_square
		elif mr / vov_square == 1:
			exact_mean_square = 2 * theta_square - 2 * (mr * dt + 1) * theta_square * np.exp(
				-mr * dt) + 2 * mr * theta * vol_0 * dt * np.exp(-mr * dt) + vol_0_square * np.exp(-mr * dt)
		else:
			exact_mean_square = (2 * mr * theta_square) / (2 * mr - vov_square) \
								- (2 * mr * theta_square) / (mr - vov_square) * np.exp(-mr * dt) \
								+ (2 * mr_square * theta_square) / (mr - vov_square) / (
										2 * mr - vov_square) * np.exp(-2 * mr * dt + vov_square * dt) \
								+ 2 * mr * theta * vol_0 * np.exp(-mr * dt) / (mr - vov_square) \
								- 2 * mr * theta / (mr - vov_square) * np.exp(-mr * dt) / (mr - vov_square) \
								+ vol_0_square * np.exp(-mr * dt) / (mr - vov_square)
		exact_var = exact_mean_square - np.square(exact_mean)

		# time discrimination for vol_t
		if model_replace == 'original':
			vol_t = vol_0 + mr * (theta - vol_0) * dt + vov * vol_0 * np.sqrt(dt) * zz
		elif model_replace == 'exp':
			vol_t = theta + (vol_0 - theta) * np.exp(-mr * dt) + vov * vol_0 * np.sqrt(dt) * zz
		elif model_replace == 'exact':
			vol_t = exact_mean + exact_var * zz
		else:
			raise Exception('model_replace input wrong')

		if milstein:
			if model_replace != 'exact':
				vol_t += (vov ** 2 / 2) * vol_0 * dt * (zz ** 2 - 1)

		# although rare, floor at zero
		vol_t[vol_t < 0] = 0.0
		return vol_t

	def cond_spot_step(self, dt, vol_0, beta):
		zz = self.rv_normal(spawn=0)

		if self.scheme in [0, 1]:
			milstein = (self.scheme == 1)
			vol_t = self.vol_step_euler(dt, vol_0, milstein=milstein)

			mean_vol = (vol_0 + vol_t) / 2
			mean_inv_vol = (1 / vol_0 + 1 / vol_t) / 2

		else:
			raise ValueError(f'Invalid scheme: {self.scheme}')

		return vol_t, mean_vol

	def cond_spot_sigma(self, texp, vol_0):
		tobs = self.tobs(texp)
		dt = np.diff(tobs, prepend=0)
		n_dt = len(dt)

		vol_t = np.full(self.n_path, vol_0)
		avgvol = np.zeros(self.n_path)

		for i in range(n_dt):
			vol_t, avgvol_inc, avgivol_inc = self.cond_states_step(dt[i], vol_t)
			avgvol += avgvol_inc * dt[i]

		avgvol /= texp

	# spot_cond = (vol_t - vol_0 - self.mr * self.theta * dt) / self.vov + \
	# 			(-self.mr * self.theta * avgivol / self.vov \
	# 			 + (self.mr / self.vov + self.vov / 4) * avgvol - self.rho * avgvar / 2) * texp
	# np.exp(self.rho * spot_cond, out=spot_cond)
	#
	# cond_sigma = np.sqrt((1.0 - self.rho ** 2) / var_0 * avgvar)

	# return spot_cond, cond_sigma

	def return_var_realized(self, texp, cond):
		return None


Path = GarchMcTimeDisc(sigma=0.2, theta=0.15)
vol1 = Path.set_num_params
vol21 = Path.vol_step_euler(0.25, 0.2, 'original', 1)
vol22 = Path.vol_step_euler(0.25, 0.2, 'exp', 1)
vol23 = Path.vol_step_euler(0.25, 0.2, 'exact', 1)
print(vol21)
