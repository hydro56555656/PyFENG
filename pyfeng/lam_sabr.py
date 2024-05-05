import numpy as np
import pandas as pd

from pyfeng import sv_abc as sv
from pyfeng import bsm


#### Use of RN generation spawn:
# 0: simulation of variance (gamma/ncx2/normal)


class LambdaSABRhMcTimeDisc(sv.SvABC, sv.CondMcBsmABC):
	"""
	Garch model with conditional Monte-Carlo simulation
	The SDE of SV is: dv_t = mr * (theta - v_t) dt + vov * v_t dB_T, v as sigma
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

	def exact_mean_variance(self, vol_0, dt):
		mr, theta, vov = self.mr, self.theta, self.vov
		mr_square, theta_square, vov_square = np.square(mr), np.square(theta), np.square(vov)
		vol_0_square = np.square(vol_0)

		# exact mean and variance
		exact_mean = theta * (1 - np.exp(-mr * dt)) + vol_0 * np.exp(-mr * dt)
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
								- 2 * mr * theta * vol_0 / (mr - vov_square) * np.exp(-2 * mr * dt + vov_square * dt) \
								+ vol_0_square * np.exp(-2 * mr * dt + vov_square * dt)
		exact_var = exact_mean_square - np.square(exact_mean)
		return exact_mean, exact_var

	def vol_step_euler(self, dt, vol_0, model_replace='original', milstein=True):
		"""
		Euler/Milstein Schemes:
		sigma_(t+dt) = sigma_t + mr * (theta - sigma_t) * dt + vov * sigma_t Z * sqrt(dt) + (vov^2/2) sigma_t (Z^2-1) dt

		Args:
			vol_0: initial volatility
			dt: delta t, time step
			model_replace: if 'original': sigma_(t+dt) = sigma_t + mr * (theta - sigma_t) * dt + vov * sigma_t Z * sqrt(dt) + (vov^2/2) sigma_t (Z^2-1) dt
			              if 'exp': theta + (sigma_t - theta) * exp(-mr * dt) + vov * sigma_t Z * sqrt(dt) + (vov^2/2) sigma_t (Z^2-1) dt
			              if 'exact' :  theta * (1-exp(-mr * dt)) + sigma_t * exp(-mr * dt) + vov * sigma_t Z * sqrt(dt) + (vov^2/2) sigma_t (Z^2-1) dt

		Returns: Variance path (time, path) including the value at t=0
		"""

		zz = self.rv_normal(spawn=0)

		mr, theta, vov = self.mr, self.theta, self.vov
		exact_mean, exact_var = self.exact_mean_variance(vol_0, dt)

		# time discrimination for vol_t
		if model_replace == 'original':
			vol_t = vol_0 + mr * (theta - vol_0) * dt + vov * vol_0 * np.sqrt(dt) * zz
		elif model_replace == 'exp':
			vol_t = theta + (vol_0 - theta) * np.exp(-mr * dt) + vov * vol_0 * np.sqrt(dt) * zz
		elif model_replace == 'exact':
			vol_t = exact_mean + np.sqrt(exact_var) * zz
		else:
			raise Exception('model_replace input wrong')

		if milstein:
			if model_replace != 'exact':
				vol_t += (vov ** 2 / 2) * vol_0 * dt * (zz ** 2 - 1)

		# although rare, floor at zero
		vol_t[vol_t < 0] = 0.0
		return vol_t

	def spot_step_euler(self, dt, beta, vol_t, spot_0):
		zz = self.rv_normal(spawn=0)

		drift = vol_t * zz * np.sqrt(dt)
		if beta == 0:
			spot_t = spot_0 + drift
		elif beta == 1:
			spot_t_log = np.log(spot_0) + drift - 1 / 2 * vol_t ** 2 * dt
			spot_t = np.exp(spot_t_log)
		else:
			spot_t_div = spot_0 / (-beta + 1) + drift - 1 / 2 * vol_t ** 2 * spot_0 ** (-beta + 1) * dt
			spot_t = (-beta + 1) * spot_t_div

		return spot_t

	def cond_vol_step(self, texp, vol_0, spot_0, beta, model='exp'):
		tobs = self.tobs(texp)
		dt = np.diff(tobs, prepend=0)
		n_dt = len(dt)

		vol_t = np.full(self.n_path, vol_0)
		spot_t = np.full(self.n_path, spot_0)

		milstein = (self.scheme == 1)
		for i in range(n_dt):
			vol_t = self.vol_step_euler(dt[i], vol_t, model, milstein)
			spot_t = self.spot_step_euler(dt[i], beta, vol_t, spot_t)

		return vol_t, spot_t

	def exact_mv_bsm(self, texp, vol_0, spot_0):

		return

	def cond_spot_sigma(self, texp, var_0):
		return NotImplementedError

	def return_var_realized(self, texp, cond):
		return None


Path = LambdaSABRhMcTimeDisc(sigma=0.2, theta=0.15)
Path.set_num_params()
vol21 = Path.vol_step_euler(0.25, 0.2, 'original', 1)
vol22 = Path.vol_step_euler(0.25, 0.2, 'exp', 1)
vol23 = Path.vol_step_euler(0.25, 0.2, 'exact', 1)

vol, spot = Path.cond_vol_step(1, 0.2, 100, 1, 'original')
vol1, spot1 = Path.cond_vol_step(1, 0.2, 100, 1, 'exp')
vol2, spot2 = Path.cond_vol_step(1, 0.2, 100, 1, 'exact')
vol_mean, vol_mean1, vol_mean2 = np.mean(vol), np.mean(vol1), np.mean(vol2)
spot_mean, spot_mean1, spot_mean2 = np.mean(spot), np.mean(spot1), np.mean(spot2)


# implied volatility
model_method = "original"
intr = 0.02
m_bsm = bsm.Bsm(sigma=0.2, intr=intr)
impvol = pd.DataFrame(0, index=[90, 95, 100, 105, 110], columns=[0.1, 0.25, 0.5, 1, 2])
for strike in [90, 95, 100, 105, 110]:
	for texp in [0.1, 0.25, 0.5, 1, 2]:
		if model_method == 'original':
			price = spot_mean * np.exp(-intr * texp) - strike
		elif model_method == 'exp':
			price = spot_mean1 * np.exp(-intr * texp) - strike
		else:
			price = spot_mean2 * np.exp(-intr * texp) - strike
		impvol.loc[strike, texp] = m_bsm.impvol(price, strike, 100, texp)


