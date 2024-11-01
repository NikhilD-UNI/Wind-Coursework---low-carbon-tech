# %%
import numpy as np
import matplotlib.pyplot as plt

from py_wake.examples.data.lillgrund import LillgrundSite, power_curve, ct_curve
from py_wake.wind_turbines._wind_turbines import WindTurbines, WindTurbine
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from py_wake.deficit_models.gaussian import BastankhahGaussian
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
from py_wake.utils.gradients import autograd
from topfarm import TopFarmProblem
from py_wake import XZGrid

# Initial inputs
n_wt = 3

n_wd = 1
wd = [180]

lb = 82                         # lower boundary constraint
ub = 300                       # upper boundary constraint

hh = 80                         # starting hub height condition
hg = 120                        # second starting hub height condition
h = np.ones([n_wt]) * hh          # hub height array
h_max = np.ones([n_wt]) * ub      # baseline hub height

for i in range(n_wt):
    h[i] = hh
    # if i % 2 == 0:
    #     h[i] = hh
    # else:
    #     h[i] = hg

print('wind farm hub heights:',h)

power = 2300
diameter = np.ones([n_wt]) * 93 # diameter [m]

# Site specification
site = LillgrundSite()

x = np.linspace(0, 93 * 4 * n_wt, n_wt)

y = [0] * n_wt

nom_power_array = power * np.ones([n_wt]) # rated power array

class SWT23(WindTurbine):   # Siemens 2.3 MW
    def __init__(self, method='linear'):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """
        WindTurbine.__init__(self, name='SWT23', diameter=93, hub_height=80,
                             powerCtFunction=PowerCtTabular(power_curve[:, 0], power_curve[:, 1], 'kw',
                                                            ct_curve[:, 1], method=method))

wind_turbines = WindTurbines(
                names=['SWT23' for i in range(len(x))],
                diameters = diameter,
                hub_heights = h,
                powerCtFunctions=[GenericWindTurbine(name='SWT23',
                                                        diameter = diameter[i],
                                                        hub_height = h[i],
                                                        power_norm = nom_power_array[i]).powerCtFunction for i in range(len(x))])

wf_model = BastankhahGaussian(site, wind_turbines)

# AEP Calculation

class PyWakeAEPCostModelComponent(AEPCostModelComponent):
    def __init__(self, windFarmModel, n_wt, wd=None, ws=None, max_eval=None, grad_method=autograd, n_cpu=1, **kwargs):
        self.windFarmModel = windFarmModel

        #objective function
        def get_aep_func(h):

            h_new = h[:n_wt]
            simres = windFarmModel(x, y, h=h_new)
            aep = simres.aep().sum()

            return aep

        #specifying the gradients
        def daep_h(h):
            return windFarmModel.aep_gradients(autograd, wrt_arg=['h'])(x, y, h)

        AEPCostModelComponent.__init__(self,
                                       input_keys=['h'],
                                       n_wt=n_wt,
                                       cost_function=get_aep_func,
                                       cost_gradient_function=daep_h,
                                       output_unit='GWh',
                                       max_eval=max_eval, **kwargs)

cost_comp = PyWakeAEPCostModelComponent(windFarmModel=wf_model, n_wt=len(x), grad_method=autograd, n_cpu=1, wd=None, ws=None)

maxiter = 20
tol = 1e-6

problem = TopFarmProblem(design_vars= {'h':(h, lb, ub)},
                        cost_comp=cost_comp,
                        driver=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=maxiter, tol=tol),
                        n_wt=n_wt,
                        expected_cost=0.001
                        )

_,state,_=problem.optimize()

h = np.around(state['h'])
print('final hub heights:',h)

#taking only the first row of turbines
x = x[:6]
y = y[:6]
h = h[:6]

# %%
sim_res_ref = wf_model(x, y, wd=[270])
sim_res_opt = wf_model(x, y, h=h, wd=[270])
plt.figure(figsize=(12,4))
sim_res_opt.flow_map(XZGrid(y=0)).plot_wake_map()
plt.ylabel('Height [m]')
plt.xlabel('x [m]')
# %%
