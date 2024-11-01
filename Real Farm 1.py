# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Colormap module
from matplotlib.colors import Normalize  # For normalizing AEP values

from py_wake.utils.gradients import autograd
from py_wake.deficit_models import *
from py_wake.deficit_models.deficit_model import *
from py_wake.wind_farm_models import *
from py_wake.rotor_avg_models import *
from py_wake.superposition_models import *
from py_wake.deflection_models import *
from py_wake.turbulence_models import *
from py_wake.ground_models import *
from py_wake.deficit_models.utils import *
from py_wake.flow_map import XYGrid
from py_wake.noise_models.iso import ISONoiseModel

from py_wake.site import UniformWeibullSite
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.examples.data.swt_dd_142_4100_noise.swt_dd_142_4100 import SWT_DD_142_4100

from topfarm import TopFarmProblem
from topfarm.plotting import XYPlotComp
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.utils import regular_generic_layout, regular_generic_layout_gradients
import V126_site_wt as RD

#specifying the site and wind turbines to use
site = RD.V126Site()
wt = RD.V126()
D = 126
windFarmModel = PropagateDownwind(
    site,
    wt,
    wake_deficitModel=NiayifarGaussianDeficit(
        a = [0.38, 0.004],
        rotorAvgModel=CGIRotorAvg(9),
        groundModel=None),
    superpositionModel=LinearSum(),
    deflectionModel=None,
    turbulenceModel=STF2017TurbulenceModel(),
    rotorAvgModel=CGIRotorAvg(9))

n_wt = 30
#boundary = [(-3644, 3780),(801,1995), (0, 0),(-4445, 1785)]
boundary= [
    (-3634, 3799),  
    (-2538, 3526), 
    (-1178, 3570),
    (-344, 2868),
    (565, 3110),
    (1252, 2978),
    (1172, 2012),
    (1023, 1318),
    (284, 1011),
    (0, 0),
    (-1084, 465),
    (-1308, 280),
    (-2161, 675),
    (-2899, 544),
    (-3522, 790),
    (-3451, 1132),
    (-3811, 1549),
    (-4449, 1783)
]
stagger = 5*D   #to create a staggered layout

# Function to generate turbine positions
def reg_func(sx, sy, rotation, **kwargs):
    x, y = regular_generic_layout(n_wt, sx, sy, stagger, rotation)
    return [x, y]

def reg_grad(sx, sy, rotation, **kwargs):
    dx_dsx, dy_dsx, dx_dsy, dy_dsy, dx_dr, dy_dr = regular_generic_layout_gradients(n_wt, sx, sy, stagger, rotation)
    return [[dx_dsx, dy_dsx], [dx_dsy, dy_dsy], [dx_dr, dy_dr]]

def aep_fun(x, y):
    aep = windFarmModel(x, y).aep().sum()
    return aep

daep = windFarmModel.aep_gradients(gradient_method=autograd, wrt_arg=['x', 'y'])

reg_grid_comp = CostModelComponent(input_keys=[('sx', 0),
                                               ('sy', 0),
                                               ('rotation', 5)],
                              n_wt=n_wt,
                              cost_function=reg_func,
                              cost_gradient_function = reg_grad,
                              output_keys= [('x', np.zeros(n_wt)), ('y', np.zeros(n_wt))],
                              objective=False,
                              use_constraint_violation=False,
                              )


aep_comp = CostModelComponent(input_keys=['x', 'y'],
                              n_wt=n_wt,
                              cost_function=aep_fun,
                              cost_gradient_function = daep,
                              output_keys= ("aep", 0),
                              output_unit="GWh",
                              maximize=True,
                              objective=True)

problem = TopFarmProblem(design_vars={'sx': (1*D, 10*D, 20*D),
                                      'sy': (1*D, 10*D, 20*D),
                                       'rotation': (100, -360, 360)
                                      },
                         constraints=[XYBoundaryConstraint(boundary,'polygon'),
                                      SpacingConstraint(3*D)],
                        grid_layout_comp=reg_grid_comp,
                        n_wt = n_wt,
                        cost_comp=aep_comp,
                        driver=EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=200),
                        plot_comp=XYPlotComp(),
                        expected_cost=0.01,
                        )

x, y = reg_func(3*D, 3*D, 50)
plt.figure()
plt.plot(x,y,'.')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Staggered wind farm layout with inter-turbine spacing of 3D')

cost, state, recorder = problem.optimize(disp=True)

problem.evaluate()

print(state)


sim_res = windFarmModel(recorder['x'][-1],recorder['y'][-1])
plt.figure(figsize=(12,8))
sim_res.flow_map(wd=225).plot_wake_map()

print(recorder['x'][-1])
print(recorder['y'][-1])

power=sim_res.Power.sel(wt=12)
wd=np.deg2rad(power.wd)
plt.figure(figsize=(12,8))
ax = plt.subplot(111, polar=True)
ax.plot(wd, power)# label=f'Turbine {12}')
ax.set_title(f'Power vs Wind Direction - Turbine {12 }')
ax.set_xlabel('Wind Direction (degrees)')
plt.legend()
plt.show()


# %%
aep_per_turbine = sim_res.aep().sum(["wd", "ws"]).values

# Step 6: Normalize AEP values for colormap
norm = Normalize(vmin=np.min(aep_per_turbine), vmax=np.max(aep_per_turbine))  # Normalize AEP range
cmap = cm.viridis  # Choose a colormap (viridis is perceptually uniform)

# Step 7: Plot the turbine positions and color them by AEP
plt.figure(figsize=(8, 6))
sc = plt.scatter(recorder['x'][-1], recorder['y'][-1], c=aep_per_turbine, cmap=cmap, s=100, label='Turbine', norm=norm)

# Add a colorbar to show the AEP scale
cbar = plt.colorbar(sc)
cbar.set_label('AEP [GWh/year]', rotation=270, labelpad=15)

for i, (x_pos, y_pos) in enumerate(zip(recorder['x'][-1], recorder['y'][-1])):
    plt.text(x_pos, y_pos + 90, f'{i}', fontsize=10, ha='center')


# Customize the plot
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Turbine Positions and AEP (colored by AEP)')
plt.grid(True)
plt.legend()
plt.show()


# %%
