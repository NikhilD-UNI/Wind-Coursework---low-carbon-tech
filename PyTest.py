# %%
import time
import numpy as np
import matplotlib.pyplot as plt
#from py_wake.utils.gradients import autograd
import V126_site_wt as RD
from py_wake.deficit_models import *
from py_wake.deficit_models.deficit_model import *
from py_wake.wind_farm_models import *
from py_wake.rotor_avg_models import *
from py_wake.superposition_models import *
from py_wake.deflection_models import *
from py_wake.turbulence_models import *
from py_wake.ground_models import *
from py_wake.deficit_models.utils import *

t = time.time()

site = RD.V126Site()
wt   = RD.V126()
x , y = [0] , [0]


wfm = PropagateDownwind(
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
sim_res = wfm(x,y)


plt.figure(figsize=(12,8))
sim_res.flow_map(wd=230).plot_wake_map()
print (wfm)
print ("Computation time (AEP + flowmap):", time.time()-t)
plt.title('AEP: %.2fGWh'%(sim_res.aep().sum()))

plt.figure()
aep = sim_res.aep()
aep.sum(['wt','ws']).plot()
plt.xlabel("Wind direction [deg]")
plt.ylabel("AEP [GWh]")
plt.title('AEP vs wind direction')

plt.figure()
aep.sum(['wt','wd']).plot()
plt.xlabel("Wind speed [m/s]")
plt.ylabel("AEP [GWh]")
plt.title('AEP vs wind speed')
