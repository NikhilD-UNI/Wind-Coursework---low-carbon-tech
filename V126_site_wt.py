# %%
import numpy as np
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
import py_wake.site
from py_wake.site.shear import PowerShear


wt_x = [0]
wt_y = [0]


#Cp not used???
Cp=np.array([[3, 0.17],
 [4, 0.376],
 [5, 0.423],
 [6, 0.439],
 [7, 0.447],
 [8, 0.446],
 [9, 0.445],
 [10, 0.417],
 [11, 0.338],
 [12, 0.261],
 [13, 0.206],
 [14, 0.165],
 [15, 0.134],
 [16, 0.11],
 [17, 0.092],
 [18, 0.077],
 [19, 0.066],
 [20, 0.056],
 [21, 0.049],
 [22, 0.042]])
power_curve = np.array([[3, 35],
 [4, 184],
 [5, 404],
 [6, 725],
 [7, 1172],
 [8, 1760],
 [9, 2482],
 [10, 3187],
 [11, 3433],
 [12, 3450],
 [13, 3450],
 [14, 3450],
 [15, 3450],
 [16, 3450],
 [17, 3450],
 [18, 3450],
 [19, 3450],
 [20, 3450],
 [21, 3450],
 [22, 3450]]) * [1, 1000]
ct_curve = np.array([[3, 0.892],
 [4, 0.821],
 [5, 0.831],
 [6, 0.818],
 [7, 0.805],
 [8, 0.778],
 [9, 0.751],
 [10, 0.665],
 [11, 0.489],
 [12, 0.356],
 [13, 0.271],
 [14, 0.215],
 [15, 0.173],
 [16, 0.143],
 [17, 0.12],
 [18, 0.102],
 [19, 0.087],
 [20, 0.076],
 [21, 0.066],
 [22, 0.059]])


class V126(WindTurbine):
    def __init__(self, method='pchip'):
        """
        Parameters
        ----------
        method : {'linear', 'pchip'}
            linear(fast) or pchip(smooth and gradient friendly) interpolation
        """
        WindTurbine.__init__(self, name='V126', diameter=126, hub_height=117,
                             powerCtFunction=PowerCtTabular(power_curve[:, 0], power_curve[:, 1], 'w',
                                                            ct_curve[:, 1], method=method, ws_cutin = 3, ws_cutout=22.5,power_idle=0))


VestasV126 = V126


class V126Site(UniformWeibullSite):
    def __init__(self, ti=.1, shear=None):
        f = [0.018227305218012867, 0.017274243507267095, 0.017512508934953538, 0.014415058375029783, 0.012866333095067906, 0.016916845365737433, 0.02740052418394091, 0.023707410054801047, 0.02573266619013581, 0.01584465094114844, 0.01703597807958065, 0.014534191088873005, 0.018942101501072194, 0.015248987371932332, 0.01834643793185609, 0.01679771265189421, 0.022754348344055278, 0.02037169406719085, 0.03252323087919943, 0.03621634500833929, 0.0696926375982845, 0.06337860376459375, 0.08827734095782702, 0.062187276626161546, 0.06028115320467, 0.03002144388849178, 0.030497974743864665, 0.021682153919466287, 0.028710984036216344, 0.02203955206099595, 0.0284727186085299, 0.021443888491779844, 0.024064808196330712, 0.017869907076483203, 0.01584465094114844, 0.012866333095067906]
        a = [6.287784093273985, 6.104404448948996, 5.409577068525783, 4.987697772959289, 5.0061177068196, 6.017070182901096, 6.02476338250131, 6.044691380240936, 5.839550869161312, 5.251027256467249, 5.081767866706889, 4.699584948032882, 4.772993804213942, 4.982581131453671, 5.361631609605629, 6.254951045076497, 6.157277297214688, 6.431410303581807, 6.625597040951533, 6.859473561765149, 7.183626535982341, 7.165766969332669, 7.1358034388975184, 6.663563203013223, 6.617519504336819, 6.2311954789199335, 6.0357036929151535, 5.536025291906218, 5.7496939137348, 5.497924605829026, 5.339894723912773, 5.8526781613221655, 6.537799595071444, 6.6492669029146105, 5.781404925497159, 5.321843068753143]
        k = [1.6791181993882405, 2.3762046911674664, 2.210926978083153, 2.600217000843248, 2.6472221206023705, 2.8524371622555362, 2.6186691491038503, 2.933512177758625, 2.953777328323653, 2.5391913339681764, 3.3458607849509057, 3.1191772328967864, 2.7073584921610787, 2.3994319848350063, 2.037400067915097, 1.6244784592221628, 1.6522580280616972, 1.719574128491506, 1.6313769928599229, 1.868401744342781, 2.227927352106376, 2.6394332950653223, 2.5766615508150363, 2.6054529952315013, 2.548041439590534, 2.562073545032758, 2.1353963099121147, 2.056894235342403, 1.931554525787624, 2.200985916984234, 2.5382229888583216, 2.276443357724797, 2.2749892590648453, 2.156964150401733, 2.2724609614940694, 2.0163287031438717]
        ti =0.1#[0.6123288311910433, 0.44781479110575223,	0.47770163971581003,	0.41311340506627936,	0.40654077823700874,	0.3802501691139403,	0.4105066543160383,	0.3708210326708478,	0.3685402706785967,	0.421989976506703,	0.3295407319484085,	0.35096805326060015, 0.3984471115877138,	0.4439305555352027,	0.5140825512142188,	0.6310849039023566,	0.62139355325244,	0.5992101502679694,	0.6286472804134772,	0.5558050348816743,	0.4744315493302034,	0.4076145570307972,	0.41649246349780916,	0.4123701591546691,	0.42067768750076834,	0.41861464386867747,	0.4928353718149253,	0.5096997546234477,	0.5393794125188672,	0.4796363322503179,	0.42213408910277206,	0.4653572790979873,	0.46562385238145904,	0.4884097983072847,	0.46608813197932, 0.5189124214949387]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=PowerShear(h_ref=10, alpha=.16))
        self.initial_position = np.array([wt_x, wt_y]).T


def main():
    wt = V126()
    print('Diameter', wt.diameter())
    print('Hub height', wt.hub_height())

    import matplotlib.pyplot as plt
    ws = np.linspace(3, 20, 100)
    plt.plot(ws, wt.power(ws) * 1e-3, label='Power')
    c = plt.plot([], [], label='Ct')[0].get_color()
    plt.ylabel('Power [kW]')
    ax = plt.gca().twinx()
    ax.plot(ws, wt.ct(ws), color=c)
    ax.set_ylabel('Ct')
    plt.xlabel('Wind speed [m/s]')
    plt.gcf().axes[0].legend(loc=1)
    plt.show()

    site = V126Site()
    site.plot_wd_distribution(n_wd=36, ws_bins=[0,5,10,15,20,25])


if __name__ == '__main__':
    main()
# %%