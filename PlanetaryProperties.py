import numpy as np
import pandas as pd
from scipy import optimize
import radvel
from radvel.plot import orbit_plots, mcmc_plots


class KeplerianAnalysis:
    """
    Radvel MCMC analysis of radial velocity data. Adapted from the Fulton et al. (2018) tutorial in the Radvel docs:
    https://radvel.readthedocs.io/en/latest/tutorials/K2-24_Fitting+MCMC.html
    """

    def __init__(self, df, lsdf, gamma, jitter, stellar_mass, starname, directory, period_col='Period',
                 xcol='Julian Date', ycol='Radial Velocity (m/s)', errcol='Error (m/s)'):
        self.df = df
        self.t = df[xcol]
        self.y = df[ycol]
        self.err = df[errcol]
        self.lsdf = lsdf
        self.post = 0
        self.mod = 0
        self.res = 0
        self.like = 0
        self.gamma = gamma
        self.jit = jitter
        self.periods = lsdf[period_col]
        self.smass = stellar_mass
        self.name = starname
        self.dir = directory
        self.paramsdf = 0

    def initialize_model(self, periods, numplanets):
        """
        Initialize the RadVel keplerian analysis model as described in the tutorial for each of the n number of planets
        specified Fulton et al. (2018).
        :param periods: The periods specified for each planet.
        :param numplanets: The number of planets in the system.
        :return: mod, the initialized radvel model.
        """
        time_base = self.t.values[0]
        params = radvel.Parameters(numplanets, basis='per tc secosw sesinw logk')

        for i in range(1, numplanets + 1):
            params['per' + str(i)] = radvel.Parameter(value=periods[i - 1])
            params['tc' + str(i)] = radvel.Parameter(value=self.t[0])
            params['secosw' + str(i)] = radvel.Parameter(value=0.01)
            params['sesinw' + str(i)] = radvel.Parameter(value=0.01)
            params['logk' + str(i)] = radvel.Parameter(value=1.1)

        mod = radvel.RVModel(params, time_base=time_base)
        mod.params['dvdt'] = radvel.Parameter(value=-0.02)
        mod.params['curv'] = radvel.Parameter(value=0.01)
        print(mod)
        return mod

    def keplerian_analysis(self, periods, numplanets):
        """
        Analyze the given parameters from initializing the model and update the posteriors, as in the tutorial
        described by Fulton et al. (2018).
        :param periods: the periods in the system.
        :param numplanets: The number of planets in the system.
        """

        # Initialize model
        mod = KeplerianAnalysis(self.df, self.lsdf, self.gamma, self.jit, stellar_mass=self.smass,
                                starname=self.name, directory=self.dir).initialize_model(periods, numplanets)
        like = radvel.likelihood.RVLikelihood(mod, self.t, self.y, self.err)
        like.params['gamma'] = radvel.Parameter(value=self.gamma, vary=False, linear=True)  # Measurement uncertainty
        like.params['jit'] = radvel.Parameter(value=self.jit)  # Stellar Activity noise

        for i in range(1, numplanets + 1):
            like.params['per' + str(i)].vary = True
            like.params['tc' + str(i)].vary = True
            like.params['secosw' + str(i)].vary = False
            like.params['sesinw' + str(i)].vary = False

        print(like)

        # Update posteriors
        post = radvel.posterior.Posterior(like)
        res = optimize.minimize(post.neglogprob_array, post.get_vary_params(), method='Powell')

        # Update the class's properties to match.
        self.like = like
        self.mod = mod
        self.res = res
        self.post = post
        print(post)
        print('\n')

    # noinspection PyTypeChecker
    def approximate_planetary_params(self, nrun=500, nwalkers=20):
        """
        Approximate the planetary mass and semi-major axis from the estimated planetary parameters implemented in
        keplerian_analysis. The mass, semi-major axis, and period are stored in a csv file labeled
        _PlanetaryProperties in the directory specified, and the rest of the orbital parameters are stored in a
        seperate csv labeled _MCMCAnalysisPlanetaryParameters.csv in the directory specified.
        :param nrun: the MCMC number of iterations
        :param nwalkers: the number of MCMC walkers
        """
        df = radvel.mcmc(self.post, nwalkers=100, nrun=nrun, savename='rawchains.h5')  # Start MCMC analysis

        # Some cool plots
        Corner = mcmc_plots.CornerPlot(self.post, df, saveplot=self.dir + "TestCornerPlot")
        Corner.plot()  # A corner plot
        trendplot = radvel.plot.mcmc_plots.TrendPlot(self.post, df, nwalkers, outfile=self.dir + 'TestTrendFile')
        trendplot.plot()  # A trend plot
        multipanel = radvel.plot.orbit_plots.MultipanelPlot(self.post, saveplot=self.dir + 'TestMultipanel')
        multipanel.plot_multipanel()  # A multipanel plot with the estimated parameters

        # Make the planetary properties pandas DataFrame.
        mass = []
        a = []

        for per in list(self.periods):
            mass_ = radvel.utils.Msini(np.average(self.y.values), per, self.smass, e=0.0)
            mass.append(mass_)
            a_ = radvel.utils.semi_major_axis(per, self.smass)
            a.append(a_)

        params_df = pd.DataFrame({"Plantary Period (JD)": list(self.periods),
                                  "Semi Major Axis (AU)": list(a), "Planetary Mass (Earth masses)": list(mass)})

        print("Saving the computed planetary properties at the directory " + self.dir + "...")
        params_df.to_csv(self.dir + self.name + '_PlanetaryProperties.csv')
        print("Done!")
        self.paramsdf = params_df
        print("Saving the computed orbital parameters at the directory " + self.dir + "...")
        df.to_csv(self.dir + self.name + '_MCMCAnalysisPlanetaryParameters.csv')
        print("Done!")
