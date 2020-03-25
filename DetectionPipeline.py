import AsciiReader as Reader

import numpy as np

import scipy.stats
from scipy.optimize import curve_fit
from scipy import optimize
import statsmodels.api as sm

from astropy.timeseries import LombScargle
import pandas as pd

import matplotlib

import radvel
from radvel.plot import orbit_plots, mcmc_plots

import os
import random

import time

matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['font.family'] = 'serif'


def check_p(p, alpha=0.01):
    """
    Checks whether a given p-value is significant given the maximum significance (alpha).
    :param p: a float, the p-value
    :param alpha: a float, the maximum significance (eg. most statistical tests use 0.01 or 0.05 for their alpha).
    Default is 0.01.
    :return:
    """
    if p < alpha:
        return True
    elif p > alpha:
        return False


def delete_seasonality(xydata, per, xcol='Julian Date', ycol='Radial Velocity (m/s)'):
    def func(x_, amp, phase):
        return amp * np.sin(np.add(np.multiply(per, x_), phase))

    x = xydata[xcol].to_numpy()
    y = xydata[ycol].to_numpy()

    params, pcov = curve_fit(func, x, y)
    diff = list()
    y_ = func(x, *params)

    for i in range(y_.size):
        diff.append(y[i] - y_[i])
    diff = np.array(diff)

    xydata_copy = xydata.copy()
    xydata_copy[ycol] = diff

    return xydata_copy


def approx_equal(x1, x2, neg_err, pos_err):
    """
    Checks whether two values are "approximately equal" to each other. That is, if the values are within a (specified)
    range of each other, the values are said to be approximately equal.
    :param x1: a float or int, the first value
    :param x2: a float or int, the second value
    :param neg_err: a float or int, the negative error (the portion of the error range less than the value)
    :param pos_err: a float or int, the positive error (the portion of the error range greater than the value)
    :return: a boolean, whether the two values are approximately equal to each other (True if they are approximately
    equal, and False if they are not).
    """
    small_x1 = x1 - neg_err
    big_x1 = x1 + pos_err
    small_x2 = x2 - neg_err
    big_x2 = x2 + pos_err
    if small_x1 < x2 < big_x1 or small_x2 < x1 < big_x2:
        return True
    else:
        return False


class CheckVariability:
    """
    Checks initial radial velocity data for unexplained variability and trends.
    """

    def __init__(self, filepath, cols=Reader.cols):
        self.fp = filepath
        self.cols = cols
        self.df = Reader.read(file_path=self.fp, column_names=self.cols)
        self.alpha = 0.01
        self.pchi_constant = 0
        self.weighted_mean = 0

    def f_test_is_variable(self, col1='Radial Velocity (m/s)', col2='Error (m/s)'):
        """
        Checks whether a stellar sample has any unexplained variability (whether it's worth looking into for further
        investigations), using a one-tailed f-test (Zechmeister et al. 2009). The default format for col1 and col2 is
        the HiRES publicly available radial velocity data (Butler, Vogt, Laughlin et al. 2017).

        :param col1: The name of the pandas DataFrame column that contains the radial velocity data.
        :param col2: The name of the pandas DataFrame column that contains the measurement error data.
        :return: boolean true or false (true if the star has unexplained variability and should be investigated further
        and false if the star's variability can be explained with instrumentation and measurement error).
        """

        # Degrees of freedom.
        df1 = self.df[col1].size - 1
        df2 = self.df[col2].size - 1

        radvel_var = np.var(self.df[col1])  # Variance of radial velocity data.
        error_var = np.var(self.df[col2])  # Variance of measurement error data.
        f = radvel_var / error_var
        p = 1 - scipy.stats.f.cdf(f, df1, df2)

        return check_p(p, self.alpha)

    def chisquared_is_variable(self, col1='Radial Velocity (m/s)', col2='Error (m/s)'):
        """
        Checks whether a stellar sample has any unexplained variability (whether it's worth looking into for further
        investigations), using a chi-squared test (Zechmeister et al. 2009). The default format for col1 and col2 is
        the HiRES publicly available radial velocity data (Butler, Vogt, Laughlin et al. 2017).

        Note: This function requires that the sample size be at least 6 data points. If there are less than 6
        data points in the sample, the function will return "0".

        :param col1: The name of the pandas DataFrame column that contains the radial velocity data.
        :param col2: The name of the pandas DataFrame column that contains the measurement error data.
        :return: boolean true or false (true if the star has unexplained variability and should be investigated further
        and false if the star's variability can be explained with instrumentation and measurement error).
        """

        rad_vels = self.df[col1].to_numpy()
        err = self.df[col2].to_numpy()

        if rad_vels.size <= 5:
            return 0

        else:
            weighted_mean = np.sum(np.multiply(err, rad_vels)) / np.sum(err)  # Compute RV weighted mean
            self.weighted_mean = weighted_mean
            p = scipy.stats.chisquare(rad_vels, weighted_mean)[1]  # Compute p-value
            self.pchi_constant = p
            return check_p(p, self.alpha)

    def is_variable_long_term(self, date_col="Julian Date", radvel_col="Radial Velocity (m/s)", err_col="Error (m/s)"):
        """
        TODO:: Get this program double-checked with Zechmeister et al. 2009 AND Dr. Haywood

        Checks whether the radial velocity data for a star contains a long-term trend, as described by Zechmeister et.
        al (2009). The default format for xcol, ycol, and err_col is based off of the HiRES publicly available radial
        velocity data (Butler, Vogt, Laughlin et al. 2017).

        Note:: If the dataset contains less than 6 datapoints, the function returns "0", as it is too small of a sample
        size to run the statistical tests used.

        :param date_col: string, the name of the column in the pandas DataFrame that contains the Julian Date.
        :param radvel_col: string,the name of the column in the pandas DataFrame that contains the radial velocity data.
        :param err_col: string, the name of the column in the pandas DataFrame that contains the measurement errors.
        :return: boolean true or false (true if the star has significant long-term variability
        and false if it does not).
        """

        y = self.df[radvel_col].to_numpy()
        X = self.df[date_col].to_numpy()
        w = self.df[err_col]

        if y.size <= 5:

            return 0

        else:

            # Fit a linear line of best fit (weighted least squares).
            mod_wls = sm.WLS(y, X, weights=w)
            res_wls = mod_wls.fit()
            print(res_wls.summary())
            m = res_wls.params[0]

            # Calculate chi-squared statistic for line of best fit and constant model.
            # Stack overflow ref: https://stackoverflow.com/questions/35730534/numpy-generate-data-from-linear-function
            x = np.arange(y.size)  # Generate data using the linear function (Garret R, Stack overflow)
            delta = np.random.uniform(-1 * np.amax(w), np.amax(w), size=y.size)
            y_ = np.add(m * x, delta)

            pslope = scipy.stats.chisquare(y_, self.weighted_mean)[1]
            pconstant = scipy.stats.chisquare(y, self.weighted_mean)[1]

            # Calculate F-statistic for p values of slope and constant models (Zechmeister et al. 2009)
            fslope = (y.size - 2) * ((pconstant - pslope) / pslope)

            # p-value from F-statistic
            p = 1 - scipy.stats.f.cdf(fslope, y.size, y.size)

            check = check_p(p, self.alpha)

            y_diff = []
            if check:
                for i in range(1, y.size):
                    y_diff.append(y[i] - y[i - 1])

            return check


class PeriodogramAnalysis:
    """
    A class to perform periodogram analyses on radial velocities as done by Zechmeister et al. (2009). The inputs are
    pandas DataFrame where the radial velocity data is stored, as well as the column names corresponding to the
    radial velocity data.
    """

    def __init__(self, data_frame, save_dir='', cols=Reader.cols):
        self.cols = cols
        self.df = data_frame
        self.lsdf = pd.DataFrame()
        self.noise_df = pd.DataFrame()
        self.dir = save_dir

    def gls(self, col_x='Julian Date', col_y='Radial Velocity (m/s)', err_col='Error (m/s)'):
        """
        Computes a generalized lomb-scargle periodogram on the object time series, as
        described by Zechmeister et al. (2009). The default format for col1 and col2 is
        the HiRES publicly available radial velocity data (Butler, Vogt, Laughlin et al. 2017).

        :param col_x: a string, the name of the column containing the x-values (time stamps).
        :param col_y: a string, the name of the column containing the y-values (radial velocities).
        :param err_col: a string, the name of the columm containing the radial velocity errors.
        :return: the period, power, and false alarm probability for each signal detected with the periodogram.
        Moreover, the lsdf pandas DataFrame is updated with the contents of the period, power, and FAP arrays.
        """

        # Compute Generalized Lomb-Scargle periodogram (Zechmeister et al. 2009)
        ls = LombScargle(self.df[col_x], self.df[col_y], self.df[err_col],
                         fit_mean=True)

        frequency, power = ls.autopower(method="slow")

        # Compute false alarm probability for each signal.
        fap = []
        print("Starting Bootstrap FAP extrapolation process...")
        for pow_ in power:
            # append(ls.false_alarm_probability(pow_, method='bootstrap'))
            fap.append(ls.false_alarm_probability(pow_))
        print("Finished Bootstrap FAP extrapolation process.")
        fap = np.array(fap)
        period_ = 1 / frequency
        df = pd.DataFrame({'Period': period_, 'Power': power, 'FAP': fap})
        self.lsdf = df

        print("Saving the Period, Power, and FAP dataset to a csv at the directory " + self.dir + "...")
        self.lsdf.to_csv(self.dir + '_LombScarglePeriodogramResults.csv')
        print("Done!")
        self.lsdf = self.lsdf.reset_index(drop=True)
        return period_, power, fap

    def fap_is_variable(self):
        """
        Checks if the FAP column of the lsdf pandas DataFrame indicates the presence of any statistically
        significant signals. Updates the lsdf panads DataFrame to pass the FAP analysis as described by
        Zechmeister et al. (2009).
        """
        for i in range(0, self.lsdf.shape[0]):
            if self.lsdf.loc[i, :]['FAP'] >= 0.01:
                self.lsdf = self.lsdf.drop(i)

    def generate_noise_df(self, col_x='Julian Date', col_y='Radial Velocity (m/s)'):
        """
        Subtracts significant variability signals from data to generate a pandas DataFrame with residuals only. Assumes
        the time series data has additive signals. Uses the lsdf (Lomb-Scargle DataFrame), cleaned for
        unexplained variability and the inputted raw radial velocity data. The residuals are stored in the
        pandas DataFrame noise_df.
        :param col_x: a string, the name of the column containing the x-values (time stamps).
        :param col_y: a string, the name of the column containing the y-values (radial velocities).
        :return: the residuals dataframe (stored as noise_df in the class).
        """
        resdf = self.df.copy()
        for per in self.lsdf['Period'].values:
            resdf.loc[:, [col_x, col_y]] = delete_seasonality(resdf.loc[:, [col_x, col_y]], per)

        self.noise_df = resdf
        print("Saving the Radial Velocity Residuals dataset to a csv at the directory " + self.dir + "...")
        self.noise_df.to_csv(self.dir + '_ResidualsData.csv')
        print("Done!")

        return resdf


class CheckStellarActivity:
    def __init__(self, radvel_df, lsdf, resdf, save=False, save_dir='', period_col='Period',
                 halpha_col='H Alpha'):
        self.df = radvel_df
        self.lsdf = lsdf
        self.resdf = resdf
        self.cols = Reader.cols
        self.jitter = 0
        self.gamma = 0
        self.halpha_col = halpha_col
        self.dir = save_dir
        self.save = save
        self.period = self.lsdf[period_col]

    def check_sa(self, sa_col, err_col='Error (m/s)', n=7):
        """
        Searches the stellar activity data for whether the stellar activity indicator (specified by sa_col)
        indicates that any of the detected statistically significant signals (or their harmonics,
        where the harmonics are factors and fractions of the initial period n = 1-7) are effects of stellar activity.
        This function deletxes any signals that are likely to be stellar activity.
        :param sa_col: a string, the name of the column that contains the stellar activity data.
        :param err_col: a string, the name of the column that contains the error data.
        :param n: an int, the highest harmonic to check for stellar activity (default is 7).
        :return:
        """

        # Check if the stellar activity data was available for the dataset (according to HiRES public data.
        # If all H-alpha signals are -1, H-alpha was not available).
        self.lsdf = self.lsdf.reset_index(drop=True)
        print(self.lsdf)
        df_ = self.lsdf.copy()
        _df_ = self.df[["Julian Date", sa_col, err_col]]
        if self.df[sa_col].values.all() == -1:
            return None

        else:
            sa_per = PeriodogramAnalysis(_df_, save_dir=self.dir + 'sa')  # Analyze stellar activity with a periodogram
            print("\nCalculating Stellar Activity Generalized Lomb Scargle Periodogram:")
            sa_per.gls(col_y=sa_col)
            sa_per.fap_is_variable()

            hlsdf = sa_per.lsdf  # the new Lomb Scargle pandas DataFrame
            avg_err = np.max(self.df[err_col].values)
            for saper in hlsdf["Period"].values:  # Check the stellar activity pandas DataFrame for harmonics
                saper_ = [saper]
                for i in range(1, n):
                    for k in range(1, n):
                        saper_.append((k / i) * saper)
                        saper_.append((i / k) * saper)
                for count, rvper in enumerate(self.lsdf["Period"].values):
                    for sap in saper_:
                        if approx_equal(sap, rvper, avg_err, avg_err):
                            self.lsdf = self.lsdf.reset_index(drop=True)
                            df_ = self.lsdf.drop(count, axis=0)  # Delete any harmonics found
            # Initialize jitter and gamma parameters for keplerian analysis
            self.jitter = np.average(self.df[self.halpha_col].values)
            self.gamma = avg_err
            self.lsdf = df_

            # Save the stellar activity DataFrame and the updated inputted DataFrame to a csv
            if self.save:
                print("Saving the updated Period, Power, and FAP dataset to a csv at the directory " + self.dir + "...")
                self.lsdf.to_csv(self.dir + '_UpdatedLombScarglePeriodogramResults.csv')
                print("Done!")

                print("Saving the Stellar Activity Period, Power, and FAP dataset to "
                      "a csv at the directory " + self.dir + "...")
                hlsdf.to_csv(self.dir + '_SALombScarglePeriodogramResults.csv')
                print("Done!")
            self.lsdf = self.lsdf.reset_index(drop=True)
            return self.lsdf

    def check_integer_harmonics(self, err=0.05):
        """
        Check confirmed periods for any whole-number and 1/n harmonics.
        If the any harmonics are detected, the period harmonic with thehighest FAP is kept.
        :param err: a float, the plus or minus error of margin when checking for harmonics.
        :return: the updated lsdf pandas DataFrame
        """

        print("Checking for Integer Multiples of Planetary Period Harmonics...")
        self.lsdf = self.lsdf.reset_index(drop=True)
        temp_df = self.lsdf.copy()  # Temporary pandas DataFrame to delete harmonics from

        indices = []  # The indices at which any harmonic is located
        for period, num in enumerate(temp_df["Period"], start=0):  # A long for loop for a good cause
            # Updates the indices with the index at which a harmonic is located
            for x in temp_df["Period"]:
                if x != period and x != 0 and period != 0:
                    if approx_equal((x / period) % 1, 0, err, err) \
                            or approx_equal((x / period) % 1, 0, err, err) \
                            or approx_equal((period / x) % 1, 1, err, err) \
                            or approx_equal((x / period) % 1, 1, err, err):
                        is_harmonic = True

                        # Remove any harmonics found and choose the harmonic with the highest FAP (Bonfils et al (2013)
                        if is_harmonic:
                            if temp_df["Period"].values.size == 1 or \
                                    temp_df["Period"].values.size == 0:
                                break
                            else:
                                new_np = temp_df.values
                                indices = list(set(indices))
                                new_np = new_np[indices]
                                index = np.argmax(np.max(new_np[:, 2]))
                                new_np = new_np[new_np[:, 2] != new_np[index, 2]]

                                for fap in new_np[:, 2]:
                                    rounded = [round(x, 5) for x in pd.DataFrame(temp_df).values[:, 2]]
                                    fap = round(fap, 5)
                                    temp_df = temp_df[rounded != fap]
                                    self.lsdf = self.lsdf.reset_index(drop=True)

                                indices = []

                                if temp_df["Period"].values.size == 1:
                                    break

        self.lsdf = temp_df
        print("Done!")
        return self.lsdf

    def check_harmonics(self, n=15, err=0.05):
        """
        Check confirmed periods for any harmonics. If the any harmonics are detected, the period harmonic with the
        highest FAP is kept.
        :param n: an integer, the number of harmonics checked (1/n and n for n=1 to the specified number)
        :param err: a float, the plus or minus error of margin when checking for harmonics.
        :return: the updated lsdf pandas DataFrame
        """

        print("Checking for (General) Planetary Period Harmonics, n = ", n, "...")
        self.lsdf = self.lsdf.reset_index(drop=True)
        temp_df = self.lsdf.copy()  # Temporary pandas DataFrame to delete harmonics from

        indices = []  # The indices at which any harmonic is located
        for x_, per in enumerate(temp_df["Period"], start=0):  # A long for loop for a good cause
            # Updates the indices with the index at which a harmonic is located
            is_harmonic = False
            while not is_harmonic:
                for x in range(1, n):
                    for y in range(2, n + 1):
                        frac1 = x / y
                        frac2 = y / x
                        per1 = per / temp_df["Period"].values
                        per2 = temp_df["Period"].values / per
                        per1 = per1[per1 != 1]
                        per2 = per2[per2 != 1]

                        # Check to see if the period is close to any fractional/integer harmonic
                        for i, per_ in enumerate(per1, start=0):
                            if approx_equal(frac1, per_, err, err):
                                is_harmonic = True
                                indices.append(i)
                                break
                            elif approx_equal(frac2, per_, err, err):
                                is_harmonic = True
                                indices.append(i)
                                break
                        for i, per_ in enumerate(per2, start=0):
                            if approx_equal(frac1, per_, err, err):
                                is_harmonic = True
                                indices.append(i)
                                break
                            elif approx_equal(frac2, per_, err, err):
                                is_harmonic = True
                                indices.append(i)
                                break
                break

            # Remove any harmonics found and choose the harmonic with the highest FAP (Bonfils et al (2013)
            if is_harmonic:
                new_np = temp_df.values
                indices = list(set(indices))
                new_np = new_np[indices]
                index = np.argmax(np.max(new_np[:, 2]))
                new_np = new_np[new_np[:, 2] != new_np[index, 2]]

                for fap in new_np[:, 2]:
                    rounded = [round(x, 5) for x in pd.DataFrame(temp_df).values[:, 2]]
                    fap = round(fap, 5)
                    temp_df = temp_df[rounded != fap]
                    self.lsdf = self.lsdf.reset_index(drop=True)

                indices = []

            if temp_df["Period"].values.size == 1:
                break

        self.lsdf = temp_df
        print("Done!")
        return self.lsdf


# noinspection PyTypeChecker
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

    def approximate_planetary_params(self, nrun=500, nwalkers=30):
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


def detect(filepath, savedir="", stellar_mass=1):
    """
    An all-in-one function for implementing the Planetary Detection Pipeline as described by Zechmeister et al (2009)
    and Bonfils et al (2013)
    :param filepath: the filepath at which the radial velocity data (ascii format) is located.
    implemented by Fulton et al. (2018).
    :param savedir: the directory at which the output files will be saved. Default will be the directory in which the
    python file is being executed.
    :param stellar_mass: the masses of the host star, in solar radii. Default is 1 solar radius.
    :return: the final period, power, fap pandas DataFrame and the keplerian mass, period, semi-major axis  pandas
    DataFrame
    """

    # lsdf = pd.DataFrame()
    # params = pd.DataFrame()

    print("File Being Searched:: ", filepath)
    tock = time.time()
    _df_ = Reader.read(filepath)

    period = PeriodogramAnalysis(_df_, save_dir=savedir)
    period.gls()
    period.fap_is_variable()
    resdf = period.generate_noise_df()
    sa = CheckStellarActivity(_df_, period.lsdf, resdf, save_dir=savedir)
    sa.check_sa('H Alpha')
    sa.check_sa('S Value')
    # print(sa.check_integer_harmonics())

    if sa.lsdf.size == 0:
        return "No Presence of Exoplanet Detected"

    if sa.lsdf.size >= 30:
        return "Too many planetary signals detected (", resdf.size, " were detected by the algorithm. Aborting..."

    else:
        try:

            print(sa.check_harmonics())

            jit = sa.jitter
            gamma = sa.gamma
            sa.lsfd = sa.lsdf.reset_index(drop=True)
            kep = KeplerianAnalysis(df=_df_, gamma=gamma, jitter=jit, lsdf=sa.lsdf,
                                    stellar_mass=stellar_mass, starname="test", directory=savedir)
            kep.keplerian_analysis(sa.lsdf['Period'].values, numplanets=sa.lsdf['Period'].values.size)
            kep.approximate_planetary_params()
            tick = time.time()

            lsdf = sa.lsdf
            params = kep.paramsdf

            print("\nTime elapsed:: ", tick - tock, " seconds.\n")

            return lsdf, params
        except ValueError:
            print(sa.lsdf['Period'].values.size, " Planets Detected. This is too large for "
                                                 "MCMC analysis.")


def star_stats(star_df, star_mass=1, star_radius=1, effective_temp=5772, stellar_type="GV8"):
    """
    Returns a dataframe with basic statistics on each star. The statistics computed on the star are as follows:
    - The mass, radius, stellar type, and effective termperature of the star is recorded.
    - The H-Alpha, S-Value, Radial Velocity Amplitude, and RV error average, variance, and standard deviation are
       computed.

    This function is built assuming columns contain rv, error, H-Alpha, and S-Value data, formatted as specified
    in the public HiRES dataset (Butler, Vogt, Laughlin et al. 2017).
    :param star_df: pandas DataFrame, the stars' raw rv measurements. The columns must contain rv, error, H-Alpha,
    and S-Value data, formatted as specified in the public HiRES dataset (Butler, Vogt, Laughlin et al. 2017).
    :param star_mass: The mass of the star that was used inputted in the star data frame (in solar masses).
    Default is 1 solar mass.
    :param star_radius: The radius of the star that was used inputted in the star data frame (in solar radii). Default
    is 1 solar radius.
    :param effective_temp: The effective temperature of the star that was used inputted in the star data frame
    (in kelvin). The default is 5772 K, the effectieve temperature of the sun.
    :param stellar_type: The stellar type of the star. The default is GV8, the stellar type of the sun.
    :return: A dataframe with the statistics on the star.
    """

    halpha_avg = np.average(star_df["H Alpha"].values[star_df["H Alpha"].values != -1])
    halpha_var = np.var(star_df["H Alpha"].values[star_df["H Alpha"].values != -1])
    halpha_stdv = np.std(star_df["H Alpha"].values[star_df["H Alpha"].values != -1])

    sval_avg = np.average(star_df["S Value"].values[star_df["S Value"].values != -1])
    sval_var = np.var(star_df["S Value"].values[star_df["S Value"].values != -1])
    sval_stdv = np.std(star_df["S Value"].values[star_df["S Value"].values != -1])

    rv_avg = np.average(star_df["Radial Velocity (m/s)"].values[star_df["Radial Velocity (m/s)"].values != -1])
    rv_var = np.var(star_df["Radial Velocity (m/s)"].values[star_df["Radial Velocity (m/s)"].values != -1])
    rv_stdv = np.std(star_df["Radial Velocity (m/s)"].values[star_df["Radial Velocity (m/s)"].values != -1])

    err_avg = np.average(star_df["Error (m/s)"].values[star_df["Error (m/s)"].values != -1])
    err_var = np.var(star_df["Error (m/s)"].values[star_df["Error (m/s)"].values != -1])
    err_stdv = np.std(star_df["Error (m/s)"].values[star_df["Error (m/s)"].values != -1])

    stats = {"Stellar Type": [stellar_type], "Star Mass": [star_mass], "Star Radius": [star_radius],
             "Effective Temperature": [effective_temp],
             "H Alpha: Average": [halpha_avg], "H Alpha: Variance": [halpha_var], "H Alpha: STDEV": [halpha_stdv],
             "S-Value: Average": [sval_avg], "S-Value: Variance": [sval_var], "S-value: STDEV": [sval_stdv],
             "RV: Average": [rv_avg], "RV: Variance": [rv_var], "RV: STDEV": [rv_stdv],
             "RV Error: Average": [err_avg], "RV Error: Variance": [err_var], "RV Error: STDEV": [err_stdv]}

    return pd.DataFrame(stats)


def main():
    """
    Adopted functionality for making directories at https://realpython.com/working-with-files-in-python/
    """
    # planets = pd.read_csv("HiRESdetections.csv")["Name"].values.tolist()
    # planets = [s + "_KECK.txt" for s in planets]

    df = pd.DataFrame({"Stellar Type": [], "Star Mass": [], "Star Radius": [], "Effective Temperature": [],
                       "H Alpha: Average": [], "H Alpha: Variance": [], "H Alpha: STDEV": [],
                       "S-Value: Average": [], "S-Value: Variance": [], "S-value: STDEV": [],
                       "RV: Average": [], "RV: Variance": [], "RV: STDEV": [],
                       "RV Error: Average": [], "RV Error: Variance": [], "RV Error: STDEV": []})

    planets = pd.read_csv("ChosenPlanets.csv")["Name"].values.tolist()
    # print(planets)
    # pd.Series(planets).to_csv("ChosenPlanets.csv")

    for x in planets[21:]:
        file = x

        path_ = "DATA_real2/"
        print(path_)
        try:
            os.mkdir(path_)
        except OSError:
            print("Creation of the directory %s failed" % path_)
            path_ = ""
        else:
            print("Successfully created the directory %s " % path_)
            path_ = str(path_ + "/")

        path_ = "DATA_real2/" + os.path.splitext(file)[0]
        print("Analyzing the stellar system ", path_, " for basic statistics...")
        file = "keck_vels/" + file
        print(path_)
        try:
            os.mkdir(path_)
        except OSError:
            print("Creation of the directory %s failed" % path_)
            path_ = ""
        else:
            print("Successfully created the directory %s " % path_)
            path_ = str(path_ + "/")

        stardf = star_stats(Reader.read(file))
        df = pd.concat([df, stardf], ignore_index=True)
        print(detect(file, savedir=path_))

    print("Done! Here is the final dataset: \n", df)
    df.to_csv("DATA_real/StellarSampleStats.csv")


if __name__ == '__main__':
    path = "DATA"
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
        path = ""
    else:
        print("Successfully created the directory %s " % path)
        path = str(path + "/")
    main()
