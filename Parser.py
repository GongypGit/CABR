import os
import re

import numpy as np
import pandas as pd

import scipy.signal
import scipy.optimize
import scipy.signal
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

class Parser:

    def __init__(self, filepath=None, treatment=None):

        self._amplitudes  = []
        self._frequencies = []
        self._level       = []
        self._threshold   = []
        self._treatment   = []
        self._filepath    = []

        self.amplitudes   = []
        self.frequencies  = []
        self.level        = []
        self.threshold    = []

        self.treatment = treatment
        self.filepath = filepath

    def write_csv(self):
        self.data.to_csv(self.filepath[0:-4:1] + '.csv')

    @property
    def treatment(self):
        if self._treatment:
            return self._treatment
        else:
            raise UnboundLocalError('Treatment has not been set. Use set_treatment() method to set treatment')

    @treatment.setter
    def treatment(self, value):
        self._treatment = value

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        self._frequency = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    @property
    def levels(self):
        return self._levels

    @levels.setter
    def levels(self, value):
        self._levels = value

    @property
    def amplitudes(self):
        return self._amplitudes

    @amplitudes.setter
    def amplitudes(self, value):
        self._amplitudes = value

    @property
    def filepath(self):
        return self._filepath

    @filepath.setter
    def filepath(self, value):
        self._filepath = value

    @property
    def id(self):
        """
        Custom Function for

        :return:
        """
        raise NotImplementedError

    @id.setter
    def id(self, filename):
        raise NotImplementedError


class EPL(Parser):
    # Backend functions not which process the text files from the ABR program

    def __init__(self, filepath, treatment=None):
        self.filepath = filepath
        self.treatment = treatment
        self.id = filepath

        self.data = pd.read_csv(self.filepath, '\t', header=6)
        self.data = self.data.drop(self.data.columns.values[-1], axis=1)

        headerinfo = pd.read_csv(self.filepath,
                                 ':',
                                 names=['name', 'value'],
                                 skipfooter=len(self.data) + 5,
                                 engine='python')

        self.threshold = headerinfo['value'].values[0]
        self.frequency = headerinfo['value'].values[1].tolist()
        self.levels = self.data['Level'].values.tolist()

        self.amplitudes = np.abs(self.data['N1 Amplitude'].values - self.data['P1 Amplitude']).values.tolist()


    def write_csv(self):
        self.data.to_csv(self.filepath[0:-4:1] + '.csv')

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, filename):
        regex = r'ABR-(\d){1,}-'
        filename = os.path.basename(self.filepath)
        id = re.search(regex, filename)
        id = re.search(r'(\d){1,}', id[0])

        self._id = id[0]


class RawABRthreshold(Parser):
    def __init__(self, filepath, treatment=None):
        super(Parser,self).__init__()

        self.filepath = filepath
        self.treatment = treatment
        self.id = filepath

        levelheader = ':LEVELS:'

        file = open(filepath,'rb')
        leveltext = file.read()
        file.close()

        # Have to convert to list then back to np array or it breaks. I dont know why.
        levels = re.search('\:LEVELS\:(\d\d;){1,}', str(leveltext))
        levels = levels[0][len(levelheader):-1:1]
        levels = np.fromstring(levels, dtype=int, sep=';')
        levels = levels.tolist()

        self.levels = np.array(levels)

        freq = re.search('FREQ\:\ \ ?(\d){1,}.(\d){1,}', str(leveltext))

        if freq:
            freq = freq[0][6:-1:1]
        else:
            freq = None

        freq = float(freq)
        freq = np.floor(freq)
        self.frequency = freq

        self.data = pd.read_csv(self.filepath, '\t', names=levels, header = 10, engine='python')

        self.threshold, self.case, self.flag = self.calculate_threshold()


    def calculate_threshold(self):

        if 'criterion' not in globals():
            criterion = .35

        levels, corrmat = self._calculate_corr_level_function_chris_version()

        def sigmoid(x, a, b, c, d):
            return a + (b-a)/(1 + (10**(d*(c-x))))

        def power(x, a, b, c):
            return a * x ** b + c

        def inverse_power(y, coeff):
            return ((y-coeff[2])/coeff[0]) ** (1/coeff[1])

        def inverse_sigmoid(y, coeff):
            return coeff[2] - np.log10(((coeff[1]-coeff[0])/(y-coeff[0])) -1)/coeff[3]

        try:
            sig_coeff,_ = scipy.optimize.curve_fit(sigmoid, levels, corrmat,
                                            p0=[np.min(corrmat),np.max(corrmat), np.max(levels)/2, 1/np.max(levels)], maxfev = 10000)
        except:
            Warning('Could not converge on sigmoid Fit: Manually setting sig_coeff')
            sig_coeff = [100,100,100,100]

        try:
            pow_coeff,_ = scipy.optimize.curve_fit(power,levels,corrmat,
                                                   p0 = [1/70, 1, .1], maxfev=10000)
        except:
            Warning('Could not converge on power Fit: Manually setting sig_coeff')
            pow_coeff = [100,100,100]
            # Handles Fringe Case when scipy.optimize.curve_fit cant hande the fit, manually sets coeff
            # to something unreasable so that pow will have a low r2 and never be used for threshold determination

        RMS_sig = mean_squared_error(corrmat, sigmoid(levels,
                                                      sig_coeff[0],
                                                      sig_coeff[1],
                                                      sig_coeff[2],
                                                      sig_coeff[3]))
        RMS_pow = mean_squared_error(corrmat, power(levels,
                                                      pow_coeff[0],
                                                      pow_coeff[1],
                                                      pow_coeff[2]))

        pow_r2 = sklearn.metrics.r2_score(corrmat, power(levels,
                                                         pow_coeff[0],
                                                         pow_coeff[1],
                                                         pow_coeff[2]))

        C1 = sig_coeff[0] < criterion and sig_coeff[1]  > criterion
        C1 = C1 and sig_coeff[3] > 0.005 and sig_coeff[3] < 1
        C2 = RMS_sig < RMS_pow
        C3 = pow_r2 > 0.7
        C4 = np.max(corrmat) > criterion
        threshold = np.max(corrmat)
        FLAG = 'clear'
        CASE='---'

        if C1:
            if C2:
                threshold = inverse_sigmoid(criterion, sig_coeff)
                CASE = 'A'
            else:
                if C3:
                    threshold = inverse_power(criterion, pow_coeff)
                    CASE = 'B'
                else:
                    if C4:
                        threshold = inverse_power(criterion, pow_coeff)
                        FLAG = 'NOISY'
                        CASE = 'C'

                    else:
                        "D: Visual"
                        FLAG = "COULD NOT BE DETERMINED"
                        CASE = 'D'
        else:  # if C1
            if C3:
                threshold = inverse_power(criterion, pow_coeff)
                CASE = 'B'
            else:
                if C4:
                    threshold = inverse_power(criterion, pow_coeff)
                    CASE = 'C'
                    FLAG = 'NOISY'
                else:
                    FLAG = "COULD NOT BE DETERMINED"
                    CASE = 'D'

        badthreshold = isinstance(threshold, complex)
        badthreshold = badthreshold or np.isnan(threshold)
        badthreshold = badthreshold or int(threshold) < 1 or int(threshold) > 100

        if badthreshold:
            threshold = None
            print('Threshold could not be determined')

        return threshold, CASE, FLAG

    @staticmethod
    def _xcov(x,y):
        """
        Reimplementation of MATLAB's xcov function

        :param x:
        :param y:
        :return:
        """
        x = (x - np.mean(x))
        y = (y - np.mean(y))

        x_autocor = np.correlate(x,x,'full')
        x_autocor = x_autocor[len(x_autocor)//2]

        y_autocor = np.correlate(y,y,'full')
        y_autocor = y_autocor[len(y_autocor)//2]

        corr = scipy.signal.correlate(x, y)

        corr = corr / np.sqrt((x_autocor * y_autocor))

        return corr

    def _calculate_corr_level_function(self):

        len_datamat = len(self.data.columns)

        corrmat = np.zeros(len_datamat-1)
        levels = np.zeros(len_datamat-1)

        for i, level in enumerate(self.data.columns):

            if i == len_datamat-1:
                break

            # Pull in unfiltered data from txt file
            x = self.data[level]
            y = self.data[self.data.columns[i+1]]

            # Construct a filter as per Kirupa's paper
            # Note, they used a zero-pole filter, while Im using a sos filter
            # It should be the same
            filter = scipy.signal.butter(4,(200,10000),
                                         fs=1/(40*10**-6),
                                         btype='bandpass', output='sos')

            x = scipy.signal.sosfilt(filter, x)
            y = scipy.signal.sosfilt(filter, y)

            # # Some Issues with Kirupa Code! Excluding last half of ABR Waveform!!!!!!
            # ######
            # x = x[0:len(x)//3:1]
            # y = y[0:len(y)//3:1]
            # ######
            # #


            # Python implementation of matlab's xcov function
            corr = self._xcov(x,y)



            # Only want correlation at lag 0, which is in middle of matrix
            # corrmat[i] = corr[len(corr)//2]

            # # CHRIS TWEAK TO ALGORITHM: due to time lag, in abr, we take max autocorr over a range
            # #
            # corrmat[i] = np.max(corr[len(corr)//2 - 30 : len(corr)//2 + 30: 1])
            # #
            # #

            levels[i] = float(level)

        return levels, corrmat

    def _calculate_corr_level_function_chris_version(self):


        filter = scipy.signal.butter(4, (200, 10000),
                                     fs=1 / (40 * 10 ** -6),
                                     btype='bandpass', output='sos')
        len_datamat = len(self.data.columns)

        corrmat = np.zeros(len_datamat)
        levels = np.zeros(len_datamat)

        for i, lev in enumerate(self.data.columns):

            levels[i] = float(lev)
            corr = []

            for j, compare_level in enumerate(self.data.columns):

                if lev != compare_level:
                    x = self.data[lev]
                    y = self.data[compare_level]

                    x = scipy.signal.sosfilt(filter, x)
                    y = scipy.signal.sosfilt(filter, y)

                    x = x[0:len(x) // 1:1]
                    y = y[0:len(y) // 1:1]

                    autocorrelation = self._xcov(x, y)
                    corr.append(np.max(autocorrelation[len(autocorrelation) // 2 - 30: len(autocorrelation) // 2 + 30: 1]))

            corrmat[i] = np.mean(corr)
        #
        # for i, level in enumerate(self.data.columns):
        #
        #     if i == len_datamat - 1:
        #         break
        #
        #     # Pull in unfiltered data from txt file
        #
        #
        #
        #     # Construct a filter as per Kirupa's paper
        #     # Note, they used a zero-pole filter, while Im using a sos filter
        #     # It should be the same
        #
        #
        #
        #
        #     # Some Issues with Kirupa Code! Excluding last half of ABR Waveform!!!!!!
        #     ######
        #
        #     ######
        #     #
        #
        #     # Python implementation of matlab's xcov function
        #
        #
        #     # Only want correlation at lag 0, which is in middle of matrix
        #     # corrmat[i] = corr[len(corr)//2]
        #
        #     # CHRIS TWEAK TO ALGORITHM: due to time lag, in abr, we take max autocorr over a range
        #     #
        #
        #     #
        #     #
        #
        #     levels[i] = float(level)

        return levels, corrmat

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, filename):
        regex = r'ABR-(\d){1,}-'
        filename = os.path.basename(self.filepath)
        id = re.search(regex, filename)
        id = re.search(r'(\d){1,}', id[0])

        self._id = id[0]

    def diagnostics(self):
        """
        Diagnostics on the parser thresholding algorithm
        :return:
        """
        print(self.filepath)
        print('Threshold: ', self.threshold)
        print('CASE: ', self.case)
        print('FLAG: ', self.flag)

        return self.case, self.flag

    def save_figure(self, savepath):
        """
        Save a figure of the abr waveforms to a seperate folder denoted by savepath

        :param savepath:
        :return:
        """
        fig, ax = plt.subplots(figsize=(4,5))
        offset = np.logspace(0, 7.5, len(self.levels), base=1.3)

        for i, value in enumerate(self.data.columns):

            waveform = self.data[value]
            length = len(waveform)
            time = np.linspace(0, len(waveform) * 40 * 10 ** -6, len(waveform)) * 1000

            bandpass_filter = scipy.signal.butter(4, (200,10000),
                                                  fs=1/(40*10**-6),
                                                  btype='bandpass', output='sos')

            waveform = scipy.signal.sosfilt(bandpass_filter, waveform)

            waveform = waveform + offset[i]

            ax.plot(time[1:length//2:1], waveform[1:length//2:1], c='k')

        plt.yticks(offset, self.data.columns)
        plt.xlabel('Time (ms)')
        plt.ylabel('Level (dB)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)

        name = self.id + '-' + str(self.frequency) + '.jpeg'

        plt.savefig(savepath+name)
        plt.close()




