import os
import re

import numpy as np
import pandas as pd


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
        return self._amp

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

        self.amp = np.abs(self.data['N1 Amplitude'].values - self.data['P1 Amplitude']).values.tolist()

    def write_csv(self):
        self.data.to_csv(self.filepath[0:-4:1] + '.csv')

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, filename):
        regex = r'ABR-\d\d?\d?\d?-'
        filename = os.path.basename(self.filepath)
        id = re.search(regex, filename)
        id = re.search(r'\d\d?\d?\d?', id[0])

        self._id = id[0]
