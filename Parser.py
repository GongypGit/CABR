import numpy as np
import pandas as pd
import re
import os


class Parser():
    # Backend functions not which process the text files from the ABR program

    def __init__(self, filepath):

        self.filepath = filepath


        self.data = pd.read_csv(self.filepath, '\t', header=6)
        self.data = self.data.drop(self.data.columns.values[-1], axis=1)

        headerinfo = pd.read_csv(self.filepath,
                                 ':',
                                 names=['name', 'value'],
                                 skipfooter=len(self.data) + 5,
                                 engine='python')

        self.threshold = headerinfo['value'].values[0]
        self.freq = headerinfo['value'].values[1]
        self.levels = self.data['Level'].values

        self.amp = np.abs(self.data['N1 Amplitude'].values - self.data['P1 Amplitude']).values


    def write_csv(self):
        self.data.to_csv(self.filepath[0:-4:1] + '.csv')

    def get_frequencies(self):
        return self.freq

    def get_threshold(self):
        return self.threshold

    def get_level(self):
        return self.levels

    def get_amplitudes(self):
        return self.amp

    def get_id(self):
        regex = r'ABR-\d\d?\d?\d?-'
        filename = os.path.basename(self.filepath)
        id = re.search(regex, filename)
        id = re.search(r'\d\d?\d?\d?', id[0])

        return id[0]


