from parser import EPL
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import copy

import itertools

class experiment:

    def __init__(self, suppath, file_regex='*-analyzed.txt', parser=EPL):

        self.suppath = suppath
        self.parser = parser

        self.conditions = []
        self.animals = []
        self.frequency = []


        for dirs in glob.glob(suppath + '/*/'):
            # Get experimental conditions in folders below suppath
            self.conditions.append(os.path.basename(dirs[0:-1:1]))

        condition_dict = []

        for condition in self.conditions:
            #  loop through experimental conditions and add animal folders to animals list

            pathlist = glob.glob(self.suppath + condition+'/*/')

            animallist = []
            experiment_classes = []


            for animal in pathlist:

                animallist.append(os.path.basename(animal[0:-1:1]))
                experiment_path_list = glob.glob(self.suppath + condition + '/' + os.path.basename(animal[0:-1:1]) + '/' + file_regex)
                animal_classes = []

                # if self.frequencies:  # Got to do it this way because numpy array is weird
                #     freq_set = True

                for experiment in experiment_path_list:

                    if not self.frequency:
                        # We need a list of frequencies later in the program.
                        # Sets freq list if it hasent been set yet
                        self.frequency.append(self.parser(experiment).frequency)

                    # Create a list of parsers classes containing data of each animal
                    animal_classes.append(self.parser(experiment))

                # Create a list of all animal lists for a single experimental condition
                experiment_classes.append(animal_classes)

            # Create a dictionary for a single condition of the form {Animal_ID: [List of animal trials]}
            condition_dict.append(dict(zip(animallist, experiment_classes)))

        # Create an dict for the experiment of the form {Condition 1: condition_dict_1,
        #                                                Condition 2: condition_dict_2}
        self.experiment_dict = dict(zip(self.conditions, condition_dict))
        self.frequency.sort()

    @staticmethod
    def list_to_str(original):
        co = copy.copy(original)
        new = []
        while co:
            new.append(str(co.pop(0)))
        return new

    def get_experiment(self):
        return self.experiment_dict



class ABR(experiment):

    def __init__(self, path, ParsingClass):

        self.path = path
        self.parse = ParsingClass

        # Need to do this to generate the experiment dict
        super().__init__(suppath=path, parser=ParsingClass)

        self.plot = self.Plot(self.experiment_dict, self.list_to_str)

    @staticmethod
    def _write_csvmat_to_file(file, csvmat):
        """
        Necessary backend function for printing csv files
        Basically transposes csvmat and fills in extra spaces with spaces (' ')
        necessary because we may want to print to a csv a list of lists with different sizes inside
        in general we have lists which are the columns of a csv
        i.e. [Header, 1, 2, 3, 4, 5]

        we need a way to print all the headers first, then the data in the appropriate columns

        example:
        csvmat = [[1, 2], [a, b, c, d], [x, y, z]]
        csvmat = _write_csvmat_to_file(file, csvmat)
        csvmat = [[1, a, x], [2, b, y], [ , c, z], [ , d, ]]

        file.wirte(csvmat)

        :param file:
        :param csvmat:
        :return:
        """
        strmat = []
        maxlen = 0

        for i in range(len(csvmat)):
            if maxlen <= len(csvmat[i]):
                maxlen = len(csvmat[i])

        for i in range(maxlen):
            line = []
            for j in range(len(csvmat)):
                try:
                    line.append(str(csvmat[j][i]) + ',')
                except:
                    line.append(' ,')
            strmat.append(line)

        for i in range(len(strmat)):
            if i > 0:
                file.write('\n')
            for j in range(len(strmat[0])):
                file.write(strmat[i][j])

    def write_agf_csv(self):

        for f in self.frequency:

            csvmat = []
            strmat = []

            # Basically this creates a string that can be transposed later
            # [Frequency, 1, 2, 3, 4, 5, 6]
            # This theme is conserved throughout the function
            csvmat.append(['frequency'] + [str(f)])

            # Need to open twice, first "w" or write, is to open a new blank file
            file = open(str(f) + 'kHz_data.csv', 'w')

            for condition in self.experiment_dict:


                csvmat.append(['Condition'] + [condition])

                for animal in self.experiment_dict[condition]:

                    for run in self.experiment_dict[condition][animal]:

                        if f == run.frequency:
                            csvmat.append(['Animal'] + [str(run.id)])
                            csvmat.append(['Level'] + self.list_to_str(run.level))
                            csvmat.append(['Amplitudes'] + self.list_to_str(run.amplitudes))

                # What im doing here is finding the maximum length of sublists
                # contained within csvmat
                # need to do this so I can create a perfectly rectangular list of lists
                # which would be impossible with two long lists of condition, frequency, and animal
            #     maxlen = 0
            #     for i in range(len(csvmat)):
            #         if maxlen <= len(csvmat[i]):
            #             maxlen = len(csvmat[i])
            #
            # for i in range(maxlen):
            #     line = []
            #     for j in range(len(csvmat)):
            #         try:
            #             line.append(str(csvmat[j][i]) + ',')
            #         except:
            #             line.append(' ,')
            #     strmat.append(line)
            #
            # for i in range(len(strmat)):
            #     if i > 0:
            #         file.write('\n')
            #     for j in range(len(strmat[0])):
            #         file.write(strmat[i][j])

            self._write_csvmat_to_file(file,csvmat)

            file.close()

    def write_thr_csv(self):

        for condition in self.experiment_dict:

            csvmat = []

            csvmat.append(['Condition'] + [condition])

            # Need to open twice, first "w" or write, is to open a new blank file
            file = open(condition + '_Thresholds.csv', 'w')

            for animal in self.experiment_dict[condition]:

                csvmat.append([animal])

                freq = []
                threshold = []

                for run in self.experiment_dict[condition][animal]:

                    freq.append(run.frequency)
                    threshold.append(run.threshold)

                threshold = [y for y, _ in sorted(zip(threshold, freq))]

                freq.sort()

                freq = ['Frequency'] + self.list_to_str(freq)
                threshold = ['Threshold'] + self.list_to_str(threshold)

                csvmat.append(freq)
                csvmat.append(threshold)

            self._write_csvmat_to_file(file, csvmat)
            file.close()

    class Plot():

        def __init__(self, experiment_dict, l2s):

            self.experiment_dict = experiment_dict
            self.list_to_string = l2s



        def _mean(self, x, y):
            """
            Takes in a list of lists of differing sizes containing a possibly different lengths
            and makes a flattened list of the mean

            :param self:
            :param data:
            :return:
            """
            z = zip(x, y)

            X = np.array([])


            for a in x:
                for i in a:
                    if not np.any(X == i):
                        X = np.append(X,i)

            Y = copy.copy(X)

            for index,val in enumerate(X):
                m = 0 # mean
                i = 0 # iteration
                for a,b in zip(x,y):
                    if len(a) != len(b):
                        raise IndexError('X,Y dimmensions missmatch')

                    if np.any(np.array(a) == val):
                        m += np.array(b)[np.array(a) == val]
                        i += 1

                if i > 0:
                    Y[index] = m/i
                else:
                    Y[index] = m


            return X.tolist(), Y.tolist()

        def _var(self, x:list, y:list):
            """
            Takes in a list of lists of differing sizes containing a possibly different lengths
            and makes a flattened list of the mean

            :param self:
            :param data:
            :return:
            """

            X, Y_mean = self._mean(x,y)

            Y = np.array(copy.copy(X))

            for index, val in enumerate(X):
                m = 0  # mean
                i = 0  # iteration
                for a, b in zip(x, y):
                    if np.any(np.array(a) == val):
                        m += (np.array(b)[np.array(a) == val] - Y_mean[index])**2
                        i += 1

                if i > 1:
                    Y[index] = m / (i-1)
                elif i == 0 or i == 1:
                    Y[index] = 0

            return X, Y.tolist()

        def _std(self,x,y):
            X,Y = self._var(x,y)
            for i,val in enumerate(Y):
                Y[i] = val ** 0.5
            return X,Y

        def threshold(self):

            fig,ax = plt.subplots()
            fig.set_size_inches(5,4)
            ax.set_xscale('log')
            legend_elements = []
            for i,condition in enumerate(self.experiment_dict):

                legend_elements.append(Line2D([0],[0],color='C'+str(i), lw=2, label=str(condition)))
                THR = []
                FREQ = []

                for animal in self.experiment_dict[condition]:
                    freq = []
                    thr = []
                    for run in self.experiment_dict[condition][animal]:
                        freq.append(run.frequency)
                        thr.append(run.threshold)
                    thr = [y for y, _ in sorted(zip(thr, freq))]
                    freq.sort()
                    THR.append(thr)
                    FREQ.append(freq)
                    ax.plot(freq, thr, '.-', c='C'+str(i), alpha=0.1)

                FREQ_mean,THR_mean = self._mean(FREQ,THR)
                _,THR_variance = self._std(FREQ,THR)
                plt.fill_between(FREQ_mean, np.array(THR_mean) - np.array(THR_variance),
                                 np.array(THR_mean) + np.array(THR_variance), alpha = .2, color = 'C'+str(i))

                ax.plot(FREQ_mean, THR_mean, '.-', c='C'+str(i), linewidth=2)



            ax.set_xscale('log')
            ax.set_xticks(FREQ_mean)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.ticklabel_format(style='plain')

            ax.set_xlabel('Frequency (kHz)')
            ax.set_ylabel('Threshold (dB)')
            ax.legend(handles=legend_elements, loc='best')
            plt.show()


        def agf(self):
            raise NotImplementedError


