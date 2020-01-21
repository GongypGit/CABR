from parser import EPL
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import copy


class Experiment:

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

                    try:
                        self.parser(experiment) # Sometimes the parser will fail. In this case just ignore it.
                    except:
                        continue

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
        """
        Take in a list and return a list of strings of each object in list. [1,2,3,4] --> ['1', '2', '3', '4']

        :param original:
        :return:
        """
        co = copy.copy(original)
        new = []
        while co:
            new.append(str(co.pop(0)))
        return new

    def get_experiment(self):
        return self.experiment_dict


class ABR(Experiment):

    def __init__(self, path, file_regex, ParsingClass):

        self.path = path
        self.parse = ParsingClass


        # Need to do this to generate the experiment dict
        super().__init__(suppath=path, file_regex= file_regex, parser=ParsingClass)

        self.plot = self.Plot(self.experiment_dict, self.list_to_str)
        self.write = self.Write(self.experiment_dict, self.list_to_str, self.frequency)

    class Write:
        def __init__(self, experiment_dict, list_to_str,frequency):
            self.experiment_dict = experiment_dict
            self.list_to_str = list_to_str
            self.frequency = frequency

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

        def agf(self):

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
                                csvmat.append(['Level'] + self.list_to_str(run.levels))
                                csvmat.append(['Amplitudes'] + self.list_to_str(run.amplitudes))

                self._write_csvmat_to_file(file,csvmat)
                file.close()

        def threshold(self):

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

                    threshold = [y for x, y in sorted(zip(freq, threshold))]

                    freq.sort()

                    freq = ['Frequency'] + self.list_to_str(freq)
                    threshold = ['Threshold'] + self.list_to_str(threshold)

                    csvmat.append(freq)
                    csvmat.append(threshold)

                self._write_csvmat_to_file(file, csvmat)
                file.close()

    class Plot:

        def __init__(self, experiment_dict, l2s):
            self.experiment_dict = experiment_dict
            self.list_to_string = l2s

            FREQ = []

            for i,condition in enumerate(self.experiment_dict):
                for animal in self.experiment_dict[condition]:
                    freq = []
                    thr = []
                    for run in self.experiment_dict[condition][animal]:
                        if run.threshold is not None:
                            freq.append(run.frequency)
                            thr.append(run.threshold)

                    FREQ.append(freq)

            # self.frequency_list, _ = self._mean(FREQ, FREQ) # WHY?
            self.frequency_list = FREQ
            self.frequency_list = sorted(self.frequency_list)

        @staticmethod
        def _mean(x: list, y: list):
            """
            This function takes in two lists of lists, x,y and creates a new list X of all the unique values in x, and
            Y of the mean value of all of unique values in x.

            e.g. x = [[1,2,3],[4,2,3]] y = [[11,12,13,][14,15,16]]
            -> X = [1,2,3,4] Y = [1, 17/2, 19/2, 14]

            :param x: list of lists that may have differnt values
            :param y: data you want averaged
            :return:
            """

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

                        m += np.array(b)[np.array(a) == val].mean()
                        i += 1

                if i > 0:
                    Y[index] = m/i
                else:
                    Y[index] = m


            X = X.tolist()
            Y = Y.tolist()

            Y = [y for x, y in sorted(zip(X, Y))]
            X = sorted(X)

            return X, Y

        def _var(self, x: list, y: list):
            """
            Does the same as _mean, but for variance.

            :param self:
            :param data:
            :return:
            """
            if np.any(np.array(x) == None) or np.any(np.array(y) == None):
                raise ValueError('List cannot conatin None Value\nx: ' + str(x) +'\ny: ' + str(y))

            X, Y_mean = self._mean(x,y)
            Y = np.array(copy.copy(X))

            for index, val in enumerate(X):
                m = 0  # mean
                i = 0  # iteration
                for a, b in zip(x, y):
                    if np.any(np.array(a) == val):
                        m += (np.array(b)[np.array(a) == val].mean() - Y_mean[index])**2
                        i += 1

                if i > 1:
                    Y[index] = m / (i-1)
                elif i == 0 or i == 1:
                    Y[index] = 0

            return X, Y.tolist()

        def _std(self, x: list, y: list):
            """
            Does the same thing as _var but for standard deviataion.

            :param x:
            :param y:
            :return:
            """
            X,Y = self._var(x,y)
            for i,val in enumerate(Y):
                Y[i] = val ** 0.5
            return X,Y

        def threshold(self, errbar=False, seperate_conditions=False):

            if not seperate_conditions:

                fig,ax = plt.subplots()
                fig.set_size_inches(5,4)
                ax.set_xscale('log')
                legend_elements = []

            for i, condition in enumerate(self.experiment_dict):
                if seperate_conditions:
                    fig, ax = plt.subplots()
                    fig.set_size_inches(5, 4)
                    ax.set_xscale('log')
                    legend_elements = []

                legend_elements.append(Line2D([0],[0],color='C'+str(i), lw=2, label=str(condition)))

                THR = []
                FREQ = []

                for animal in self.experiment_dict[condition]:

                    freq = []
                    thr = []

                    for run in self.experiment_dict[condition][animal]:

                        if run.threshold is not None:
                            freq.append(run.frequency)
                            thr.append(run.threshold)
                        else:
                            run.save_figure('/Users/cx926/Desktop/CABR/ChunjieBadABR/')

                    thr = [y for x, y in sorted(zip(freq, thr))] # sorts thr for values in freq

                    freq.sort()
                    THR.append(thr)
                    FREQ.append(freq)
                    ax.plot(freq, thr, '.-', c='C'+str(i), alpha=0.1)


                FREQ_mean, THR_mean = self._mean(FREQ, THR)
                _, THR_variance = self._std(FREQ, THR)

                if errbar:
                    ax.errorbar(FREQ_mean, THR_mean, yerr=THR_variance, c='C'+str(i), linewidth=2)
                else:
                    plt.fill_between(FREQ_mean, np.array(THR_mean) - np.array(THR_variance),
                                     np.array(THR_mean) + np.array(THR_variance), alpha = .2, color = 'C'+str(i))
                    plt.plot(FREQ_mean,THR_mean, '.-', c='C'+str(i))

                if seperate_conditions:
                    ax.set_xscale('log')
                    ax.set_xticks(FREQ_mean)
                    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
                    ax.spines['right'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.ticklabel_format(style='plain')
                    ax.set_xlabel('Frequency (kHz)')
                    ax.set_ylabel('Threshold (dB)')
                    ax.legend(handles=legend_elements, loc='best', frameon=False)
                    plt.title('Threshold')
                    plt.show()

            if not seperate_conditions:
                ax.set_xscale('log')
                ax.set_xticks(FREQ_mean)
                ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.ticklabel_format(style='plain')
                ax.set_xlabel('Frequency (kHz)')
                ax.set_ylabel('Threshold (dB)')
                ax.legend(handles=legend_elements, loc='best',frameon=False)
                plt.title('Threshold')
                plt.show()

        def agf(self, frequency=None, errbar=None):

            fig,ax = plt.subplots()
            fig.set_size_inches(5,4)
            ax.set_xscale('log')
            legend_elements = []

            for i, condition in enumerate(self.experiment_dict):

                legend_elements.append(Line2D([0],[0],color='C'+str(i), lw=2, label=str(condition)))
                lvl = []
                amp = []

                for animal in self.experiment_dict[condition]:
                    for run in self.experiment_dict[condition][animal]:
                        if np.array(run.frequency) == frequency:
                            ax.plot(run.levels, run.amplitudes, '-', c='C'+str(i), alpha=0.1)
                            amp.append(run.amplitudes)
                            lvl.append(run.levels)

                lvl_mean, amp_mean = self._mean(lvl, amp)
                _, amp_variance = self._std(lvl, amp)

                if errbar:
                    ax.errorbar(lvl_mean, amp_mean, yerr=amp_variance, c='C' + str(i), linewidth=2)
                else:
                    plt.fill_between(lvl_mean, np.array(amp_mean) - np.array(amp_variance),
                                     np.array(amp_mean) + np.array(amp_variance), alpha = .2, color = 'C'+str(i))
                    ax.plot(lvl_mean, amp_mean, '.-', c='C' + str(i), linewidth=2)



            plt.title('Amplitude Growth Function')
            ax.set_xticks(lvl_mean)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('Level (dB)')
            ax.set_ylabel('N1P1 Amplidue')
            ax.legend(handles=legend_elements, loc='best',frameon=False)

            plt.show()
