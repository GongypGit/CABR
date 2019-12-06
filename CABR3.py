from parser import EPL
import glob
import os
import numpy as np
import itertools

p = EPL('/Users/cx926/Desktop/CABR/data/1191/ABR-1191-4-analyzed.txt')


class experiment:

    def __init__(self, suppath, file_regex='*-analyzed.txt', parser=EPL):

        self.suppath = suppath
        self.parser = parser

        self.conditions = []
        self.animals = []
        self.frequencies = []
        freq_set = False


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
                if self.frequencies:
                    freq_set = True

                for experiment in experiment_path_list:
                    if not freq_set:
                        self.frequencies.append(self.parser(experiment).frequencies)


                    animal_classes.append(self.parser(experiment))
                experiment_classes.append(animal_classes)
                print(experiment_classes)
            condition_dict.append(dict(zip(animallist, experiment_classes)))

        print(condition_dict)
        self.experiment_dict = dict(zip(self.conditions, condition_dict))
        self.frequencies.sort()

    @staticmethod
    def list_to_str(original):
        new = []
        while original:
            new.append(str(original.pop(0)))
        return new

    def write_csv(self):

        for f in self.frequencies:
            csvmat = []
            strmat = []
            csvmat.append(['frequency'] + [str(f)])

            file = open(str(f) + 'kHz_data.csv', 'w')
            file = open(str(f) + 'kHz_data.csv', 'a')

            for condition in self.experiment_dict:
                print(condition)
                csvmat.append(['Condition'] + [condition])
                for animal in self.experiment_dict[condition]:
                    print(animal)

                    for run in self.experiment_dict[condition][animal]:

                        if f == run.frequencies:

                            csvmat.append(['Animal'] + [str(run.id)])
                            csvmat.append(['Level']+self.list_to_str(run.level))
                            csvmat.append(['Amplitudes'] + self.list_to_str(run.amplitudes))

                # What im doing here is finding the maximum length of sublists
                # contained within csvmat
                # need to do this so I can create a perfectly rectangular list of lists
                # which would be impossible with two long lists of condition, frequency, and animal
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
                if i>0:
                    file.write('\n')
                for j in range(len(strmat[0])):
                    file.write(strmat[i][j])

            file.close()





        # for i in range(len(bigcsvmat))




    def get_experiment(self):
        return self.experiment_dict






class ABR():
    def __init__(self, path, ParsingClass):

        self.path = path
        self.parse = ParsingClass
        self.experiment = experiment()

    def test(self):
        print('butt')

    class experiment:

        def __init__(self,path):

            print('Found Experimental Conditions:')
            for dirs in glob.glob(path+'/*/'):
                print(os.path.basename(dirs[0:-1:1]))




