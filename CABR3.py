from parser import EPL
import glob
import os
import itertools

p = EPL('/Users/cx926/Desktop/CABR/data/1191/ABR-1191-4-analyzed.txt')


class experiment:

    def __init__(self, suppath, file_regex='*-analyzed.txt', parser=EPL):

        self.suppath = suppath
        self.parser = parser

        self.conditions = []
        self.animals = []


        for dirs in glob.glob(suppath + '/*/'):
            # Get experimental conditions in folders below suppath
            self.conditions.append(os.path.basename(dirs[0:-1:1]))

        condition_dict = []
        for condition in self.conditions:
            #  loop through experimental conditions and add animal folders to animals list
            pathlist = glob.glob(self.suppath + condition+'/*/')
            animallist = []


            for animal in pathlist:

                animallist.append(os.path.basename(animal[0:-1:1]))
                experiment_path_list = glob.glob(self.suppath + condition + '/' + os.path.basename(animal[0:-1:1]) + '/' + file_regex)
                experiment_classes = []

                for experiment in experiment_path_list:

                    experiment_classes.append(self.parser(experiment))
            condition_dict.append(dict(zip(animallist, experiment_classes)))

        self.experiment_dict = dict(zip(self.conditions, condition_dict))

    @staticmethod
    def list_to_str(original):
        new = []
        while original:
            new.append(str(original.pop(0)))

        return new


    def write_csv(self):

        freq = []
        for condition in self.experiment_dict:
            file = open(condition+'.csv','w')

            for animal in self.experiment_dict(condition):

                if not freq:
                    freq = self.experiment_dict[condition][animal].frequencies




















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




test = experiment('/Users/cx926/Desktop/CABR/Super/')
test.write_csv()