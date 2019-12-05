from Parser import Parser
import glob
import os
p = Parser('/Users/chrisbuswinka/Documents/CABR/data/1191/ABR-1191-4-analyzed.txt')




class CABR():

    def __init__(self, path, ParsingClass ):

        self.path = path



        self.parse = ParsingClass
        self.experiment = experiment(path)

class experiment():

    def __init__(self,path ):

        print('Found Experimental Conditions:')
        for dirs in glob.glob(path+'/*/'):
            print(os.path.basename(dirs[0:-1:1]))



a = CABR('./data/', Parser)

a.parse.get_id()