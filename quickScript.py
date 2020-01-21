from parser import RawABRthreshold
import glob


path = '/Users/cx926/Desktop/CABR/Corey/'


files = glob.glob(path+'ABR*')
print(files)
for file in files:
    run = RawABRthreshold(file)
    run.save_figure('/Users/cx926/Desktop/CABR/UnanalyzedData/Images/')

