import parser
import CABR3

p = parser.EPL('/Users/cx926/Desktop/CABR/data/1191/ABR-1191-4-analyzed.txt')

abr = CABR3.ABR(path='/Users/cx926/Desktop/CABR/Super/', ParsingClass=parser.EPL)
abr.write_thr_csv()
abr.get_experiment()
# x,y = [[1,2,3],[1,2,3,4]],  [[2,7,11],[3,8,12,15]]
#
# a,b = abr.plot._mean(x,y)
# b = abr.plot._var(x,y)
# print(b)

abr.plot.threshold()