import parser
import CABR3

p = parser.EPL('/Users/cx926/Desktop/CABR/data/1191/ABR-1191-4-analyzed.txt')



test = CABR3.experiment('/Users/cx926/Desktop/CABR/Super/')
test.write_csv()