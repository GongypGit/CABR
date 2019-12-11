import parser
import CABR3

p = parser.EPL('/Users/cx926/Desktop/CABR/data/1191/ABR-1191-4-analyzed.txt')

abr = CABR3.ABR(path='/Users/cx926/Desktop/CABR/Super/', ParsingClass=parser.EPL)
abr.write_thr_csv()
abr.write_agf_csv()
abr.get_experiment()
abr.plot.threshold()
abr.plot.agf(8)
