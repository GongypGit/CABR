import parser
import CABR

abr = CABR.ABR(path='/Users/cx926/Desktop/CABR/Super/', ParsingClass=parser.EPL)
abr.write.agf()
abr.write.threshold()
abr.get_experiment()
abr.plot.threshold()
abr.plot.agf(8)
