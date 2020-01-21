import parser
import CABR

# Sloppy Code but necessary in the circumstance
# Sets criterion for thresholding
global criterion
criterion = .25

abr = CABR.ABR(path='/Users/cx926/Desktop/CABR/Chunjie/', file_regex='ABR-*', ParsingClass=parser.RawABRthreshold)
# # abr.write.agf()
# # abr.write.threshold()
# # abr.get_experiment()
abr.plot.threshold(seperate_conditions=True, errbar = False)
# abr.plot.agf(8)


p = parser.RawABRthreshold('/Users/cx926/Desktop/CABR/Chunjie/Pre/57/ABR-70157-1')
# p.save_figure(None)
