import numpy as np
import scipy

#NAPH:
x = np.array([0.887,2.735,2.927,4.523,7.960,12.890])
y1 = np.array([6.828,16.196,15.825,16.254,23.239,37.775])
y2 = np.array([1.143,3.693,3.592,3.709,5.610,9.567])
c1 = np.corrcoef(x, y1)
c2 = np.corrcoef(x, y2)
print('NAP corr (gcms vs STD) = ' + np.str(c1[1][0]))
print('NAP corr (gcms vs WAF) = ' + np.str(c2[1][0]))
#scipy.stats.pearsonr(x, y1)

err = np.abs(y1-x)/x*100
print(r'NAP relative error with STD: ' + np.str(np.round(err.mean())) + ' +- ' + np.str(np.round(err.std())))
err = np.abs(y2-x)/x*100
print(r'NAP relative error with WAF: ' + np.str(np.round(err.mean())) + ' +- ' + np.str(np.round(err.std())))



#PHE:
x = np.array([0.242,0.365,0.563,0.693,0.828,1.361])
y1 = np.array([3.291,4.059,4.498,5.673,8.300,13.900])
y2 = np.array([0.103,0.125,0.136,0.171,0.247,0.407])
c1 = np.corrcoef(x, y1)
c2 = np.corrcoef(x, y2)
print('PHE corr (gcms vs STD) = ' + np.str(c1[1][0]))
print('PHE corr (gcms vs WAF) = ' + np.str(c2[1][0]))

err = np.abs(y1-x)/x*100
print(r'PHE relative error with STD: ' + np.str(np.round(err.mean())) + ' +- ' + np.str(np.round(err.std())))
err = np.abs(y2-x)/x*100
print(r'PHE relative error with WAF: ' + np.str(np.round(err.mean())) + ' +- ' + np.str(np.round(err.std())))

#FLU:
x = np.array([0.098,0.200,0.197,0.351])
y1 = np.array([0.354,2.101,2.366,3.361])
y2 = np.array([-0.010,0.196,0.227,0.345])
c1 = np.corrcoef(x, y1)
c2 = np.corrcoef(x, y2)
print('FLU corr (gcms vs STD) = ' + np.str(c1[1][0]))
print('FLU corr (gcms vs WAF) = ' + np.str(c2[1][0]))

err = np.abs(y1-x)/x*100
print(r'FLU relative error with STD: ' + np.str(np.round(err.mean())) + ' +- ' + np.str(np.round(err.std())))
err = np.abs(y2-x)/x*100
print(r'FLU relative error with WAF: ' + np.str(np.round(err.mean())) + ' +- ' + np.str(np.round(err.std())))

#PYR:
x = np.array([0.024,0.069,0.056,0.078])
y1 = np.array([6.643,16.196,15.825,16.254])
y2 = np.array([4.241,9.950,9.724,9.986])
c1 = np.corrcoef(x, y1)
c2 = np.corrcoef(x, y2)
print('PYR corr (gcms vs STD) = ' + np.str(c1[1][0]))
print('PYR corr (gcms vs WAF) = ' + np.str(c2[1][0]))

err = np.abs(y1-x)/x*100
print(r'PYR relative error with STD: ' + np.str(np.round(err.mean())) + ' +- ' + np.str(np.round(err.std())))
err = np.abs(y2-x)/x*100
print(r'PYR relative error with WAF: ' + np.str(np.round(err.mean())) + ' +- ' + np.str(np.round(err.std())))
