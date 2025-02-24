import scipy.io
from scipy.ndimage import variance

data = scipy.io.loadmat('vectorField_RL_2019_P2.mat')

fields = data['mapGrid']
f_variance =  data['mapFunc'][0][0][2]

frequency_id = 0
field = fields[0][frequency_id]

dx = field[0]
dy = field[1]
STD = field[2]
SE = field[3]


print(field[0][3][4])

a = 1