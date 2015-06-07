import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

#represents the line
class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def get_y(self, x):
        return self.y1 + (self.y2 - self.y1) * (x - self.x1) / (self.x2 - self.x1)

#generate N samples
def generate_sample(n):
    sample = pd.DataFrame({'x': [], 'y': []})
    for i in range(0, n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        sample = pd.concat([sample, pd.DataFrame({'x': [x], 'y': y})],
                ignore_index = True)
    return sample 

# Perceptron Learning Algorithm (PLA)
# Generates data and runs PLA
# Input: n - sample size
#        show_plot - display a plot of data points, default = False
# Output: tuple of
#         - number of iterations required to converge
#         - probability of false predictions on new data
def pla(n, show_plot = False):

    # build a line through two points
    l = []
    for i in range(4):
        l.append(random.uniform(-1, 1))
    
    line = Line(l[0], l[1], l[2], l[3])
    
    xline = [-1, 1]
    yline = map(lambda p: line.get_y(p), xline)
    
    # generate a sample of size n
    sample = generate_sample(n)
    
    # add target function, 1 if above the line, -1 if below
    sample['f'] = sample.apply(lambda row: 1 if row['y'] > line.get_y(row['x']) else -1, axis = 1)
    
    # convert X to matrix
    x = sample.as_matrix(columns = ['x', 'y'])
    
    # add intercept to data
    x = np.insert(x, 0, 1, axis = 1)
    
    # Initialization
    # vector of weights including intercept
    w = np.array([[0, 0, 0]]).transpose()
    
    # counter of iterations
    it = 0
    
    while(True):
        # compute hypothesis h(x)
        h = np.sign(np.dot(x, w))
    
        # determine whether h(x) == f(x)
        match = (np.sign(sample['f'].values) == np.sign(h.transpose())).transpose()
    
        # find the index i of the first x_i such that h(x_i) != f(x_i)
        #i = np.where(match == False)[0][0]
        i_array = np.nonzero(match == False)
    
        if (i_array[0].size == 0):
            break
        else:
            # update weights vector, w = w + f(x_i) * x_i
            i = i_array[0][0]
            w = (w.transpose() + sample.f[i] * x[i,]).transpose()
            it += 1
    
    if (show_plot == True):
        pos = sample[sample.f == 1]
        neg = sample[sample.f == -1]
        
        plt.figure()
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.plot(xline, yline, c = 'green', label = 'Target')
        plt.scatter(x = pos.x, y = pos.y, s = 20, c = 'red', marker = 'o')
        plt.scatter(x = neg.x, y = neg.y, s = 80, c = 'blue', marker = '+')
        plt.legend()
        plt.show()

    # estimate on new data
    n_new = 100
    sample = generate_sample(n_new)
    sample['f'] = sample.apply(lambda row: 1 if row['y'] > line.get_y(row['x']) else -1, axis = 1)
    x = sample.as_matrix(columns = ['x', 'y'])
    x = np.insert(x, 0, 1, axis = 1)
    g = np.sign(np.dot(x, w))
    match = (np.sign(sample['f'].values) == np.sign(g.transpose())).transpose()
    p = np.nonzero(match == False)[0].size / float(n_new)

    return it, p 
    


if __name__ == '__main__':
    n = 100
    t = 1000 
    print "Sample size: ", n, "\n"
    print "Number of experiments: ", t, "\n"
    it_list = []
    p_list = []
    for k in range(0, t):
        it, p = pla(n, False)
        it_list.append(it)
        p_list.append(p)

    print "Average number of iterations required for PLA to converge: ", np.mean(it_list), "\n"
    print "Average probability of false predictions on new data: ", np.mean(p_list), "\n"
