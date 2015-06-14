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
    x = []
    y = []
    for i in range(0, n):
        x.append(random.uniform(-1, 1))
        y.append(random.uniform(-1, 1))
        #sample = pd.concat([sample, pd.DataFrame({'x': [x], 'y': y})],
        #        ignore_index = True)
    sample = pd.DataFrame({'x': x, 'y': y})
    return sample 

# Linear regression for classification
# Generates data and runs it
# Input: n - sample size
#        show_plot - display a plot of data points, default = False
# Output: 
#       - E_in - fraction of missclassified points in-sample
#       - E_out - fraction of missclassified points out-of-sample
#       - it - number of iterations PLA takes to converge with initial weights
#       given by linear regression
def lr(n, show_plot = False):

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
    
    w = np.dot(np.dot(np.linalg.inv(np.dot(x.transpose(), x)), x.transpose()), sample.f)
    #g = np.sign(np.dot(w.transpose(), x))
    g = np.sign(np.dot(x, w.transpose()))
    
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

    match = (np.sign(sample['f'].values) == g)
    E_in = np.nonzero(match == False)[0].size / float(n)
    # estimate on new data
    n_new = 1000
    sample_new = generate_sample(n_new)
    sample_new['f'] = sample_new.apply(lambda row: 1 if row['y'] > line.get_y(row['x']) else -1, axis = 1)
    x_new = sample_new.as_matrix(columns = ['x', 'y'])
    x_new = np.insert(x_new, 0, 1, axis = 1)
    g_new = np.sign(np.dot(x_new, w))
    match = (np.sign(sample_new['f'].values) == g_new)
    E_out = np.nonzero(match == False)[0].size / float(n_new)

    # run PLA with w found by linear regression
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
    return (E_in, E_out, it)
    


if __name__ == '__main__':
    n = 10 
    t = 1000
    print "Sample size: ", n, "\n"
    print "Number of experiments: ", t, "\n"
    E_in_list = []
    E_out_list = []
    it_list = []
    for k in range(0, t):
        E_in, E_out, it = lr(n, False)
        E_in_list.append(E_in)
        E_out_list.append(E_out)
        it_list.append(it)

    print "Average E_in: ", np.mean(E_in_list), "\n"
    print "Average E_out: ", np.mean(E_out_list), "\n"
    print "Number of iterations of PLA: ", np.mean(it_list), "\n"
