import numpy as np
import os
import matplotlib.pyplot as plt
import sys
# first we need to get the os
path = os.getcwd()

def plot(file,p,labda,color,name,pname):
    y = 0.0
    for j in range(len(p)):
        for i in range (len(fileName)):
            file = os.path.join(path,name+"-"+pname[j]+"-"+fileName[i])
            print(file)
            data = np.load(file)
            data = data.flatten()
            print(data)
            iteration = np.arange(len(data))
            max_v = np.amax(data)
            if(max_v >= y):
                y = max_v
            #print(data.shape)
            print(np.amax(data))
            plt.plot(iteration,data, c=color[i], label=labda[i],linewidth=0.5)
        #plt.xlim([-1,len(data)])
        #print(y)
        #plt.ylim([0,y])
        plt.legend(loc='best')
        plt.title(name + " ("+p[j]+")")
        plt.savefig(name + " ("+p[j]+")", bbox_inches='tight')
        plt.clf()

if __name__ == "__main__":
    name = "Plot"
    try:
        name = sys.argv[1]
    except IndexError:
        name = "Plot"
    picture = ["Image3","Image10","Image11","Image12","Image26"]
    pictureName = ["original3","original10","original11","original12","original26"]
    fileName = ["lamda0.npy","lamda0.01.npy","lamda0.1.npy","lamda0.5.npy","lamda0.9.npy","lamda2.npy"]
    labda = ["lambda = 0.0","lambda = 0.01","lambda = 0.1","lambda = 0.5","lambda = 0.9","lambda = 2"]
    color = ['b','c','lime','blueviolet','gold','r']
    plot(fileName,picture,labda,color,name,pictureName)
