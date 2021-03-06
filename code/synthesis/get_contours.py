import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd).rvs()

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def rotate_2D(x,y,angle=np.pi/4):
    xy = np.array([x,y])
    c ,s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    rot = np.dot(R,xy)
    return rot[0],rot[1]

class defect:
    # constructor
    def __init__(self,thickness=0.2,resolution=100,r=1):
        self.fraction = 1
        self.resolution = resolution
        self.thickness = thickness
        self.theta = np.linspace(0, 2*np.pi*self.fraction, self.resolution)
        self.r = r
        self.x1 = self.r*np.cos(self.theta) 
        self.y1 = self.r*np.sin(self.theta)

        self.x2 = (self.r-self.thickness)*np.cos(theta) 
        self.y2 = (self.r-self.thickness)*np.sin(theta)
    def plot(self):
        fig, ax = plt.subplots(1)
        ax.scatter(self.x1, self.y1)
        ax.scatter(self.x2, self.y2)
        ax.set_aspect(1)
        plt.grid(linestyle='--')
        plt.show()
   
    def add_bump(self,mu=0,sigma=2,lumpyness=-1):
        #top = int(self.resolution/2) # apply only to the first half
        top_1 = np.where(self.y1 >= 0)
        top_2 = np.where(self.y2 >= 0)
        #plt.plot(self.x1[top_1],self.y1[top_1])
        
        gauss_y1 = gaussian(self.x1[top_1],mu,sigma)
        gauss_y2 = gaussian(self.x2[top_2],mu,sigma)
        
        #plt.plot(x1[top_1],gauss_y1-np.min(gauss_y1))
        #plt.plot(x2[top_2],- lumpyness *(gauss_y2-np.min(gauss_y2)))
        
        self.y1[top_1] = self.y1[top_1] + gauss_y1-np.min(gauss_y1)
        self.y2[top_2] = self.y2[top_2] - lumpyness *(gauss_y2-np.min(gauss_y2)) #+ np.max(y2)
        
    
    def rotate(self,angle=np.pi/4):
        self.x1, self.y1 = rotate_2D(self.x1,self.y1,angle=angle)
        self.x2, self.y2 = rotate_2D(self.x2,self.y2,angle=angle)

    def cut(self,fraction=1/2):
        angle = fraction*2*np.pi
        
        split = np.where( np.arctan2(self.y1,self.x1)+np.pi < angle )
        self.x1, self.y1 = self.x1[split], self.y1[split]
        self.x2, self.y2 = self.x2[split], self.y2[split]
     
    # need some work
    def fill(self,cutoff=0.2):
        right_bound = [ print(i,np.abs(self.x1[i]-self.x1[i-1]), np.abs(self.x1[i-1]-self.x1[i-2])) for i,x in enumerate(self.x1) if np.abs(self.x1[i]                 -self.x1[i-1]) > cutoff * np.abs(self.x1[i-1]-self.x1[i-2]) ]
        right_bound = [ i for i,x in enumerate(self.x1) if np.abs(self.x1[i]-self.x1[i-1]) > 0.1 + np.abs(self.x1[i-1]-self.x1[i-2]) ]
        print(right_bound)
        left_bound = int(right_bound[0]-1)
        plt.plot(self.x1,self.y1)
        plt.plot(self.x2,self.y2)
        pair = ([self.x1[right_bound],self.x2[right_bound]],[self.y1[right_bound],self.y2[right_bound]])
        plt.plot(pair[0],pair[1])
        pair = ([self.x1[left_bound],self.x2[left_bound]],[self.y1[left_bound],self.y2[left_bound]])
        plt.plot(pair[0],pair[1])
    
    # not too usefull
    def get_csv(self,path='contour.csv'):
        self.x = np.concatenate((self.x1,self.x2))
        self.y = np.concatenate((self.y1,self.y2))
        d = {'x': self.x, 'y': self.y}
        df = pd.DataFrame(data=d)
        df.to_csv(path, index=False,sep=" ")
        #print(len(self.x))
        #print(df)
        
        
  # Geting random shape
d = defect()
for i in range(10):
    sigma = get_truncated_normal(mean=3, sd=1, low=0.01, upp=1000)
    print(sigma)
    lumpyness = get_truncated_normal(mean=-1, sd=1, low=-1, upp=5)
    angle = np.random.normal(np.pi/4,4)
    d.add_bump(sigma=sigma,lumpyness=lumpyness)
    d.rotate(angle=angle)
    d.plot()
d.plot()
d.cut(1/4)
d.plot()


d.fill()
