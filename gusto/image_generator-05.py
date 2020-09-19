# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 20:38:51 2020

@author: sijas
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.stats import truncnorm
import xml.dom.minidom
import xml.etree.ElementTree as gfg
import cv2
from optparse import OptionParser

def lap(u,dx=1.0):
    return (+1*np.roll(u,+1,axis=0)
            +1*np.roll(u,-1,axis=0)
            +1*np.roll(u,+1,axis=1)
            +1*np.roll(u,-1,axis=1)
            -4*u) / dx**2

def noise(n,total_time, scale):
    ''' 
    grid size
    n = 512
    N=(n,n)
    # diffusion coefficient
    D = 1.0
    # spatial dimensions
    L = 100.0
    dx = L / n
    x = np.arange(0,L,dx)
    # time
    t = 0.0
    total_time = 0.5'''
    N=(n,n)
    t = 0.0
    D = 1.0
    scale = 100.0
    dx = scale / n
    dt = 0.2 * 0.5 * dx**2 / D
    f = lambda u: u - u**3

    u = 2*np.random.random(N)-1.0
    while t<total_time:
        t += dt
        # u = (u+ np.random.random(N)*u*0.02)/(1+0.02)
        u = u + dt * (f(u) + D * lap(u,dx) ) 
    return u

parser = OptionParser()
parser.add_option("-f", "--file", dest="filename_base",
                  help="write report to FILE", metavar="FILE")
parser.add_option("-p", "--path",
                  dest="image_path", default='./',
                  help="path to samve image")

parser.add_option("-n", "--number",
                  dest="number_of_images", default=10,
                  help="how many images do you want")

parser.add_option("-d", "--numberd",
                  dest="number_of_defects", default=10,
                  help="how many defects do you want")

parser.add_option("-b", "--backgroud",
                  action ="store_false",
                  dest="backgroud", default=True,
                  help="do you need backound noise")

(options, args) = parser.parse_args()
option_dict = vars(options)
file_base = option_dict['filename_base']
image_path = option_dict['image_path']
num_images = int(option_dict['number_of_images'])
num_defects = int(option_dict['number_of_defects'])
noisy_backraund = bool(option_dict['backgroud'])
# print(file_base,image_path,num_images,num_defects,noisy_backraund)

def append_object(root,label="NAME",pose="Unspecified",truncated="0",dificult="0",xmin="0",ymin="0",xmax="0",ymax="0"):
    m4 = gfg.Element("object") 
    root.append(m4)
    c1 = gfg.SubElement(m4, "name") 
    c1.text = label
    # print(label)
    c2 = gfg.SubElement(m4, "pose") 
    c2.text = pose
    c3 = gfg.SubElement(m4, "truncated") 
    c3.text = truncated
    c4 = gfg.SubElement(m4, "dificult") 
    c4.text = dificult
    c5 = gfg.Element("bndbox") 
    m4.append(c5)

    d1 = gfg.SubElement(c5, "xmin") 
    d1.text = xmin
    d2 = gfg.SubElement(c5, "ymin") 
    d2.text = ymin
    d3 = gfg.SubElement(c5, "xmax") 
    d3.text = xmax
    d4 = gfg.SubElement(c5, "ymax") 
    d4.text = ymax

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
    def __init__(self,thickness=0.2,resolution=100,r=1,center=(0,0),color=0.8):
        cmap = cm.get_cmap('afmhot')
        rgb=cmap(color)
        self.center=center
        self.color=rgb
        self.fraction = 1
        self.resolution = resolution
        self.thickness = thickness
        self.theta = np.linspace(0, 2*np.pi*self.fraction, self.resolution)
        self.r = r
        self.x1 = self.r*np.cos(self.theta)+center[0] 
        self.y1 = self.r*np.sin(self.theta)+center[1]

        self.x2 = (self.r-self.thickness)*np.cos(self.theta) +center[0]
        self.y2 = (self.r-self.thickness)*np.sin(self.theta) +center[1]
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
        self.x1, self.y1 = rotate_2D(self.x1-self.center[0],self.y1-self.center[1],angle=angle)
        self.x2, self.y2 = rotate_2D(self.x2-self.center[0],self.y2-self.center[1],angle=angle)
        self.x1, self.y1 =self.x1+self.center[0], self.y1+self.center[1]
        self.x2, self.y2 =self.x2+self.center[0], self.y2+self.center[1]


    def cut(self,fraction=1/2):
        angle = fraction*2*np.pi
        self.fraction=fraction
#         split = np.where( np.arctan2(self.y1,self.x1)+np.pi < angle ) # kazkaip cia gustai per sudetingai sugalvojai
        split= np.where(self.theta<angle) #pakeiciau nes kitai neveike kai defekto centras ne (0,0)
        self.x1, self.y1 = self.x1[split], self.y1[split]
        self.x2, self.y2 = self.x2[split], self.y2[split]
     
    # need some work
    def fill(self,axs,cutoff=0.2):
        if self.fraction<1:
            axs.set_aspect(1)
#             right_bound = [ print(i,np.abs(self.x1[i]-self.x1[i-1]), np.abs(self.x1[i-1]-self.x1[i-2])) for i,x in enumerate(self.x1) if np.abs(self.x1[i]                 -self.x1[i-1]) > cutoff * np.abs(self.x1[i-1]-self.x1[i-2]) ]
#             right_bound = [ i for i,x in enumerate(self.x1) if np.abs(self.x1[i]-self.x1[i-1]) > 0.1 + np.abs(self.x1[i-1]-self.x1[i-2]) ]

#             left_bound = int(right_bound[0]-1)
            axs.plot(self.x1,self.y1,color=self.color)
            axs.plot(self.x2,self.y2,color=self.color)
            pair_r = ([self.x1[0],self.x2[0]],[self.y1[0],self.y2[0]])#([self.x1[right_bound],self.x2[right_bound]],[self.y1[right_bound],self.y2[right_bound]])
            # plt.plot(pair[0],pair[1])
            pair_l = ([self.x1[-1],self.x2[-1]],[self.y1[-1],self.y2[-1]])#([self.x1[left_bound],self.x2[left_bound]],[self.y1[left_bound],self.y2[left_bound]])
            # plt.plot(pair[0],pair[1])
            r_l=np.sqrt((pair_l[0][0]-np.average(pair_l[0]))**2+(pair_l[1][0]-np.average(pair_l[1]))**2)
            r_r=np.sqrt((pair_r[0][0]-np.average(pair_r[0]))**2+(pair_r[1][0]-np.average(pair_r[1]))**2)

            Circle1=plt.Circle((np.average(pair_l[0]),np.average(pair_l[1])),r_l,color=self.color)
            Circle2=plt.Circle((np.average(pair_r[0]),np.average(pair_r[1])),r_r,color=self.color)
            axs.add_artist(Circle1)
            axs.add_artist(Circle2)
            # plt.show()

            xs=np.array([self.x1,self.x2])
            ys=np.array([self.y1,self.y2])
            xs[0,:] = xs[0,::-1]
            ys[0,:] = ys[0,::-1]
            axs.fill(np.ravel(xs), np.ravel(ys),color=self.color, edgecolor=self.color)
#             axs.scatter(self.x1[0],self.y1[0],color='black')
#             axs.scatter(self.x1[-1],self.y1[-1],color='orange')
#             axs.scatter(self.x2[0],self.y2[0],color='black')
#             axs.scatter(self.x2[-1],self.y2[-1],color='orange')
        else:
            # print('Full circle')
            axs.plot(self.x1, self.y1,color=self.color)
            axs.plot(self.x2, self.y2,color=self.color)
            axs.set_aspect(1)
            xs=np.array([self.x1,self.x2])
            ys=np.array([self.y1,self.y2])
            xs[0,:] = xs[0,::-1]
            ys[0,:] = ys[0,::-1]
            axs.fill(np.ravel(xs), np.ravel(ys),color=self.color, edgecolor=self.color)
#             plt.grid(linestyle='--')
#             plt.show()
   
    def get_csv(self,path='contour.csv'):
        self.x = np.concatenate((self.x1,self.x2))
        self.y = np.concatenate((self.y1,self.y2))
        d = {'x': self.x, 'y': self.y}
        df = pd.DataFrame(data=d)
        df.to_csv(path, index=False,sep=" ")

class label_generator(defect):
    def __init__(self,Nr=10,scale=2,radius=30,thickness=10,defect_color=0.8):
        '''Nr - number of defects, scale - size of immage in micrometer, radius - defect radius in nm, thickness in nm '''
        self.Nr=Nr
        self.scale=scale
        self.defect_color=defect_color
        def scale_radius_and_thickness(scale,radius,thickness): 
            '''scale = [micor m], radius=[ nm ]'''
            r_visable=radius/1000
            visable_thickness=thickness/1000/scale
            return r_visable,visable_thickness
        r,th = scale_radius_and_thickness(scale,radius,thickness)
        self.radius=r
        self.thickness=th
    def color(self,color_map_string='afmhot'):
        cmap = cm.get_cmap(color_map_string)
        rgb=cmap(self.defect_color)
        return rgb
    def get_coordinates(self):
        x=(0.04*self.scale+np.random.random_sample(self.Nr)*self.scale)*0.9
        y=(0.04*self.scale+np.random.random_sample(self.Nr)*self.scale)*0.9
        pairs=np.transpose(np.array([x,y]))
        return pairs
    def get_radii(self,dev=0.1):
        array=np.random.normal(loc=self.radius,scale=dev*self.radius,size=self.Nr)
        return(array)
    def get_thicknesses(self,sd=0.5):
        #array=np.random.normal(loc=self.thickness,scale=dev*self.thickness,size=self.Nr)
        array = np.zeros(self.Nr)
        for i,k in enumerate(array):
            array[i]=get_truncated_normal(mean=self.thickness, sd=0.5, low=0.8*self.thickness, upp=1.5*self.thickness)
        return(array)
    def get_cut_angle(self):
        array=(1-0.3)*np.random.random_sample(self.Nr)+0.3
        return array
    def get_box(self):
        self.box=[0,scale,0,scale]
        return self.box
    def get_depth(self,dev=0.05,color_map_string='afmhot'):
        # cmap = cm.get_cmap(color_map_string)
        colors = np.random.normal(loc=self.defect_color,scale=dev*self.defect_color,size=self.Nr)
        # rgbs = np.transpose(np.delete(np.transpose(rgbs),3,0))
        return colors
    def get_sigma_lumpyness(self,dev=0.1):
        #sigma = get_truncated_normal(mean=1, sd=0.5, low=0.01, upp=10)
        #lumpyness = get_truncated_normal(mean=-1, sd=0.2, low=-1, upp=1)
        #array_s=np.random.normal(loc=sigma,scale=dev*sigma,size=self.Nr)
        array_s = np.zeros(self.Nr)
        array_l = np.zeros(self.Nr)
        for i,k in enumerate(array_s):
            array_s[i]=get_truncated_normal(mean=1, sd=0.5, low=0.01, upp=10)
            array_l[i]=get_truncated_normal(mean=-1, sd=0.2, low=-1, upp=1)

        #array_s=np.random.normal(loc=sigma,scale=dev*sigma,size=self.Nr)
        #array_l=np.random.normal(loc=lumpyness,scale=abs(dev*lumpyness),size=self.Nr)
        return array_s,array_l

    def add_shit(self,axs,number_of_shits=1,size_defoult=0.08,defect_color=0.95):
        print('number_of_shits:',number_of_shits)
        self.box_shit=np.zeros((number_of_shits,4))
        if number_of_shits==0:
            return 'nothing'
        else:
            n=15
            for i in range(number_of_shits):
                size=np.random.random()*size_defoult
                xs=[]
                ys=[]
                coordinate_x=(0.04*self.scale+np.random.random()*self.scale)*0.9
                coordinate_y=(0.04*self.scale+np.random.random()*self.scale)*0.9
                x=np.random.random(n)*size+size+coordinate_x
                y=np.random.random(n)*size+size+coordinate_y
                r=np.random.random(n)*size
                theta=np.linspace(0,np.pi*2,n)
                x2=r*np.sin(theta)+size+coordinate_x
                y2=r*np.cos(theta)+size+coordinate_y
                cmap = cm.get_cmap('afmhot')
                rgb=cmap(defect_color)
                axs.scatter(x,y,s=350/0.08*size,color=rgb,zorder=10)
                axs.scatter(x2,y2,s=350/0.08*size,color=rgb,zorder=10)
                xs.append(x)
                ys.append(y)
                xs.append(x2)
                ys.append(y2)
                xmin=np.min(xs)
                xmax=np.max(xs)
                ymin=np.min(ys)
                ymax=np.max(ys)
                padding=8
                self.box_shit[i]=np.array([np.int(xmin/self.scale*512-padding),512-np.int(ymin/self.scale*512-padding),np.int(xmax/self.scale*512+padding),512-np.int(ymax/self.scale*512+padding)],dtype=int)

                # axs.plot((np.min(xs)/1.03,np.max(xs)*1.03),(np.min(ys)/1.03,np.max(ys)*1.03),zorder=11)

    def generate_immage_with_boxes(self,axs,noisy_backraund=True):
        if noisy_backraund==True:
            time=0.1+np.random.random()*2
            u=noise(512,time,self.scale)-0.3-np.random.random()*0.5
            print(time,-0.3-np.random.random()*0.25)
            p = axs.imshow(u,cmap='afmhot', vmin=-1.0, vmax=1.0,extent=[0,self.scale,0,self.scale])
        self.radii=self.get_radii()
        self.thicknesses=self.get_thicknesses()
        self.cut_angles=self.get_cut_angle()
        self.sigma,self.lumpyness = self.get_sigma_lumpyness()
        self.colors=self.get_depth()
        self.centers=self.get_coordinates()
        self.box=np.zeros((self.Nr,4))
        self.Nr_shit=np.random.randint(low=0,high=5)
        self.add_shit(axs,number_of_shits=self.Nr_shit)

        for i in range(self.Nr):
            r=self.radii[i]
            thickness=self.thicknesses[i]
            color=self.colors[i]
            center=self.centers[i]
            #d = defect(r=self.radii[i],center=self.centers[i],thickness=self.thickness,color=self.colors[i])
            d = defect(r=r,center=self.centers[i],thickness=thickness,color=self.colors[i])
            d.add_bump()#sigma=self.sigma[i],lumpyness=self.lumpyness[i])
            d.cut(self.cut_angles[i])
            # print("cut_angle:",self.cut_angles[i])
            d.rotate(angle=np.random.random()*np.pi*2)
            d.fill(axs)

            # plt.axis('off')
            axs.spines['left'].set_visible(False)
            axs.spines['right'].set_visible(False)
            axs.spines['bottom'].set_visible(False)
            axs.spines['top'].set_visible(False)
            axs.tick_params(which='both',bottom=False,left=False,labelbottom=False,labelleft=False)
            xmin=np.min(np.array([d.x1,d.x2]))
            xmax=np.max(np.array([d.x1,d.x2]))
            ymin=np.min(np.array([d.y1,d.y2]))
            ymax=np.max(np.array([d.y1,d.y2]))
            padding = 4
            self.box[i]=np.array([np.int(xmin/self.scale*512-padding),512-np.int(ymin/self.scale*512-padding),np.int(xmax/self.scale*512+padding),512-np.int(ymax/self.scale*512+padding)],dtype=int)
            #print(np.round(self.box[i]))
            #self.box[i]=np.array([xmin,ymin,xmax,ymax])
            #plt.plot([xmin,xmin,xmax,xmax,xmin],[ymin,ymax,ymax,ymin,ymin],color='black')


    def get_labels(self):
        radii=self.get_radii()
        thicknesses=self.get_thicknesses()
        #cut_angles=self.get_cut_angle()
        cut_angles=self.cut_angles
        cut_angles_4=np.abs(cut_angles-1/4)
        cut_angles_3=np.abs(cut_angles-0.5)
        cut_angles_2=np.abs(cut_angles-3/4)
        cut_angles_1=np.abs(cut_angles-1)
        cut_angles_global=np.array([cut_angles_4,cut_angles_3,cut_angles_2,cut_angles_1])
        cut_angles_global_t=np.transpose(cut_angles_global)
        fourth=np.where(cut_angles_global[0]==cut_angles_global_t.min(1))
        half=np.where(cut_angles_global[1]==cut_angles_global_t.min(1))
        three_forths=np.where(cut_angles_global[2]==cut_angles_global_t.min(1))
        full=np.where(cut_angles_global[3]==cut_angles_global_t.min(1))
        cut_angle_labels=np.zeros(self.Nr)
        cut_angle_labels[fourth]=0.25
        cut_angle_labels[half]=0.5
        cut_angle_labels[three_forths]=0.75
        cut_angle_labels[full]=1
        self.fraction=cut_angle_labels
        # print(self.fraction)
        # print(cut_angles_global_t)
        dict = {'radii': radii, 'thicknesses': thicknesses, 'fraction': cut_angle_labels,'box' : self.box}
        
        return dict
    def get_labels2(self,image_name='test',path='./'):
        fileName= image_name+".xml"
        root = gfg.Element("annotation") 
            
        m1 = gfg.Element("folder") 
        m1.text = path
        root.append(m1) 
        m2 = gfg.Element("filename") 
        m2.text = image_name+".jpg"
        root.append(m2)

        m3 = gfg.Element("path") 
        m3.text = path+image_name+".jpg"
        root.append(m3)

        m3 = gfg.Element("source") 
        root.append(m3)

        b1 = gfg.SubElement(m3, "database") 
        b1.text = "Unknown"

        m4 = gfg.Element("size")
        root.append(m4)

        b2 = gfg.SubElement(m4, "with") 
        b2.text = "512"
        b3 = gfg.SubElement(m4, "height") 
        b3.text = "512"
        b4 = gfg.SubElement(m4, "depth") 
        b4.text = "3"

        m5 = gfg.Element("segmented")
        m5.text = "0"
        root.append(m5)

        for i in range(self.Nr):
            #append_object(root,label=str(self.fraction[i]),xmin=str(np.int(self.box[i][0])),ymin=str(np.int(self.box[i][1])),xmax=str(np.int(self.box[i][2])),ymax=str(np.int(self.box[i][3])))
            append_object(root,label=str(self.fraction[i]),xmin=str(np.int(self.box[i][0])),ymin=str(np.int(self.box[i][3])),xmax=str(np.int(self.box[i][2])),ymax=str(np.int(self.box[i][1])))
            # print(np.int(self.box[i][0]))
        for i in range(self.Nr_shit):
            append_object(root,label=str(-1),xmin=str(np.int(self.box_shit[i][0])),ymin=str(np.int(self.box_shit[i][3])),xmax=str(np.int(self.box_shit[i][2])),ymax=str(np.int(self.box_shit[i][1])))
        tree = gfg.ElementTree(root) 

        with open (path+"/"+fileName, "wb") as files : 
            # print(path+"/"+fileName)
            tree.write(files) 


        dom = xml.dom.minidom.parse(path+"/"+fileName) # or xml.dom.minidom.parseString(xml_string)
        pretty_xml_as_string = dom.toprettyxml()
        # print(pretty_xml_as_string)

cmap = cm.get_cmap('afmhot')
x=np.linspace(0.2,1,10)
rgb=cmap(x)
#np.random.randint(low=0,high=3)


for i in range(num_images):
    image_name = file_base+"_"+str(i)

    r=label_generator(Nr=num_defects,thickness=20,defect_color=0.65+np.random.random()*0.1)
    fig ,axs = plt.subplots(1,1,figsize=(10,10))
    axs.set_xlim((0,2))
    axs.set_ylim((0,2))
    r.generate_immage_with_boxes(axs,noisy_backraund=noisy_backraund)
    # add_shit(axs,number_of_shits=np.random.randint(low=0,high=3))
    # add_shit(axs,number_of_shits=np.random.randint(low=0,high=6),size=0.01)
    r.get_labels()

    axs.set_facecolor(rgb[0])
    #plt.show()
    # print(image_name)
    fig.savefig(image_path+"/"+image_name+".jpg",dpi=66.5,pad_inches=0,bbox_inches='tight')
    plt.close()
    img = cv2.imread(image_path+"/"+image_name+".jpg")
    blur = cv2.blur(img,(3+np.random.randint(low=0,high=2),3+np.random.randint(low=0,high=2)))
    cv2.imwrite(image_path+"/"+image_name+".jpg",blur)
    a=r.get_labels2(path=image_path,image_name=image_name)
#print(a)

#with open('test.png', 'w') as outfile:
#    fig.canvas.print_png(outfile)
