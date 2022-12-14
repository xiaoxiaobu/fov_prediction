from cProfile import label
import numpy as np
import os
from glob import glob
import numpy.matlib

import pandas as pd
import matplotlib.pyplot as plt



# Here is our main process for computing the area of 
# intersecting spherical caps. a1 and a2 are the straight line
# distance of the radius of each cap's respective base. r is the sphere's
# radius. theta is the angular separation between the caps (in radians).
def AreaOfCapsIntersection(r, theta):
    a = np.deg2rad(50)       #50 degree fov for calculate
    b = np.deg2rad(50)       #50 degree fov for calculate
    c = theta
    s = (a + b + c) / 2
    k = np.sqrt((np.sin(s - a) * np.sin(s - b) * np.sin(s - c) / np.sin(s)))
    A = 2 * np.arctan(k / np.sin(s - a))
    B = 2 * np.arctan(k / np.sin(s - b))
    C = 2 * np.arctan(k / np.sin(s - c))
    T1 = r**2 * (A + B + C - np.pi)
    T2 = T1
    S1 = 2 * B * r**2 * (1 - np.cos(a))
    S2 = 2 * A * r**2 * (1 - np.cos(b))
    return S1 + S2 - (T1 + T2)

# 
def cal_angle(v1,v2):
    v1 = np.array([v1[2],v1[0],v1[1]])
    v2 = np.array([v2[2],v2[0],v2[1]])
    assert((np.linalg.norm(v1)!=0) and (np.linalg.norm(v2)!=0))
    dot_product = np.dot(v1,v2)
    arccos = np.arccos(dot_product/((np.linalg.norm(v1))*(np.linalg.norm(v2))))
    return arccos

def Bucketize(data,interval):
    j = 0
    i = 0
    out = []
    while(i<data.shape[0]):
        tmp = data[i,:]
        out.append(tmp)
        j = j+1
        i = i+1
        while(i<data.shape[0] and data[i,0]<=(j-1)*interval):
            i = i+1
    return np.array(out)

def vec2Ang(vec):
    x,y,z = vec[2],vec[0],vec[1]
    azimuth = np.arctan2(y,x)
    elevation = np.arctan2(z,np.sqrt(x**2+y**2))
    r = np.sqrt(x**2+y**2+z**2)
    return azimuth,elevation,r

def haversine(lat1,lon1,lat2,lon2):
    dlat = lat2-lat1
    dlon = lon2-lon1
    a = np.sin(dlat/2.0)**2+np.cos(lat2)*np.cos(lat1)*np.sin(dlon/2.0)**2
    c = 2*np.arcsin(np.sqrt(a))
    return c/np.pi

def AreaofCap(theta,r):
    return 2*np.pi*r*(r-r*np.cos(theta))


path = './traces'

#sample rate
ff = 30
interval = 1/ff
starttime = 0
endtime = 70

directoryNames = os.listdir(path)

vids = [f for f in range(1,31) if(f<15 or f>16)]
print(vids)
viewers = dict()

idx = 0

for vid in vids: #????????????
    viewer = []
    for f in directoryNames: # ??????????????????
        file = os.path.join(path,f,f'{f}_{vid}.csv')
        if(os.path.isfile(file)==False):
            continue
        data = np.loadtxt(file,delimiter=',')
        i = 0
        for j in range(1,31): #???????????????????????????
            while(data[i,0]<1/29*j):
                i+=1
            # print(i)
            viewer.append(data[i,:])

            idx = idx+1

    viewer = np.array(viewer)
    print(viewer.shape)
    viewers[f'{vid}'] = viewer

all_video_area = dict()
all_video_dis = dict()

user = 0   #??????????????????????????????30???

for k in viewers:
    # print(k)
    video = viewers[k]
    
    area_between_user = []
    dis_between_user = []
    for frame in range(0,29):  #???????????????????????????
        vp1 = video[user*30+frame,5:8]
        vp2 = video[user*30+frame+1,5:8]
        theta = cal_angle(vp1,vp2)
        area = AreaOfCapsIntersection(1,theta)  # intersection area
        [lat1,lon1,r] = vec2Ang(vp1)
        [lat2,lon2,r] = vec2Ang(vp2)

        dis = haversine(lat1,lon1,lat2,lon2)

        area_between_user.append(area/AreaofCap(np.deg2rad(50),1))
        dis_between_user.append(dis)

    all_video_area[k] = np.array(area_between_user)
    all_video_dis[k] = np.array(dis_between_user)


for k in viewers:
    area_seq = all_video_area[k]
    dis_seq = all_video_dis[k]
    df = pd.DataFrame({'x':area_seq,'y':dis_seq})


    print(df.x.corr(df.y))



# x = [x for x in range(1,30)]
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twinx()

# print(area_between_user)
# print(dis_between_user)
# ax1.plot(x,area_between_user,'g-',label='area overlap')
# ax2.plot(x,dis_between_user,'r--',label='distance')

# ax1.set_xlabel('users')
# ax1.set_ylabel('area overlap')
# ax2.set_ylabel('distance')

# ax1.set_ylim(0,1)
# ax2.set_ylim(0.50,0)
# fig.legend(loc='upper right',bbox_to_anchor=(1,1),bbox_transform=ax1.transAxes)
# plt.grid(1)
# plt.show()








# for startTime in np.arange(0,1,1/30):
#     num = 0
#     for i in range(0,30):
#         data = viewer[i]
#         num = num+1
#         i = 0
#         while(i<data.shape[0] and data[i,0]<startTime):
#             i = i+1
#         [theta,phi,r] = vec2Ang(data[i,5:8])

        




# for vid in range(1,31):
#     if(vid==15 or vid == 16):
#         continue

#     i = 0

#     Traj = []
#     for f in directoryNames:
#         file = os.path.join(path,f,f'{f}_{vid}.csv')
#         if(os.path.isfile(file)==False):
#             continue
#         data = np.loadtxt(file,delimiter=',')
#         data = Bucketize(data,interval)

#         Traj.append(data[0:ff*60,5:8])
#         if(len(Traj[i][:,0])<ff*60):
#             temp = numpy.matlib.repmat(Traj[i][-1,:],ff*60-len(Traj[i][:,0]),1)
#             Traj[i] = np.vstack(Traj[i],temp)

#         i = i+1

#     n_users = len(Traj)
#     n_frames = ff*60

#     Traj_tmp = []
#     for i_u in range(n_users):
#         tmp = Traj[i_u]
#     print(Traj[0][0,:]) 


        









