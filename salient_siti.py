import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# si = [84.336,119.81,53.487,48.125,60.663,48.35,92.732,67.346,84.675 \
# ,25.885,65.181,52.655,83.546,59.520 ,32.351,63.064,42.082,49.939,57.625] 

# ti=[1.422,27.578,26.371,2.059,2.425,13.254,1.246,14.665,3.361,11.103,4.012,3.201,1.069,4.897,9.531,5.983,9.973,5.539,27.022]

# txt = ['Abbottsford','Bar','Cockpit','Cows','Diner','DroneFlight','GazaFishermen',
# 'Fountain','MattSwift','Ocean','PlanEngeryBioLab','PortoRiverside','Sofa','Touvet','Turtle','TeatroRegioTorino',
# 'UnderwaterPark','Warship','Waterpark']

# plt.scatter(si,ti)
# print(len(txt))
# for i in range(len(txt)):
#     plt.annotate(txt[i],xy=(si[i],ti[i]),xytext=(si[i]+0.1,ti[i]+0.1),fontsize=8)


# plt.ylim([0,30])
# plt.xlim([0,125])
# plt.xlabel('Spatial index',fontsize=12)
# plt.ylabel('Temporal index',fontsize=12)
# plt.grid(True)
# plt.show()


val = [ -0.999549722,-0.999982831,-0.999991906,-0.999875232,-0.999990434,-0.999990018,-0.999973726,-0.997937495,-0.999857691,-0.9988602,\
    -0.999823113,-0.999932872,-0.999983922,-0.989181444,-0.999231169,-0.999828828,-0.99904227,-0.997099749,-0.999818089,-0.999697329,-0.999802791,\
-0.999960689,-0.999607557,-0.999947389,-0.99992002,-0.999966143,-0.999998236,-0.999962151]
video = [f for f in range(1,31) if(f<15 or f >16)]

plt.bar(video,val)

plt.ylim([0,-1.2])
plt.xlabel('视频',fontsize=12)
plt.ylabel('相关系数')
plt.show()