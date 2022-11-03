import matplotlib.pyplot as plt

si = [84.336,119.81,53.487,48.125,60.663,48.35,92.732,67.346,84.675 \
,25.885,65.181,52.655,83.546,59.520 ,32.351,63.064,42.082,49.939,57.625] 

ti=[1.422,27.578,26.371,2.059,2.425,13.254,1.246,14.665,3.361,11.103,4.012,3.201,1.069,4.897,9.531,5.983,9.973,5.539,27.022]

txt = ['Abbottsford','Bar','Cockpit','Cows','Diner','DroneFlight','GazaFishermen',
'Fountain','MattSwift','Ocean','PlanEngeryBioLab','PortoRiverside','Sofa','Touvet','Turtle','TeatroRegioTorino',
'UnderwaterPark','Warship','Waterpark']

plt.scatter(si,ti)
print(len(txt))
for i in range(len(txt)):
    plt.annotate(txt[i],xy=(si[i],ti[i]),xytext=(si[i]+0.1,ti[i]+0.1),fontsize=8)


plt.ylim([0,30])
plt.xlim([0,125])
plt.xlabel('Spatial index',fontsize=12)
plt.ylabel('Temporal index',fontsize=12)
plt.grid(True)
plt.show()
