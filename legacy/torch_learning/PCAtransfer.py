import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
columns = ['Element_name','1','2','3','4','5','6','7','8']
features=['1','2','3','4','5','6','7','8']
train =pd.read_csv('D:/PYTHON/rtd_training.csv',names=columns)
train2 =pd.read_csv('D:/PYTHON/rtd_testing.csv',names=columns)
train3 =pd.read_csv('D:/PYTHON/rtd_ex2.csv',names=columns)

#print(train)

train['Element_name']=pd.Categorical(train['Element_name']) 
train2['Element_name']=pd.Categorical(train2['Element_name']) 
train3['Element_name']=pd.Categorical(train3['Element_name']) 

#print(train['Element_name'])

from sklearn.decomposition import PCA
pca = PCA(n_components=3)

# Separating out the features
x = train.loc[:, features]
x2 = train2.loc[:, features]
x3 = train3.loc[:, features]

# Separating out the target
y = train.loc[:,'Element_name']
y2 = train2.loc[:,'Element_name']
y3 = train3.loc[:,'Element_name']

X_pca_3 = pca.fit_transform(x)
pca3d=pd.DataFrame(pca.transform(x), columns=['PCA%i' % i for i in range(3)])
finalpca= pd.concat([pca3d,y],axis=1)
print(finalpca)
X2_pca_3 = pca.fit_transform(x2)
pca3d2=pd.DataFrame(pca.transform(x2), columns=['PCA%i' % i for i in range(3)])
finalpca2= pd.concat([pca3d2,y2],axis=1)
X3_pca_3 = pca.fit_transform(x3)
pca3d3=pd.DataFrame(pca.transform(x3), columns=['PCA%i' % i for i in range(3)])
finalpca3= pd.concat([pca3d3,y3],axis=1)
print(finalpca3)

# initialize scatter plot and label axes
classes=[
    'kL1-50-4', 'kL2-50-4', 'kL3-50-4', 'kR1-50-4', 'kR2-50-4', 'kR3-50-4',\
    'mL1c-50-15', 'mL1r-50-15', 'mL1t-50-15', 'mL2c-50-15', 'mL2r-50-15',\
    'mL2t-50-15', 'mL3c-50-15', 'mL3r-50-15', 'mL3t-50-15', 'mR1c-50-15',\
    'mR1r-50-15', 'mR1t-50-15', 'mR2c-50-15', 'mR2r-50-15', 'mR2t-50-15',\
    'mR3c-50-15', 'mR3r-50-15', 'mR3t-50-15', 'sL1-50-33', 'sL2-50-33',\
    'sL3-50-33', 'sR1-50-33', 'sR2-50-33', 'sR3-50-33'
]

colors=['brown','darkred','maroon','mistyrose','red','salmon','forestgreen','aquamarine','deepskyblue','limegreen',\
'turquoise','skyblue','green','lightseagreen','powderblue','black','grey','silver','dimgray','darkgray',\
'lightgray','dimgrey','darkgrey','lightgrey','mediumslateblue','mediumpurple','rebeccapurple','fuchsia','magenta','deeppink']
#print(colors)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

for target, color in zip(classes,colors):
    indicesToKeep = finalpca['Element_name'] == target

    ax.scatter(finalpca.loc[indicesToKeep, 'PCA0']
               ,finalpca.loc[indicesToKeep, 'PCA1']
               ,finalpca.loc[indicesToKeep, 'PCA2']
               ,c = color,s=60,edgecolor='green')

for target, color in zip(classes,colors):
    indicesToKeep = finalpca2['Element_name'] == target

    ax.scatter(finalpca2.loc[indicesToKeep, 'PCA0']
               ,finalpca2.loc[indicesToKeep, 'PCA1']
               ,finalpca2.loc[indicesToKeep, 'PCA2']
               ,c = color,s=60,marker='^',edgecolor='blue')

for target, color in zip(classes,colors):
    indicesToKeep = finalpca3['Element_name'] == target

    ax.scatter(finalpca3.loc[indicesToKeep, 'PCA0']
               ,finalpca3.loc[indicesToKeep, 'PCA1']
               ,finalpca3.loc[indicesToKeep, 'PCA2']
               ,c = color,s=60,marker='s',edgecolor='red')

ax.legend(classes,loc=10, bbox_to_anchor=(0.06, 0.85),
          fancybox=True, shadow=True, ncol=2, fontsize = 7)

plt.show()

