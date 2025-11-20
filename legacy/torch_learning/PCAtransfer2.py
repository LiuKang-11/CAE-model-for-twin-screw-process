import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
columns = ['Element_name','1','2','3','4','5','6','7','8']
features=['1','2','3','4','5','6','7','8']
train =pd.read_csv('D:/PYTHON/PCA_2TARGETstd_test.csv',names=columns)
#print(train)

train['Element_name']=pd.Categorical(train['Element_name']) 
#print(train['Element_name'])
my_color=train['Element_name'].cat.codes

from sklearn.decomposition import PCA
pca = PCA(n_components=3)

# Separating out the features
x = train.loc[:, features]
# Separating out the target
y = train.loc[:,'Element_name']

X_pca_3 = pca.fit_transform(x)
pca3d=pd.DataFrame(pca.transform(x), columns=['PCA%i' % i for i in range(3)])
finalpca= pd.concat([pca3d,y],axis=1)
print(finalpca)




# assign x,y,z coordinates from PC1, PC2 & PC3
xs = X_pca_3.T[0]
ys = X_pca_3.T[1]
zs = X_pca_3.T[2]

# initialize scatter plot and label axes
classes=[
    'kL1-50-4', 'kL2-50-4', 'kL3-50-4', 'kR1-50-4', 'kR2-50-4', 'kR3-50-4',\
    'mL1c-50-15', 'mL1r-50-15', 'mL1t-50-15', 'mL2c-50-15', 'mL2r-50-15',\
    'mL2t-50-15', 'mL3c-50-15', 'mL3r-50-15', 'mL3t-50-15', 'mR1c-50-15',\
    'mR1r-50-15', 'mR1t-50-15', 'mR2c-50-15', 'mR2r-50-15', 'mR2t-50-15',\
    'mR3c-50-15', 'mR3r-50-15', 'mR3t-50-15', 'sL1-50-33', 'sL2-50-33',\
    'sR1-50-33', 'sR2-50-33', 'sR3-50-33'
]

colors=['brown','darkred','maroon','mistyrose','red','salmon','forestgreen','aquamarine','deepskyblue','limegreen',\
'turquoise','skyblue','green','lightseagreen','powderblue','black','grey','silver','dimgray','darkgray',\
'lightgray','dimgrey','darkgrey','lightgrey','mediumslateblue','mediumpurple','fuchsia','magenta','deeppink']
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
               ,c = color,s=10,marker='s')

ax.legend(classes,loc=10, bbox_to_anchor=(0.06, 0.85),
          fancybox=True, shadow=True, ncol=2, fontsize = 7)
plt.show()
