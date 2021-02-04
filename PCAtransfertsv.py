import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import csv
columns = ['Element_name','1','2','3','4','5','6','7','8','9','10','11','12']
features=['1','2','3','4','5','6','7','8','9','10','11','12']

for i  in range(10):
    name =pd.read_csv(('D:/python/350_2t100_1_writer/350_2t100_1_1/NN_result/0000{}/default/metadata.tsv'.format(i+0)),sep = "\t",names=columns[0:1])
    tensors =pd.read_csv(('D:/python/350_2t100_1_writer/350_2t100_1_1/NN_result/0000{}/default/tensors.tsv'.format(i+0)),sep = "\t",names=columns[1:])
    train= pd.concat( [name,tensors], axis=1 )

    name =pd.read_csv('D:/python/外插_temp100_5_100/NN_result/00100/default/metadata.tsv',sep = "\t",names=columns[0:1])
    tensors =pd.read_csv('D:/python/外插_temp100_5_100/NN_result/00100/default/tensors.tsv',sep = "\t",names=columns[1:])
    train2= pd.concat( [name,tensors], axis=1 )

    name =pd.read_csv('D:/python/外插_temp100_5_100/NN_result/00100/default/metadata.tsv',sep = "\t",names=columns[0:1])
    tensors =pd.read_csv('D:/python/外插_temp100_5_100/NN_result/00100/default/tensors.tsv',sep = "\t",names=columns[1:])
    train3= pd.concat( [name,tensors], axis=1 )




    #train3 =pd.read_csv('D:/PYTHON/rtd_SL31.csv',names=columns)

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

    colors=['pink','pink','pink','red','red','red','grey','grey','grey','grey',\
    'grey','grey','grey','grey','grey','grey','grey','grey','grey','grey',\
    'grey','grey','grey','grey','aquamarine','aquamarine','aquamarine','blue','blue','blue']
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
                ,c = color,s=40)
    '''
    for target, color in zip(classes,colors):
        indicesToKeep = finalpca2['Element_name'] == target

        ax.scatter(finalpca2.loc[indicesToKeep, 'PCA0']
                ,finalpca2.loc[indicesToKeep, 'PCA1']
                ,finalpca2.loc[indicesToKeep, 'PCA2']
                ,c = color,s=40,marker='^')

    for target, color in zip(classes,colors):
        indicesToKeep = finalpca3['Element_name'] == target

        ax.scatter(finalpca3.loc[indicesToKeep, 'PCA0']
                ,finalpca3.loc[indicesToKeep, 'PCA1']
                ,finalpca3.loc[indicesToKeep, 'PCA2']
                ,c = color,s=40,marker='s')
    '''
    ax.legend(classes,loc=10, bbox_to_anchor=(0.06, 0.85),
            fancybox=True, shadow=True, ncol=2, fontsize = 7)
    fig.savefig(('D:/python/gif_test/350_{}.png'.format(i+0))) 
    plt.show()


