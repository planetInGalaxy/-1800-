# %% [markdown]
# # 导入数据

# %%
import pandas as pd
import numpy as np

# %%
data = pd.read_excel('./in/data-20230322.xlsx')
data

# %%
y = data.iloc[:,4]
x = data.iloc[:,5:]
print(x)
print(y)

# %% [markdown]
# # 数据划分

# %%
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size=0.3,random_state=120)

# %%
pd.value_counts(Ytrain)

# %%
pd.value_counts(Ytest)

# %%
Xtrain.to_excel("./out/Xtrain0.xlsx")
Xtest.to_excel("./out/Xtest0.xlsx")
pd.DataFrame(Ytrain).to_excel("./out/Ytrain0.xlsx")
pd.DataFrame(Ytest).to_excel("./out/Ytest0.xlsx")

# %%
#恢复索引
for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])
#恢复索引
for i in [Ytrain, Ytest]:
    i.index = range(i.shape[0])

# %%
Xtrain.to_excel("./out/Xtrain.xlsx")
Xtest.to_excel("./out/Xtest.xlsx")
Ytrain.to_excel("./out/Ytrain.xlsx")
Ytest.to_excel("./out/Ytest.xlsx")

# %% [markdown]
# # 建模输入

# %%
Xtrainmodel = pd.read_excel("./in/Xtrainmodel.xlsx")

# %%
Xtestmodel = pd.read_excel("./in/Xtestmodel.xlsx")

# %%
Xtrainmodel = Xtrainmodel.fillna(0)
Xtestmodel = Xtestmodel.fillna(0)

# %%
Xtrainmodel

# %%
Xtestmodel

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,roc_auc_score,recall_score,precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# %% [markdown]
# # 随机森林分类&调参

# %%
Ytrain.shape

# %%

param_test3={'max_depth':range(1,20,1)}
grid_search_2=GridSearchCV(estimator=RandomForestClassifier(n_estimators=200, random_state=68),param_grid=param_test3,scoring='accuracy',cv=10)
grid_search_2.fit(Xtrainmodel,Ytrain)
print(grid_search_2.best_params_)
print(grid_search_2.best_score_)

# %%
from sklearn.model_selection import cross_val_score
rfc = RandomForestClassifier(n_estimators=180,random_state=85)
rfc_c = rfc.fit(Xtrainmodel, Ytrain)
rfcy_pred = rfc_c.predict(Xtestmodel)

# %%
from sklearn import metrics
# 计算测试集的R2、MSE值
print(metrics.accuracy_score(Ytest, rfcy_pred))
print(metrics.f1_score(Ytest, rfcy_pred))
print(metrics.roc_auc_score(Ytest, rfcy_pred))
print(metrics.matthews_corrcoef(Ytest, rfcy_pred))

# %%
conf_mattest = confusion_matrix(Ytest, rfcy_pred)

# %%
conf_mattest

# %%
print(cross_val_score(rfc_c,Xtrainmodel,Ytrain,cv=10,scoring="accuracy").mean())
print(cross_val_score(rfc_c,Xtrainmodel,Ytrain,cv=10,scoring="f1").mean())
print(cross_val_score(rfc_c,Xtrainmodel,Ytrain,cv=10,scoring="roc_auc").mean())

# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
ycm_pred = cross_val_predict(rfc_c,Xtrainmodel,Ytrain,cv=10)
conf_mat = confusion_matrix(Ytrain, ycm_pred)

# %%
conf_mat 

# %%
from sklearn.model_selection import GridSearchCV
param_test1={'n_estimators':range(1,300,10)}
grid_search=GridSearchCV(estimator=RandomForestClassifier(random_state=85),param_grid=param_test1,scoring='accuracy',cv=10)
grid_search.fit(Xtrainmodel,Ytrain)
print(grid_search.best_params_)
print(grid_search.best_score_)

# %%
param_test3={'max_depth':range(1,20,1)}
grid_search_2=GridSearchCV(estimator=RandomForestClassifier(n_estimators=180, random_state=85),param_grid=param_test3,scoring='accuracy',cv=10)
grid_search_2.fit(Xtrainmodel,Ytrain)
print(grid_search_2.best_params_)
print(grid_search_2.best_score_)

# %% [markdown]
# # 支持向量机分类&调参

# %%
Xtrainmodel.shape

# %%
Xtestmodel.shape

# %%
from sklearn import svm
svm_ = svm.SVC()
svm_c = svm_.fit(Xtrainmodel, Ytrain)
svmy_pred = svm_c.predict(Xtestmodel)

# %%
print(metrics.accuracy_score(Ytest, svmy_pred))
print(metrics.f1_score(Ytest, svmy_pred))
print(metrics.roc_auc_score(Ytest, svmy_pred))
print(metrics.matthews_corrcoef(Ytest, svmy_pred))

# %%
svmconf_mattest = confusion_matrix(Ytest, svmy_pred)
print(svmconf_mattest)

# %%
print(cross_val_score(svm_c,Xtrainmodel, Ytrain,cv=10,scoring="accuracy").mean())
print(cross_val_score(svm_c,Xtrainmodel, Ytrain,cv=10,scoring="f1").mean())
print(cross_val_score(svm_c,Xtrainmodel, Ytrain,cv=10,scoring="roc_auc").mean())

# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
svmycm_pred = cross_val_predict(svm_c,Xtrainmodel,Ytrain,cv=10)
svmconf_mat = confusion_matrix(Ytrain, svmycm_pred)
print(svmconf_mat)

# %%
pd.value_counts(Ytrain)

# %%
pd.value_counts(Ytest)

# %%
Ytrain.shape

# %%
Ytest.shape

# %% [markdown]
# # K近邻分类&调参

# %%
from sklearn import model_selection

# %%
from sklearn.neighbors import KNeighborsClassifier

# %%
# 设置待测试的不同k值
K = np.arange(1,8)
# 构建空的列表，用于存储r2
AUC = []
for k in K:
    # 使用10重交叉验证的方法，比对每一个k值下KNN模型的预测r2
    cv_result = model_selection.cross_val_score(KNeighborsClassifier(n_neighbors = int(k)),
                                                                         Xtrainmodel,Ytrain,cv = 10, scoring='accuracy')
    AUC.append(cv_result.mean())
AUC

# %%
import matplotlib.pyplot as plt

# %%
# 从k个平均r2中挑选出最大值所对应的下标    
arg_max = np.array(AUC).argmax()
# 中文和负号的正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
# 绘制不同K值与平均预测r2之间的折线图
plt.plot(K, AUC)
# 添加点图
plt.scatter(K, AUC)
# 添加文字说明
plt.text(K[arg_max], AUC[arg_max], 'k = %s' %int(K[arg_max]))
# 显示图形
plt.xlabel("Selected k value",fontsize=15)
plt.ylabel("The score of cross-validation",fontsize=15)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("Selected k value.png",dpi=600,bbox_inches='tight')#保存图片
plt.show

# %%
neigh = KNeighborsClassifier(n_neighbors=7)
neigh_c = neigh.fit(Xtrainmodel, Ytrain)
knny_pred = neigh_c.predict(Xtestmodel)

# %%
print(metrics.accuracy_score(Ytest, knny_pred))
print(metrics.f1_score(Ytest, knny_pred))
print(metrics.roc_auc_score(Ytest, knny_pred))
print(metrics.matthews_corrcoef(Ytest, knny_pred))

# %%
knnconf_mattest = confusion_matrix(Ytest, knny_pred)
print(knnconf_mattest)

# %%
print(cross_val_score(neigh_c,Xtrainmodel, Ytrain,cv=10,scoring="accuracy").mean())
print(cross_val_score(neigh_c,Xtrainmodel, Ytrain,cv=10,scoring="f1").mean())
print(cross_val_score(neigh_c,Xtrainmodel, Ytrain,cv=10,scoring="roc_auc").mean())

# %%
knnycm_pred = cross_val_predict(neigh_c,Xtrainmodel,Ytrain,cv=10)
knnconf_mat = confusion_matrix(Ytrain, knnycm_pred)
print(knnconf_mat)

# %% [markdown]
# # 梯度提升决策树分类&调参

# %%
from sklearn.ensemble import GradientBoostingClassifier
GB = GradientBoostingClassifier(random_state=68, n_estimators=50)
GB_c = GB.fit(Xtrainmodel,Ytrain)
GBy_pred = GB_c.predict(Xtestmodel)
GBy0_pred = GB_c.predict(Xtrainmodel)

# %%
# 计算测试集的R2、MSE值
print(metrics.accuracy_score(Ytest, GBy_pred))
print(metrics.f1_score(Ytest, GBy_pred))
print(metrics.roc_auc_score(Ytest, GBy_pred))
print(metrics.matthews_corrcoef(Ytest, GBy_pred))

# %%
from sklearn.model_selection import GridSearchCV
param_test5={'n_estimators':range(1,300,10)}
grid_search5=GridSearchCV(estimator=GradientBoostingClassifier(random_state=68),param_grid=param_test5,scoring='accuracy',cv=10)
grid_search5.fit(Xtrainmodel,Ytrain)
print(grid_search5.best_params_)
print(grid_search5.best_score_)

# %%
gbconf_mattest = confusion_matrix(Ytest,GBy_pred)
print(gbconf_mattest)

# %%
print(cross_val_score(GB_c,Xtrainmodel, Ytrain,cv=10,scoring="accuracy").mean())
print(cross_val_score(GB_c,Xtrainmodel, Ytrain,cv=10,scoring="f1").mean())
print(cross_val_score(GB_c,Xtrainmodel, Ytrain,cv=10,scoring="roc_auc").mean())

# %%
GBycm_pred = cross_val_predict(GB_c,Xtrainmodel,Ytrain,cv=10)
GBconf_mat = confusion_matrix(Ytrain, GBycm_pred)
print(GBconf_mat)

# %% [markdown]
# # 决策树分类模型&调参

# %%
from sklearn import tree

# %%
from sklearn.model_selection import GridSearchCV #网格搜索

# %%
# 预设各参数的不同选项值
max_depth = [5,7,9,15,20,30]
min_samples_leaf = [5,6,7,8,9,10,15,20,30]
min_samples_split = [5,6,7,8,9,10,15,20,30]
parameters = {'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}
# 网格搜索法，测试不同的参数值
grid_dt = GridSearchCV(estimator = tree.DecisionTreeClassifier(), param_grid = parameters, cv=10)
# 模型拟合
grid_dt_c = grid_dt.fit(Xtrainmodel, Ytrain)
# 返回最佳组合的参数值
grid_dt_c.best_params_,grid_dt_c.best_score_

# %%
dtc = tree.DecisionTreeClassifier(max_depth = 9, min_samples_leaf = 6, min_samples_split = 10,random_state=85)
dtc_c = dtc.fit(Xtrainmodel, Ytrain)
dtcy_pred = dtc_c.predict(Xtestmodel)

# %%
print(metrics.accuracy_score(Ytest, dtcy_pred))
print(metrics.f1_score(Ytest, dtcy_pred))
print(metrics.roc_auc_score(Ytest, dtcy_pred))
print(metrics.matthews_corrcoef(Ytest, dtcy_pred))

# %%
dtcconf_mattest = confusion_matrix(Ytest,dtcy_pred)
print(dtcconf_mattest)

# %%
print(cross_val_score(dtc,Xtrainmodel, Ytrain,cv=10,scoring="accuracy").mean())
print(cross_val_score(dtc,Xtrainmodel, Ytrain,cv=10,scoring="f1").mean())
print(cross_val_score(dtc,Xtrainmodel, Ytrain,cv=10,scoring="roc_auc").mean())

# %%
dtcycm_pred = cross_val_predict(dtc,Xtrainmodel,Ytrain,cv=10)
dtcconf_mat = confusion_matrix(Ytrain, dtcycm_pred)
print(dtcconf_mat)


