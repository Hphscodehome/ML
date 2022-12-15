#输入库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,metrics,datasets
from zipfile import ZipFile
import pandas as pd
import csv
""" #输入数据
digits=datasets.load_digits()
digits_images_target=list(zip(digits.images,digits.target))
#画出数据(在图像识别中用到)
#枚举前四个数据集，并画出图像enumerate((digits_images_target[:4]))
for (i,(images,label)) in enumerate((digits_images_target[:4])):
    plt.subplot(4,4,i+1)
    plt.imshow(images,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.axis('off')
    plt.title('Training:%d'%label)
#训练模型 """
myzip=ZipFile("train.csv.zip")
fp=myzip.open("train.csv")
train_data=pd.read_csv(fp,header=None)
fp.close()
myzip.close()
myzip=ZipFile("test.csv.zip")
fp=myzip.open("test.csv")
test_data=pd.read_csv(fp,header=None)
fp.close()
myzip.close()
X_train=train_data.iloc[:,:40]
Y_train = train_data.iloc[:,40]
X_test = test_data
clf=svm.SVC(gamma=0.001)
clf.fit(X_train,Y_train)
predicted=clf.predict(X_test)
print(X_train[:20])
print(predicted[:20])
with open("submission.csv", "w", encoding="utf-8", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['Id','Expected'])
    for i in range(len(predicted)):
        csv_writer.writerow([i,predicted[i]])
    print("写入数据成功")
    f.close()
""" data=digits.images
n_samples=len(data)
data=data.reshape((n_samples,-1))
X=data[:n_samples//2]
y=digits.target[:n_samples//2]
clf=svm.SVC(gamma=0.01)
clf.fit(X,y)
#print(clf)
#预测
expect=digits.target[n_samples//4:]
predicted=clf.predict(data[n_samples//4:])
print('Classification report for classifier %s:\n%s\n'%(clf,metrics.classification_report(expect,predicted)))
print('clf_report Confusion Matrix:\n%s'%metrics.confusion_matrix(expect,predicted))
fpr, tpr, thresholds = metrics.roc_curve(expect,predicted, pos_label=2)
print('Classification ACU report for classifier %s:\n%s\n'%(clf,metrics.auc(fpr, tpr)))
#print (predicted)
#确认验证集的输入和预测值
images_and_predictions = list(zip(digits.images[n_samples // 4:], predicted))
for (i,(expect,predicted)) in enumerate((images_and_predictions[:12])):
    plt.subplot(4,4,i+5)
    plt.subplots_adjust(left=0.2,right=1.0,bottom=0.1,top=1.5)
    plt.imshow(expect,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.axis('off')
    plt.title('prediction:%d'%predicted)
plt.show() """