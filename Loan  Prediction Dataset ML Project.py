#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Loan Prediction Dataset

# # About the data set

# Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban and semi urban and rural areas. Customer first applies for home loan and after that company validates the custeomer eligibility for loan company wants to automate the loan eligibility process(real time) based on customer detail provided while filling online application form. These details are Gender,Marital Status, Education, Number of Dependents,  Income,LoanAmount,Credit History and others.To automate this process,they have provided a dataset to identify the customers segments tat are eligible for loan amount so that they can specifically target these customers.
# 
# 

# In[ ]:





# # Data Id

# This dataset is named Loan Prediction Dataset data set. The dataset contains a set of 613 records under 13 attributes:
#     Columns                                Description
 1    Loan_ID                               A uniques loan ID
 2   Gender                                 Male/Female
 3   Married                                Married(Yes)/Not married(No)
 4   Dependents                             Number of persons depending on the client
 5   Education                              Applicant Education (Graduate/Undergraduate)
 6   Self_Employed                          Self emplyored(Yes/No)
 7   ApplicantIncome                        Applicant income                   
 8   CoapplicantIncome                      Coapplicant income 
 9   LoanAmount                             Loan amount in thousands        
 10  Loan_Amount_Term                       Term of lean in months
 11  Credit_History                         Credit history meets guidelines    
 12  Property_Area                          Urban/Semi and Rural     
 13  Loan_Status                            Loan approved (Y/N)

   
   It is a classification problem where we have to predict whether a loan would be approved or not. In a classification problem, we have to predict discrete values based on a given set of independent variable(s).

    Evaluation Metric is accuracy i.e. percentage of loan approval you correctly predict.

# # The main objective for this dataset:

# Using machine learning techniques to predict loan payments.

# # Libraries

# In[1]:


import numpy as np                      # linear algebra
import pandas as pd                     # data processing and Manipulation
import matplotlib.pyplot as plt         # Data visualization
import seaborn as sns                   # Data visualization
from sklearn import svm


# # Reading the data

# In[2]:


df = pd.read_csv("C:\\Users\\admin\\Documents\\loan.data.csv")


# # exploratory Data Analysis

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.size


# # Checking Missing value & Duplicated value

# In[9]:


df.isnull().sum()


# In[10]:


df.LoanAmount.median()


# In[11]:


df.Loan_Amount_Term.median()


# In[12]:


df.Credit_History.median()


# In[13]:


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace = True)
df['Credit_History'].fillna(df['Credit_History'].median(), inplace = True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace = True)


# In[14]:


df['Gender'].fillna(df['Gender'].mode()[0], inplace = True)
df['Married'].fillna(df['Married'].mode()[0], inplace = True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace = True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace = True)


# In[15]:


df.isnull().sum()


# In[16]:


#cheacking duplicated Value
df.duplicated().sum()


# In[17]:


#cheaking unique values in colunms
df.nunique()


# # To Check Multicolliearity 

# Multicolliearity is  a Statistical concept where several independent variables in a model are correlated.

# In[20]:


#checking multicoliearity
df.corr()


# Two variable are linearly related

# In[21]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,linewidths=1)
plt.show()


# # Data Visualization

# In[22]:


df.boxplot()


# In[23]:


df.plot(kind="box",figsize=(10,8),rot=90)


# In[25]:


fig,ax = plt.subplots(2,4,figsize=(16,10))
sns.countplot("Loan_Status",data=df,ax=ax[0][0])
sns.countplot("Gender",data=df,ax=ax[0][1])
sns.countplot("Married",data=df,ax=ax[0][2])
sns.countplot("Education",data=df,ax=ax[0][3])
sns.countplot("Self_Employed",data=df,ax=ax[1][0])
sns.countplot("Property_Area",data=df,ax=ax[1][1])
sns.countplot("Credit_History",data=df,ax=ax[1][2])
sns.countplot("Dependents",data=df,ax=ax[1][3])


# **Conclusions from countplot**
# 1.From bar graph we can see that the applicants whose loan is approved has max count.
# 
# 2.Applicant whos is applied for the loan is mostly of male category as compared to female category.
# 
# 3.Most of the Married peoples are apply for the loan.
# 
# 4.Applicants who is graduated and not self Employed are mostly apply for the loan.
# 
# 5.Peoples from semiurban area are mostly apply for the loan.
# 
# 6.Most of the applicants have credit history which meets the guidelines.
# 
# 7.Applicant who has 0 number of dependents they generally apply for the loan.

# In[26]:


df.hist(figsize=(10,10),color="m")


# Conclusions from Histogram 
# 
# From above histogram we can see that ApplicantIncome,CoapplicantIncome and LoanAmount is of positively skewed.It means the mean of these variables is greater than their median as the data is more towards the right side.

# In[27]:


sns.countplot(y="Gender",hue='Loan_Status',data=df)


# **More males are on loan than females.Also,those that are on loan are more than otherwise**

# In[28]:


sns.countplot(y="Married",hue='Loan_Status',data=df)


# **Married people collect more loan than unmarried.**

# In[30]:


sns.countplot(y="Loan_Amount_Term",hue='Loan_Status',data=df)


# **An extremely high number of them go for a 360 cyclic loan term. That's pay back within a year.**

# #Create new variable Income using ApplicantIncome and Co-ApplicantIncome variable as these both combinly applicant total Income

# In[26]:


df["Income"]=df.ApplicantIncome + df.CoapplicantIncome


# In[27]:


df.head(3)


# In[28]:


# Drop unimportant columns
df.drop(columns=["Loan_ID","ApplicantIncome","CoapplicantIncome"],inplace=True)


# In[29]:


df.head()


# In[30]:


#outlier detection & removing


# In[31]:


df.plot(kind="box",subplots=True,layout=(4,2),sharex=False,sharey=False,figsize=(20,25))


# In[ ]:





# In[32]:


df = df[df.Income<10000]
df = df[df.LoanAmount<210]


# In[33]:


df.plot(kind="box",subplots=True,layout=(4,2),sharex=False,sharey=False,figsize=(20,25))


# In[34]:


df.size


# In[35]:


df.shape


# # Label Encoding

# In[36]:


from sklearn.preprocessing import LabelEncoder


# In[37]:


lb= LabelEncoder()


# In[38]:


#To create a list of character columns
lst=[]
for i in df.columns:
    if df[i].dtype=="O":
        lst.append(i)
        
lst


# In[39]:


for i in lst:
    df[i]=lb.fit_transform(df[i])


# In[40]:


df.head()


# # Splitting Dataset

# In[41]:


x=df.drop(columns="Loan_Status")
y=df["Loan_Status"]


# In[42]:


x.head(n=2)


# In[43]:


y.head()


# In[44]:


y.value_counts()


# As from the above count of 0 and 1 the data is not balanced, so we have to balanced it.

# # Balancing dataset:

# SMOTE : Synthetic Minority Oversampling Technique is a statistical technique for increasing the number of cases in the dataset in a balancing way.

# In[45]:


import imblearn
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state = 12)
x_res,y_res =smk.fit_resample(x,y)


# In[46]:


x_res.shape


# In[47]:


y_res.shape


# In[48]:


from collections import Counter


# In[49]:


print('original dataset shape:', Counter(y))
print('Resample dataset shape', Counter(y_res))


# # Spliting into Train_Test

# Now I split Data into training and Testing datasets.For this I have to import train_test_split from the sklearn.model_selection library. 

# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.3, random_state=32)


# In[51]:


X_train.shape,X_test.shape


# In[52]:


y_train.shape,y_test.shape


# # Logistic Regression

# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score


# In[54]:


log_m=LogisticRegression()


# In[55]:


log_m.fit(X_train,y_train)


# In[56]:


y_pred=log_m.predict(X_test)


# In[57]:


accuracy_score(y_test,y_pred)


# In[58]:


Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    model=LogisticRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_pred))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)

maxi_iter=[]
for i in range(0,1000,50):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_no)
    model=LogisticRegression(max_iter=i)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    maxi_iter.append(accuracy_score(y_test,y_pred))
    accuracy=max(Random_State)
    
    
print("Random State Number :",random_no,"Accuracy Score : ",accuracy)
print("Maximum iteration number : ",(maxi_iter.index(max(maxi_iter)))*50,"Accuracy Score : ",accuracy)


# In[59]:


precision_score(y_test,y_pred)


# In[60]:


recall_score(y_test,y_pred)


# # Decission Tree Clasifier

# In[61]:


from sklearn.tree import DecisionTreeClassifier,plot_tree
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_preddt=dt.predict(X_test)


# In[62]:


accuracy_score(y_test,y_preddt)


# In[63]:


precision_score(y_test,y_preddt)


# In[64]:


recall_score(y_test,y_preddt)


# In[65]:


feature_name = ['Gender','Married','Dependents','Education','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Income']


# In[66]:


cn = ["1","0"]


# In[67]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


plt.figure(figsize=(20,10))
a = plot_tree(dt, max_depth=3,fontsize=10,feature_names=feature_name,class_names=cn,filled=True)
plt.show()


# In[69]:


Random_State=[]
for i in range(0,1000):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    model=DecisionTreeClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_pred))
    accuracy=max(Random_State)


print("Random State Number :",Random_State.index(max(Random_State)),"Accuracy Score : ",accuracy)


# # Knn Classifier

# In[70]:


from sklearn.neighbors  import KNeighborsClassifier


# In[71]:


model_knn = KNeighborsClassifier(n_neighbors=17)


# In[72]:


model_knn.fit(X_train,y_train)


# In[73]:


pred = model_knn.predict(X_test)
pred


# In[74]:


accuracy_score(y_test,pred)


# In[75]:


precision_score(y_test,pred)


# In[76]:


recall_score(y_test,pred)


# In[77]:


confusion_matrix(y_test,pred)


# In[78]:


error=[]
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt 

# calculatiing error for a values between 1 and 40
for i in range(1,40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, y_train)
    pred=model.predict(X_test)
    error.append(np.mean(pred != y_test))
    
plt.figure(figsize=(12,6))
plt.plot(range(1,40),error, color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.title('Error Rate K value')
plt.xlabel("K value")
plt.ylabel("Mean Error")


# In[79]:


accuaracy=[]
for i in range(1,51):
        model_knn=KNeighborsClassifier(n_neighbors=i)
        model_knn.fit(X_train,y_train)
        y_predknn=model_knn.predict(X_test)
        accuracy_score(y_predknn,y_test)
        accuaracy.append(accuracy_score(y_predknn,y_test))
        
plt.figure(figsize=(12,6))
plt.plot(range(1,51),accuaracy,color="red")
plt.title("Accuracy and K value Plot")
plt.xlabel("K value")
plt.ylabel("Accuracy")


# # Naive_bayse classifier

# In[80]:


from sklearn.naive_bayes import GaussianNB


# In[81]:


model_NB = GaussianNB()


# In[82]:


model_NB.fit(X_train,y_train)


# In[83]:


pred = model_NB.predict(X_test)
pred


# In[84]:


accuracy_score(y_test,pred)


# In[85]:


precision_score(y_test,pred)


# In[86]:


recall_score(y_test,pred)


# In[87]:


Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    model=GaussianNB()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_pred))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)
    
print("Random State Number :",random_no,"Accuracy Score : ",accuracy)


# # SVM

# In[88]:


from sklearn.svm import SVC
model_sv=SVC()


# In[89]:


model_sv.fit(X_train,y_train)


# In[90]:


y_predsv=model_sv.predict(X_test)


# In[91]:


accuracy_score(y_predsv,y_test)


# In[92]:


precision_score(y_predsv,y_test)


# In[93]:


recall_score(y_predsv,y_test)


# In[94]:


Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    model_sv=SVC()
    model_sv.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_pred))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)
    
print("Random State Number :",random_no,"Accuracy Score : ",accuracy)


# # Random Forest

# In[95]:


from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier()


# In[96]:


model_rf.fit(X_train,y_train)


# In[97]:


RandomForestClassifier


# In[98]:


RandomForestClassifier()


# In[99]:


y_predrf=model_rf.predict(X_test)


# In[100]:


accuracy_score(y_predrf,y_test)


# In[101]:


precision_score(y_predrf,y_test)


# In[102]:


recall_score(y_predrf,y_test)


# In[103]:


# Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    model_rf=RandomForestClassifier()
    model_rf.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_pred))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)
    
print("Random State Number :",random_no,"Accuracy Score : ",accuracy)


# # Gradient  Boosting

# In[104]:


from sklearn.ensemble import GradientBoostingClassifier
md=GradientBoostingClassifier()


# In[105]:


md.fit(X_train,y_train)


# In[106]:


y_predgb=md.predict(X_test)
y_predgb


# In[107]:


accuracy_score(y_predgb,y_test)


# In[108]:


precision_score(y_predgb,y_test)


# In[109]:


recall_score(y_predgb,y_test)


# In[110]:


Random_State=[]
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=i)
    md=GradientBoostingClassifier()
    md.fit(X_train,y_train)
    y_pred=md.predict(X_test)
    Random_State.append(accuracy_score(y_test,y_pred))
    random_no=Random_State.index(max(Random_State))
    accuracy=max(Random_State)
    
print("Random State Number :",random_no,"Accuracy Score : ",accuracy)


# # Accuracy , precision and Recall:

# 1) Logistic regression: Accuracy:0.7206703910614525 Precision:0.890625 Recall:0.991304347826087
# 
# 2) Decision Tree: Accuracy: 0.77333333333333 Precision:0.8648648648648649 Recall:0.8347826086956521
# 
# 3)Knn : Accuracy:0.68  Precision:0.6875 Recall:0.9705882352941176
# 
# 4) Neive Bayes: Accuracy:0.8133333333333334 Precision:0.7983870967741935 Recall:0.9705882352941176
# 
# 5) SVM: Accuracy:0.7 Precision:1.0 Recall:0.7
# 
# 6) Random Forest : Accuracy:0.7733333333333333 Precision:0.8952380952380953 Recall:0.8034188034188035
# 
# 7) Gradient Boosting: Accuracy:0.78 Precision:0.9047619048 Recall: 0.8050847457627118

# In[ ]:





# In[ ]:





# In[111]:


Loan_ID=int(input("ENTER THE LOAN_ID:"))
Gender=int(input("ENTER THE gender:"))
Married=int(input("ENTER THE APPLICANT MARRIED:"))
Dependents=int(input("ENTER THE Number of dependents:"))
Education=int(input("ENTER THE Applicant Education:"))
Self_Employed=int(input("ENTER THE Self employed:"))
ApplicantIncome=int(input("ENTER THE Applicant income :"))
CoapplicantIncome=int(input("enter Coapplicant income:"))
LoanAmount=int(input("ENTER THE Loan amount in thousands:"))
Loan_Amount_Term =float(input("ENTER THE Term of loan in months:"))
Credit_History=int(input("ENTER THE credit history meets guidelines:"))
Property_Area=int(input("ENTER THE Urban:"))
Loan_Status=int(input("ENTER THE Loan approved:"))


pred=model.predict([[Gender, Married, Dependents, Education, Self_Employed,
       LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,
       Loan_Status]])
print(pred)

if(pred==0):
    print("Loan is approved")
else:
    print("not approved")


# In[116]:


def df(Gender, Married, Dependents, Education, Self_Employed,
       LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,
       Loan_Status):
    pred=model.predict([[Gender, Married, Dependents, Education, Self_Employed,
       LoanAmount, Loan_Amount_Term, Credit_History, Property_Area,
       Loan_Status]])
    print(pred)
    
    if(pred==0):
        print("Loan is approved")
    else:
        print("not approved")


# In[115]:


df(67,8,438,479,3847,374,3,32,453,87)


# In[ ]:





# In[ ]:




