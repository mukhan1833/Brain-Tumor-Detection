import streamlit as st
from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix
import warnings
import os
import cv2
import streamlit as st
import io

st.set_option('deprecation.showPyplotGlobalUse', False)
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'Detection System'], 
        icons=['house', 'gear'], menu_icon="cast", default_index=1)
    selected

if selected=="Home":
    st.header("Brain Tumor Detection System")
    st.subheader("What is Brain tumor?")
    st.write("A brain tumor, known as an intracranial tumor, is an abnormal mass of tissue in which cells grow and multiply uncontrollably, seemingly unchecked by the mechanisms that control normal cells.")
    st.write("""Doctors use many tests to find, or diagnose, a brain tumor and learn the type of brain tumor. They also do tests to find out if it has spread to another part of the body from where it started. This is called metastasis and is rare for a primary brain tumor. Doctors may also do tests to learn which treatments could work best.
For most types of tumors, taking a sample of the possible tumor is the only sure way for the doctor to know if an area of the body has a tumor. This may be done in a procedure called a biopsy or by removing part or all of the tumor with surgery. In a biopsy, the doctor takes a small sample of tissue for testing in a laboratory. If this is not possible, the doctor may suggest other tests that will help make a diagnosis.""")
    st.image('https://i0.wp.com/post.healthline.com/wp-content/uploads/2022/02/2009199_Understanding-Brain-Tumors-01.jpg?w=1155&h=1887')
    st.write("""Different parts of the brain control different functions, so brain tumor symptoms will vary depending on the tumor’s location. For example, a brain tumor located in the cerebellum at the back of the head may cause trouble with movement, walking, balance and coordination. If the tumor affects the optic pathway, which is responsible for sight, vision changes may occur.
The tumor’s size and how fast it’s growing also affect which symptoms a person will experience.
In general, the most common symptoms of a brain tumor may include:
    1) Headaches
    2) Seizures or convulsions
    3) Difficulty thinking, speaking or finding words
    4) Personality or behavior changes
    5) Weakness, numbness or paralysis in one part or one side of the body
    6) Loss of balance, dizziness or unsteadiness
    7) Loss of hearing
    8) Vision changes
    9) Confusion and disorientation
    10) Memory loss
""")
    st.image("https://www.hopkinsmedicine.org/-/media/images/health/1_-conditions/brain-tumors/brain-tumor-symptoms-locations.ashx?h=722&w=900&hash=1BF3CBAAFD103E5B8644DE3A1896E7FC")

if selected=="Detection System":
    st.header("Brain Tumor Detection System")
    path = os.listdir('brain_tumor/Training/')
    classes = {'no_tumor':0, 'pituitary_tumor':1}
    st.subheader("Enter your brain mri scan here:")
    X = []
    Y = []
    for cls in classes:
        pth = 'brain_tumor/Training/'+cls
        for j in os.listdir(pth):
            img = cv2.imread(pth+'/'+j, 0)
            img = cv2.resize(img, (200,200))
            X.append(img)
            Y.append(classes[cls])

    X = np.array(X)
    Y = np.array(Y)

    X_updated = X.reshape(len(X), -1)

    np.unique(Y)
    pd.Series(Y).value_counts()


    plt.imshow(X[0], cmap='gray')

    def save_uploadedfile(uploadedfile):
     with open(uploadedfile.name,"wb") as f:
         f.write(uploadedfile.getbuffer())
    
    X_updated = X.reshape(len(X), -1)

    xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,test_size=.20)

    xtrain = xtrain/255
    xtest = xtest/255

    from sklearn.decomposition import PCA

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    import warnings
    warnings.filterwarnings('ignore')

    lg = LogisticRegression(C=0.1)
    lg.fit(xtrain, ytrain)

    sv = SVC()
    sv.fit(xtrain, ytrain)

    pred = sv.predict(xtest)


    misclassified=np.where(ytest!=pred)


    dec = {0:'No Tumor', 1:'Positive Tumor'}
    plt.figure(figsize=(12,8))
    p = os.listdir('brain_tumor/Testing/')
    c=1

    image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
    if image_file is not None:
        save_uploadedfile(image_file)
#for i in os.listdir('brain_tumor/')[:9]:
    #plt.subplot(3,3,c)  
        img = cv2.imread(image_file.name,0)
        img1 = cv2.resize(img, (200,200))
        img1 = img1.reshape(1,-1)/255
        p = sv.predict(img1)
        res=plt.title(dec[p[0]])
        st.image(img)
        if dec[p[0]] == "No Tumor":
            plt.title(dec[p[0]])
            st.header("Congratulations ! their is no pituatory tumor detected")
        else:
            st.header("Pituatory tumor detected! go to you Doctor immediately we hope you get well soon!")
        clf = SVC(kernel='linear')
        clf = clf.fit(xtrain, ytrain)
        matrix = plot_confusion_matrix(clf, xtest,ytest,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
        plt.title('Confusion matrix for our classifier')
        st.pyplot()
 