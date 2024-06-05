import streamlit as st
import joblib

SVM = joblib.load('SVM.pickle')
XGB = joblib.load('XGBOOST.pickle')
NB = joblib.load('NB.pickle')

st.title("Monkey Pox Prediction")

st.write("")
st.write("#### Machine Learning Project by")
left, center, right = st.columns(3)
left.write("###### 2602089742 - Hany wijaya")
center.write("###### 2602160510 - Tasya Aulianissa")
right.write("###### 2602093802 - Wilbert Yang")

st.divider()

st.image("./assets/monkehehe.jpeg",use_column_width=True)

st.write("### Input Data")

st.write("##### Select the following suffered symptoms")
col1, col2 = st.columns(2)

col1.write("Rectal Pain")
rectal_pain = col2.checkbox("Yes", key='1')
col1.write("Sore Throat")
sore_throat = col2.checkbox("Yes", key='2')
col1.write("Penil Oedema")
penil_oedema = col2.checkbox("Yes", key='3')
col1.write("Oral Lesion")
oral_lesions = col2.checkbox("Yes", key='4')
col1.write("Solitary Lesion")
solitary_lesion = col2.checkbox("Yes", key='5')
col1.write("Swollen Tonsils")
swollen_tonsils = col2.checkbox("Yes", key='6')
col1.write("HIV Infenction")
hiv_infection = col2.checkbox("Yes", key='7')
col1.write("Sexually Transmitted Infection")
sti = col2.checkbox("Yes", key='8')

systematic_illness = st.selectbox(
    "Select the Systemic Illness",
    ["None", "Fever", "Swollen Lymph Nodes", "Muscle Aches and Pain"]
)

input = [int(rectal_pain), int(sore_throat), int(penil_oedema), int(oral_lesions), 
        int(solitary_lesion), int(swollen_tonsils), int(hiv_infection), int(sti)]

input_display = {
    'Rectal Pain':int(rectal_pain),
    'Sore throat':int(sore_throat),
    'Penile Oedema':int(penil_oedema),
    'Oral Lesions':int(oral_lesions),
    'Solitary Lesion':int(solitary_lesion),
    'Swollen Tonsils':int(swollen_tonsils),
    'HIV Infection':int(hiv_infection),
    'STI':int(sti)
    }

st.sidebar.write("")
st.sidebar.title("Input Tracker")
for symptom, value in input_display.items():
    st.sidebar.write(f"{symptom}:", bool(value))
st.sidebar.write("Systemic Illness:", systematic_illness)

st.divider()

st.write("### Predict the Result")
left, right = st.columns(2)
selected_model = left.selectbox("Select a model to predict the output", ["SVM", "Naive_Bayes","XGBOOST"])
st.write("")
predict_btn = st.button(" Predict ")

st.write("")

def encode_si(x):
    #fever, muscle, none, swollen
    if x == "Fever":
        return list([1,0,0,0])
    elif x == "Muscle Aches and Pain":
        return list([0,1,0,0])
    elif x == "Swollen Lymph Nodes":
        return list([0,0,0,1])
    else:
        return list([0,0,1,0])

features = input + encode_si(systematic_illness)

if predict_btn:
    features = [features]
    if selected_model == 'XGBOOST':
        predicted_value = XGB.predict(features)
    elif selected_model == 'Naive_Bayes':
        predicted_value = NB.predict(features)
    else:
        predicted_value = SVM.predict(features)

    st.write(f"Prediction with {selected_model}:")
    if bool(predicted_value):
        st.markdown(":red[Monkey Pox Detected!]")
    else:
        st.markdown(":green[Monkey Pox not Detected!]")

