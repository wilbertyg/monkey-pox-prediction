import streamlit as st
import joblib

SVM = joblib.load('SVM.pickle')
XGB = joblib.load('XGBOOST.pickle')
NB = joblib.load('NB.pickle')

st.title("Monkey Pox Prediction")

st.write("")
st.write("#### Machine Learning Project by")
left, center, right = st.columns(3)
left.write("2602089742 - Hany Wijaya")
center.write("2602160510 - Tasya Aulianissa")
right.write("2602093802 - Wilbert Yang")

st.divider()

st.image("./assets/monkehehe.jpeg",use_column_width=True)

st.write("### Input Data")

st.write("##### Select the following suffered symptoms (you may select more than 1)")

rectal_pain = st.checkbox("[-] Rectal Pain", key='1')
sore_throat = st.checkbox("[-] Sore Throat", key='2')
penil_oedema = st.checkbox("[-] Penil Oedema", key='3')
oral_lesions = st.checkbox("[-] Oral Lesion", key='4')
solitary_lesion = st.checkbox("[-] Solitary Lesion", key='5')
swollen_tonsils = st.checkbox("[-] Swollen Tonsils", key='6')
hiv_infection = st.checkbox("[-] HIV Infenction", key='7')
sti = st.checkbox("[-] Sexually Transmitted Infection", key='8')

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

