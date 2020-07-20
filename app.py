import streamlit as st
import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))


def predict_forest(ffmc,dmc,dc,isi,temp,rh,wind,rain):
    input=np.array([[ffmc,dmc,dc,isi,temp,rh,wind,rain]]).astype(np.float64)

    prediction = model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0],2)
    print(type(pred))
    return float(pred)

def main():
    st.title("FOREST FIRE PREDICTION")

    ffmc= st.slider("FFMC index from the FWI system:",18.7,96.20)
    dmc = st.slider("DMC index from the FWI system:",1.1,291.3)
    dc = st.slider("DC index from the FWI system:",7.9,860.6)
    isi = st.slider("ISI index from the FWI system:",0.0,56.10)
    temp = st.slider("Temperature in Celsius degrees: ",2.2,33.30)
    rh = st.slider("Relative Humidity in %:",15,100)
    wind = st.slider("Wind Speed in km/h:",0.40,9.40)
    rain = st.slider("Rain in mm/m2",0.0,6.4)


    if st.button("predict"):
       output=predict_forest(ffmc,dmc,dc,isi,temp,rh,wind,rain)
       st.success('The Probability of Fire Taking Place is {}'.format(output))


       if output > 0.5:
           st.markdown("### your forest is in danger")
       else:
           st.markdown("### your forest is safe")


if __name__=='__main__':
    main()