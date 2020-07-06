import streamlit as st
import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))


def predict_forest(oxygen,temperature,humidity):
    input=np.array([[oxygen,temperature,humidity]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    print(type(pred))
    return float(pred)

def main():
    st.title("project")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px>
    <h2 style="color:white;text-align:center;"> FOREST FIRE PREDICTION </h2> 
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    oxygen = st.text_input("Oxygen")
    temperature = st.text_input("temperature")
    humidity = st.text_input("humidity")

    if st.button("predict"):
       output=predict_forest(oxygen,temperature,humidity)
       st.success('the probability of fire takes place is{}'.format(output))


       if output > 0.5:
           st.markdown("### your forest is in danger")
       else:
           st.markdown("### your forest is safe")


if __name__=='__main__':
    main()