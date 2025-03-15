import streamlit as st

def SideBar():
    st.navigation(tabs).run()

tabs = {
    "Desciption": [
        st.Page('pages/description/machine_learning_description.py', title="Machine Learning"),
        st.Page("pages/description/deep_learning_description.py", title="Neural Network"),
    ],
    "Demo": [
        st.Page("pages/demo/machine_learning_demo.py", title="Machine Learning"),
        st.Page("pages/demo/deep_learning_demo.py", title="Neural Network"),
    ],
}



