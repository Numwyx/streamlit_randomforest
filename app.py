import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap

st.set_page_config(    
    page_title="Risk Assessment of 7-Day Mortality in Critically Ill Patients with Traumatic Spinal Cord Injury",
    page_icon="⭕",
    layout="wide"
)

st.markdown('''
    <h1 style="font-size: 20px; text-align: center; color: black; background: #1E88E5; border-radius: .5rem; margin-bottom: 1rem;">
    Risk Assessment of 7-Day Mortality in Critically Ill Patients with Traumatic Spinal Cord Injury
    </h1>''', unsafe_allow_html=True)

# 导入模型文件
with open("best_rf.pkl", 'rb') as f:
    model = pickle.load(f)

with st.expander("**Predict input**", True):
    col = st.columns(5)
    
d1 = {"Cervical":1, "Thoracic":2, "Lumbar":3, "Unspecified":4}
d2 = {"Complete":1, "Incomplete":2, "Unspecified":3}
d3 = {"Unused":0, "Used":1}
dd = [d1, d2, d3]

x = ['injury_site', 'injury_type', 'vasoactive_drug', 
    'charlson_comorbidity_index', 'sbp_min', 'ph_min', 
    'wbc_max', 'sodium_max', 'lactate_min', 'temperature_min']
x1 = ['Injury Site', 'Injury Type', 'Vasoactive Drug', 
    'Charlson Comorbidity Index', 'sbp_min', 'ph_min', 
    'wbc_max', 'sodium_max', 'lactate_min', 'temperature_min']
y = [1.0, 3.0, 1.0, 4.0, 77.0, 7.27, 10.4, 138.0, 0.7, 35.72]

inputdata = {}

for i, j in zip(x, y):
    if x.index(i) in [0, 1, 2]:
        inputdata[i] = dd[x.index(i)][col[x.index(i)%5].selectbox(i, dd[x.index(i)])]
    elif x.index(i)==3:
        inputdata[i] = col[x.index(i)%5].number_input(i, value=int(j), min_value=0, max_value=37, step=1)
    else:
        inputdata[i] = col[x.index(i)%5].number_input(i, value=float(j), min_value=0.00,)

predata = pd.DataFrame([inputdata])
data = predata.copy()

with st.expander("**Predict result**", True):
    d = model.predict_proba(predata).flatten()
    
    # SHAP 值的计算  
    explainer = shap.TreeExplainer(model)  # 使用TreeExplainer来解释随机森林模型  
    shap_values = explainer.shap_values(predata.iloc[0, :])

    shap_plot = shap.plots.force(
        explainer.expected_value[1], 
        shap_values[:,1].flatten(), 
        predata.iloc[0, :], 
        show=False, 
        matplotlib=True)
    
    for text in plt.gca().texts:  # 遍历当前坐标轴的所有文本对象 
        if "=" in text.get_text():
            text.set_rotation(-15)  # 设置旋转角度，修改为你希望的角度 
            text.set_va('top')
        text.set_bbox(dict(facecolor='none', alpha=0.5, edgecolor='none'))
    
    plt.tight_layout()  
    st.pyplot(plt.gcf())
    
    st.markdown(f'''
    <div style="font-size: 20px; text-align: center; color: red; background: transparent; border-radius: .5rem; margin-bottom: 1rem;">
    The 7-day mortality risk for this critically ill spinal cord injury patient is: {round(d[1]*100, 2)}%
    </div>''', unsafe_allow_html=True)









