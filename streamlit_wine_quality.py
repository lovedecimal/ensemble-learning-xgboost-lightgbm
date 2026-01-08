import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 页面配置
st.set_page_config(page_title="葡萄酒质量预测", page_icon="🍷")
st.title("🍷 葡萄酒质量预测（基于随机森林）")

# 加载数据+训练模型（首次运行自动训练）
@st.cache_resource
def train_model():
    # 读取数据集（确保CSV文件与代码同目录）
    data = pd.read_csv("winequalityN.csv")
    # 处理分类特征
    data['type'] = data['type'].map({'red':0, 'white':1})
    # 填充缺失值（避免训练报错）
    data = data.fillna(data.mean())
    # 划分特征与目标变量
    X = data.drop('quality', axis=1)
    y = data['quality']
    # 划分训练集
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    # 训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    # 返回模型、特征名、数据（解决作用域问题）
    return model, X.columns, data

model, feature_names, data = train_model()

# 交互输入
st.sidebar.header("输入葡萄酒特征")
inputs = {}
for feat in feature_names:
    if feat == 'type':
        inputs[feat] = st.sidebar.selectbox("葡萄酒类型", [0,1], format_func=lambda x: '红葡萄酒' if x==0 else '白葡萄酒')
    else:
        inputs[feat] = st.sidebar.slider(
            f"{feat}",
            float(data[feat].min()),
            float(data[feat].max()),
            float(data[feat].mean())
        )

# 预测+展示结果
if st.button("预测质量"):
    X_pred = np.array([list(inputs.values())])
    y_pred = model.predict(X_pred)[0]
    st.success(f"预测葡萄酒质量评分：{y_pred:.2f}")
    # 质量分级
    if y_pred >= 8:
        st.info("等级：优质葡萄酒")
    elif y_pred >= 6:
        st.info("等级：中等葡萄酒")
    else:
        st.info("等级：普通葡萄酒")