🍷 葡萄酒质量预测系统 (Wine Quality Prediction)

这是一个基于 **Streamlit** 框架构建的交互式 Web 应用，利用 **随机森林回归 (Random Forest Regressor)** 机器学习模型来预测葡萄酒的质量评分。该项目旨在演示如何使用集成学习方法解决回归问题，并通过直观的界面展示预测结果。

✨ 核心功能
*   **质量预测**：输入葡萄酒的各项理化指标，实时预测其质量评分（0-10分）。
*   **智能分级**：根据预测分数自动将葡萄酒分为三个等级：
    *   **8分及以上**：优质葡萄酒
    *   **6-7分**：中等葡萄酒
    *   **6分以下**：普通葡萄酒
*   **交互式输入**：支持通过侧边栏滑块调整特征值，红/白葡萄酒类型选择。

🛠️ 技术栈
*   **框架**: Streamlit (用于构建数据应用界面)
*   **机器学习**: Scikit-learn (Random Forest 算法)
*   **数据处理**: Pandas, NumPy

📦 环境依赖
在运行项目前，请确保已安装以下 Python 库。你可以使用以下命令进行安装：

pip install streamlit scikit-learn pandas numpy

🚀 快速开始

1.  **克隆项目**
        git clone https://github.com/lovedecimal/ensemble-learning-xgboost-lightgbm.git
    cd ensemble-learning-xgboost-lightgbm
    

2.  **准备数据**
    请确保数据文件 winequalityN.csv 位于项目根目录下（与 streamlit_wine_quality.py 同级）。

3.  **启动应用**
        streamlit run streamlit_wine_quality.py
    
    应用启动后，浏览器会自动打开 http://localhost:8501。

📂 项目结构
ensemble-learning-xgboost-lightgbm/
├── streamlit_wine_quality.py    # 主程序代码
├── winequalityN.csv             # 数据集文件 (需自行准备)
└── README.md                    # 项目说明文档

📝 代码说明
*   **模型训练**：代码在后台自动加载数据，处理缺失值（用均值填充），将红/白葡萄酒映射为 0/1，并划分训练集进行模型训练。
*   **缓存机制**：使用 @st.cache_resource 装饰器缓存训练好的模型，避免每次交互都重新训练，提升响应速度。
*   **用户界面**：利用 Streamlit 的 sidebar 组件创建输入控件，主窗口展示预测结果和等级信息。

📊 数据集说明
*   项目依赖外部数据集 winequalityN.csv。
*   数据集包含红葡萄酒和白葡萄酒的理化指标（如酸度、残糖、酒精度等）及对应的质量评分。
*   *注意：若数据集中存在缺失值，程序会自动用该列均值填充以保证训练正常进行。*

📄 许可证
本项目基于 MIT 许可证开源。
