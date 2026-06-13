import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from main import commonGSK
from softGSK import SoftGSK
st.set_page_config(page_title = 'GSK', layout = 'wide')
st.title('Визуализатор GSK')
st.markdown('Разделение конечных множеств с использованием строгого и мягкого методов GSK.')
st.sidebar.header('1. Настройки данных')
data_source = st.sidebar.radio('Источник данных:', ['Тесты', 'Загрузить CSV'])
X, y = None, None
if data_source == 'Тесты':
    dataset_type = st.sidebar.selectbox('Выберите тест:',['Идеально разделимые', 'Пересекающиеся', 'С выбросом', 'Полумесяцы'])
    if dataset_type == 'Идеально разделимые':
        X, y = make_blobs(n_samples=200, centers=[[2, 2], [8, 8]], cluster_std=1.0, random_state=42)
    elif dataset_type == 'Пересекающиеся':
        X, y = make_blobs(n_samples=200, centers=[[4, 4], [5, 5]], cluster_std=1.5, random_state=42)
    elif dataset_type == 'С выбросом':
        X, y = make_blobs(n_samples=200, centers=[[2, 2], [8, 8]], cluster_std=1.0, random_state=42)
        X = np.vstack([X, [7.5, 7.5]])
        y = np.append(y, 0)
    elif dataset_type == 'Полумесяцы':
        X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
else:
    uploaded_file = st.sidebar.file_uploader('Загрузите CSV (колонки: x1, x2, label)', type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
st.sidebar.header('2. Выбор алгоритма')
algo_type = st.sidebar.radio('Метод разделения:', ['Строгий GSK', 'Мягкий GSK'])
mu_val = None
if algo_type == 'Мягкий GSK': mu_val = st.sidebar.slider('Штрафной параметр (mu)', min_value=0.01, max_value=1.0, value=0.1, step=0.01, help = 'Чем меньше C, тем сильнее штраф.')
max_iter = st.sidebar.number_input('Максимум итераций:', min_value=100, max_value=10000, value=1000, step=100)
if X is not None and y is not None:
    classes = np.unique(y)
    P1 = X[y == classes[0]]
    P2 = X[y == classes[1]]
    st.sidebar.button('ЗАПУСТИТЬ АЛГОРИТМ', use_container_width=True, type='primary')
    start_time = time.time()
    w, beta = None, None
    success = True
    try:
        if algo_type == 'Строгий GSK':
            model = commonGSK(P1, P2)
            w, beta = model.fit(max_iter=max_iter)
        else:
            model = SoftGSK(P1, P2)
            w, beta = model.fit(mu=mu_val, max_iter=max_iter)

        elapsed_time = time.time() - start_time
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-5: success = False
    except Exception as e:
        success = False
        st.error(f'Ошибка выполнения: {e}')
    st.subheader('Результаты оптимизации')
    col1, col2, col3, col4 = st.columns(4)
    if success:
        y_true = np.concatenate([np.ones(len(P1)), -np.ones(len(P2))])
        X = np.vstack([P1, P2])
        projections = np.dot(X, w) + beta
        y_pred = np.sign(projections)
        y_pred[y_pred == 0] = 1
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        st.markdown('Метрики')
        col1, col2, col3, col4 = st.columns(4)

        with col1: st.metric(label='Accuracy', value=f'{acc:.3f}')
        with col2: st.metric(label='Precision', value=f'{prec:.3f}')
        with col3: st.metric(label='Recall', value=f'{rec:.3f}')
        with col4: st.metric(label='F1-score', value=f'{f1:.3f}')

    else: st.warning('Алгоритм выродился: множества линейно неразделимы в строгой постановке. Переключитесь на "Мягкий GSK".')
    st.subheader('Визуализация разделяющей прямой')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(P1[:, 0], P1[:, 1], color='royalblue', edgecolors='k', s=60, label=f'Множество 1 ({len(P1)} точек)')
    ax.scatter(P2[:, 0], P2[:, 1], color='crimson', edgecolors='k', s=60, label=f'Множество 2 ({len(P2)} точек)')
    if success and w is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if abs(w[1]) > 1e-5:
            x_vals = np.array(xlim)
            y_vals = -(w[0] * x_vals + beta) / w[1]
            ax.plot(x_vals, y_vals, 'k--', linewidth=2.5, label='Разделяющая прямая')
            margin = 1 / w[1]
            ax.plot(x_vals, y_vals + margin, 'k:', alpha=0.5, label='Граница полосы P1')
            ax.plot(x_vals, y_vals - margin, 'k:', alpha=0.5, label='Граница полосы P2')
        else:
            x_val = -beta / w[0]
            ax.axvline(x=x_val, color='k', linestyle='--', linewidth=2.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
else:
    st.info("Пожалуйста, выберите набор данных в левом меню.")
