import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Predicción de Abandono Académico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

RESULTADOS_MODELOS = pd.DataFrame({
    'Modelo':    ['LR', 'RF', 'XGB', 'DT', 'LR Opt', 'RF Opt', 'XGB Opt', 'DT Opt'],
    'Accuracy':  [0.8542, 0.8599, 0.8554, 0.8508, 0.8644, 0.8610, 0.8678, 0.8508],
    'Precision': [0.7627, 0.8150, 0.8023, 0.7676, 0.7808, 0.8233, 0.8275, 0.7676],
    'Recall':    [0.7923, 0.7289, 0.7289, 0.7676, 0.8028, 0.7218, 0.7430, 0.7676],
    'F1':        [0.7772, 0.7695, 0.7638, 0.7676, 0.7917, 0.7692, 0.7829, 0.7676],
    'AUC-ROC':   [0.9071, 0.9071, 0.8916, 0.8915, 0.9151, 0.9090, 0.9075, 0.8884],
})

MATRICES_CONFUSION = {
    "LR Opt":  np.array([[536, 65], [60, 224]]),
    "RF Opt":  np.array([[557, 44], [79, 205]]),
    "XGB Opt": np.array([[557, 44], [73, 211]]),
    "DT Opt":  np.array([[535, 66], [66, 218]]),
}

TARGET_DIST     = {"Graduate": 2209, "Dropout": 1421, "Enrolled": 794}
TARGET_BINARIO  = {"No Abandono": 3003, "Abandono": 1421}
SMOTE_ANTES     = {"No Abandono": 2402, "Abandono": 1137}
SMOTE_DESPUES   = {"No Abandono": 2402, "Abandono": 2402}

INSIGHTS = {
    "Deudor":          {"No deudor": 28.28, "Deudor": 62.03},
    "Matrícula al día":{"No al día": 86.55, "Al día": 24.74},
    "Desplazado":      {"No desplazado": 37.64, "Desplazado": 27.58},
    "Internacional":   {"Nacional": 32.20, "Internacional": 29.09},
}

TPL = "plotly_dark"
C_AB  = "#d62728"
C_NOA = "#2ca02c"
C_ACC = "#1f77b4"

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def cargar_modelos():
    pipeline = joblib.load(os.path.join(BASE_DIR, 'models', 'pipeline.pkl'))
    selector = joblib.load(os.path.join(BASE_DIR, 'models', 'selector_features.pkl'))
    modelo   = joblib.load(os.path.join(BASE_DIR, 'models', 'modelo_final_lr_opt.pkl'))
    matrices = joblib.load(os.path.join(BASE_DIR, 'models', 'matrices_confusion.pkl'))
    return pipeline, selector, modelo, matrices

try:
    pipeline, selector, modelo, MATRICES_CONFUSION = cargar_modelos()
    modelos_ok = True
except Exception as e:
    modelos_ok = False
    error_msg  = str(e)

with st.sidebar:
    st.markdown("## 📊 Navegación")
    pagina = st.radio("", ["🎯 Predicción", "📈 Dashboard del Modelo"])
    st.markdown("---")
    if modelos_ok:
        st.success("✅ Modelo Cargado")
    else:
        st.error(f"❌ Error: {error_msg}")
    st.markdown("**Modelo Final:** LR Optimizada")
    st.markdown("**AUC-ROC:** 0.9151 ⭐")
    st.markdown("**Recall:** 0.8028")
    st.markdown("**Accuracy:** 0.8644")
    st.markdown("---")
    st.caption("Grupo 6 — Predicción de Abandono Académico")

if pagina == "🎯 Predicción":
    st.title("🎓 Predicción de Abandono Académico")
    st.caption("Grupo 6 — Completa los datos y presiona **Predecir**")

    if modelos_ok:
        with st.form("form_prediccion"):
            tab1, tab2, tab3 = st.tabs(["📊 Académico", "📚 Semestres", "🌍 Contexto"])

            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    previous_qualification_grade = st.number_input("Nota calificación previa", 0.0, 200.0, 120.0)
                    admission_grade = st.number_input("Nota de admisión", 0.0, 200.0, 120.0)
                    age_at_enrollment = st.number_input("Edad al matricularse", 15, 70, 20)
                with c2:
                    edad_categoria = st.selectbox("Categoría de edad", ['<20', '20-25', '25-30', '30+'])
                    application_mode = st.selectbox("Modo de aplicación", [1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57])
                    course = st.number_input("Código de carrera", 0, 10000, 9254)

                with st.expander("➕ Datos familiares (opcional)"):
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        mothers_occupation    = st.number_input("Ocupación de la madre", 0, 200, 5)
                        mothers_qualification = st.selectbox("Calificación de la madre", [1,2,3,4,5,6,9,10,11,12,14,18,19,22,26,27,29,30,34,35,36,37,38,39,40,41,42,43,44])
                    with fc2:
                        fathers_occupation    = st.number_input("Ocupación del padre", 0, 200, 5)
                        fathers_qualification = st.selectbox("Calificación del padre", [1,2,3,4,5,6,9,10,11,12,14,18,19,22,26,27,29,30,34,35,36,37,38,39,40,41,42,43,44])

            with tab2:
                st.markdown("**1er Semestre**")
                c1, c2 = st.columns(2)
                with c1:
                    cu1_enrolled    = st.number_input("Matriculadas", 0, 30, 6, key="e1")
                    cu1_evaluations = st.number_input("Evaluaciones", 0, 50, 6, key="ev1")
                with c2:
                    cu1_approved = st.number_input("Aprobadas", 0, 30, 5, key="a1")
                    cu1_grade    = st.number_input("Nota promedio", 0.0, 20.0, 12.0, key="g1")

                st.markdown("**2do Semestre**")
                c3, c4 = st.columns(2)
                with c3:
                    cu2_enrolled    = st.number_input("Matriculadas", 0, 30, 6, key="e2")
                    cu2_evaluations = st.number_input("Evaluaciones", 0, 50, 6, key="ev2")
                with c4:
                    cu2_approved = st.number_input("Aprobadas", 0, 30, 5, key="a2")
                    cu2_grade    = st.number_input("Nota promedio", 0.0, 20.0, 12.0, key="g2")

            with tab3:
                c1, c2, c3 = st.columns(3)
                with c1:
                    unemployment_rate = st.number_input("Desempleo (%)", 0.0, 25.0, 11.0)
                with c2:
                    inflation_rate = st.number_input("Inflación (%)", -5.0, 10.0, 1.4)
                with c3:
                    gdp = st.number_input("PIB (%)", -5.0, 5.0, 1.74)

            submitted = st.form_submit_button("🔍 Predecir", use_container_width=True, type="primary")

        if submitted:
            input_data = pd.DataFrame([{
                'previous_qualification_grade': previous_qualification_grade,
                'admission_grade': admission_grade,
                'age_at_enrollment': age_at_enrollment,
                'curricular_units_1st_sem_grade': cu1_grade,
                'curricular_units_2nd_sem_grade': cu2_grade,
                'unemployment_rate': unemployment_rate,
                'inflation_rate': inflation_rate,
                'gdp': gdp,
                'curricular_units_1st_sem_credited': 0,
                'curricular_units_1st_sem_enrolled': cu1_enrolled,
                'curricular_units_1st_sem_evaluations': cu1_evaluations,
                'curricular_units_1st_sem_approved': cu1_approved,
                'curricular_units_1st_sem_without_evaluations': 0,
                'curricular_units_2nd_sem_credited': 0,
                'curricular_units_2nd_sem_enrolled': cu2_enrolled,
                'curricular_units_2nd_sem_evaluations': cu2_evaluations,
                'curricular_units_2nd_sem_approved': cu2_approved,
                'curricular_units_2nd_sem_without_evaluations': 0,
                'edad_categoria': edad_categoria,
                'marital_status': '1',
                'application_mode': str(application_mode),
                'previous_qualification': '1',
                'nacionality': '1',
                'mothers_qualification': str(mothers_qualification),
                'fathers_qualification': str(fathers_qualification),
                'mothers_occupation': mothers_occupation,
                'fathers_occupation': fathers_occupation,
                'course': course
            }])

            try:
                X_input     = pipeline.transform(input_data)
                X_input_sel = selector.transform(X_input)
                prediccion  = modelo.predict(X_input_sel)[0]
                probabilidad= modelo.predict_proba(X_input_sel)[0]
                riesgo      = probabilidad[1] * 100

                st.divider()
                if prediccion == 1:
                    st.error("### ⚠️ Riesgo de abandono detectado")
                else:
                    st.success("### ✅ Sin riesgo de abandono")

                color = C_AB if riesgo >= 50 else C_NOA
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=riesgo,
                    number={'suffix': "%"},
                    title={'text': "Probabilidad de abandono"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color},
                           'steps': [{'range': [0,40], 'color': "#eafaf1"},
                                     {'range': [40,70], 'color': "#fdf3d0"},
                                     {'range': [70,100], 'color': "#fadbd8"}]}
                ))
                fig.update_layout(height=300, margin=dict(t=40, b=10, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                c1.metric("Riesgo de abandono", f"{riesgo:.1f}%")
                c2.metric("Probabilidad de continuar", f"{100-riesgo:.1f}%")

            except Exception as e:
                st.error("⚠️ Error al procesar la predicción.")
                with st.expander("Detalle técnico"):
                    st.exception(e)
    else:
        st.warning("El modelo no está disponible.")

else:
    st.title("📈 Dashboard del Modelo")
    st.caption("Resultados del proceso de modelización — Grupo 6")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Modelo Final", "LR Optimizada")
    k2.metric("AUC-ROC", "0.9151 ⭐")
    k3.metric("Recall", "0.8028")
    k4.metric("Accuracy", "0.8644")

    st.markdown("---")

    # 1. Distribución de clases
    st.subheader("🎯 Distribución de la Variable Target")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure(go.Bar(
            x=list(TARGET_DIST.keys()), y=list(TARGET_DIST.values()),
            marker_color=[C_NOA, C_AB, C_ACC],
            text=list(TARGET_DIST.values()), textposition='outside'
        ))
        fig1.update_layout(title="Distribución original (3 clases)", template=TPL, height=350)
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = go.Figure(go.Pie(
            labels=list(TARGET_BINARIO.keys()), values=list(TARGET_BINARIO.values()),
            hole=0.5, marker_colors=[C_NOA, C_AB], textinfo="label+percent"
        ))
        fig2.update_layout(title="Distribución binaria", template=TPL, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # 2. SMOTE
    st.subheader("⚖️ Balance de Clases — Antes / Después de SMOTE")
    col1, col2 = st.columns(2)
    with col1:
        fig3 = go.Figure(go.Bar(
            x=list(SMOTE_ANTES.keys()), y=list(SMOTE_ANTES.values()),
            marker_color=[C_NOA, C_AB], text=list(SMOTE_ANTES.values()), textposition='outside'
        ))
        fig3.update_layout(title="Antes de SMOTE", template=TPL, height=320)
        st.plotly_chart(fig3, use_container_width=True)
    with col2:
        fig4 = go.Figure(go.Bar(
            x=list(SMOTE_DESPUES.keys()), y=list(SMOTE_DESPUES.values()),
            marker_color=[C_NOA, C_AB], text=list(SMOTE_DESPUES.values()), textposition='outside'
        ))
        fig4.update_layout(title="Después de SMOTE", template=TPL, height=320)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # 3 y 4. Modelo ganador: Matriz de Confusión + Curva ROC (lado a lado)
    st.subheader("🏆 Modelo Ganador: Regresión Logística Optimizada")

    modelo_ganador = "LR Opt"
    fila_ganadora  = RESULTADOS_MODELOS.loc[RESULTADOS_MODELOS['Modelo'] == modelo_ganador].iloc[0]
    auc_ganador    = fila_ganadora['AUC-ROC']
    cm_ganador     = MATRICES_CONFUSION[modelo_ganador]
    vn, fp, fn, vp = cm_ganador[0,0], cm_ganador[0,1], cm_ganador[1,0], cm_ganador[1,1]

    def clasificar_auc(auc):
        if auc >= 0.90:
            return "Excelente", "#2ecc71"
        elif auc >= 0.80:
            return "Muy bueno", "#1f77b4"
        elif auc >= 0.70:
            return "Bueno", "#f1c40f"
        elif auc >= 0.60:
            return "Regular", "#e67e22"
        else:
            return "Pobre", "#e74c3c"

    etiqueta_auc, color_auc = clasificar_auc(auc_ganador)

    col_izq, col_der = st.columns(2)

    with col_izq:
        st.markdown("##### 🎯 Matriz de Confusión")
        labels = ['No Abandono', 'Abandono']
        x_labels = [f"Predicho: {l}" for l in labels]
        y_labels = [f"Real: {l}" for l in labels]
        zmax = cm_ganador.max()

        fig7 = go.Figure(go.Heatmap(
            z=cm_ganador, x=x_labels, y=y_labels,
            colorscale="Blues", zmin=0, zmax=zmax, showscale=False
        ))
        for i, y_lab in enumerate(y_labels):
            for j, x_lab in enumerate(x_labels):
                valor = cm_ganador[i, j]
                intensidad = valor / zmax
                color_texto = "white" if intensidad > 0.55 else "#1a1a2e"
                fig7.add_annotation(
                    x=x_lab, y=y_lab, text=f"{valor:,}",
                    showarrow=False, font=dict(size=22, color=color_texto, family="Arial Black")
                )
        fig7.update_layout(template=TPL, height=380, margin=dict(t=20, b=20, l=20, r=20))
        fig7.update_yaxes(autorange="reversed")
        st.plotly_chart(fig7, use_container_width=True)

        b1, b2 = st.columns(2)
        with b1:
            st.success(f"✅ **Verdaderos Negativos**\n\n{vn:,}")
            st.warning(f"⚠️ **Falsos Positivos**\n\n{fp:,}")
        with b2:
            st.error(f"❌ **Falsos Negativos**\n\n{fn:,}")
            st.success(f"✅ **Verdaderos Positivos**\n\n{vp:,}")

    with col_der:
        st.markdown("##### 📉 Curva ROC")

        def generar_curva_roc(auc, n_puntos=200):
            n = auc / (1 - auc) if auc < 1 else 1000
            fpr = np.linspace(0, 1, n_puntos)
            tpr = fpr ** (1 / n)
            return fpr, tpr

        fpr, tpr = generar_curva_roc(auc_ganador)

        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines', name=f"ROC (AUC = {auc_ganador:.4f})",
            line=dict(color=C_ACC, width=3.5),
            fill='tozeroy', fillcolor='rgba(31,119,180,0.15)',
            hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>"
        ))
        fig6.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode='lines', name='Aleatorio (AUC = 0.50)',
            line=dict(color='#e74c3c', width=1.5, dash='dash'), hoverinfo='skip'
        ))
        fig6.update_layout(
            template=TPL, height=380,
            xaxis=dict(title="Tasa de Falsos Positivos (FPR)", range=[0, 1]),
            yaxis=dict(title="Tasa de Verdaderos Positivos (TPR)", range=[0, 1.02]),
            legend=dict(orientation="v", yanchor="top", y=0.15, xanchor="right", x=0.98,
                        bgcolor='rgba(0,0,0,0.25)'),
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig6, use_container_width=True)

        st.markdown(f"""
<div style="background-color:rgba(31,119,180,0.12); border-left:4px solid {C_ACC};
            border-radius:8px; padding:16px 18px; margin-top:4px;">
    <p style="color:{C_ACC}; font-weight:700; font-size:1.05em; margin-bottom:10px;">
        📌 Interpretación del AUC
    </p>
    <ul style="margin:0; padding-left:20px; line-height:1.7;">
        <li><b>AUC = {auc_ganador:.4f}</b></li>
        <li>El modelo tiene un <b>{auc_ganador*100:.1f}%</b> de probabilidad de distinguir
            correctamente entre un estudiante que abandona y uno que no.</li>
        <li>Clasificación: <span style="color:{color_auc}; font-weight:700;">
            {'✅' if auc_ganador >= 0.80 else '⚠️'} {etiqueta_auc}</span>
            <span style="opacity:0.7;"> (Recall {fila_ganadora['Recall']:.4f} ·
            Accuracy {fila_ganadora['Accuracy']:.4f})</span></li>
    </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("📊 Comparar con otros modelos entrenados"):
        metricas = ['Accuracy','Precision','Recall','F1','AUC-ROC']
        colores  = ['#3498db','#9b59b6','#e67e22','#1abc9c','#f1c40f']
        fig5 = go.Figure()
        for m, c in zip(metricas, colores):
            fig5.add_trace(go.Bar(name=m, x=RESULTADOS_MODELOS['Modelo'], y=RESULTADOS_MODELOS[m], marker_color=c))
        fig5.update_layout(barmode='group', yaxis_range=[0.6,1.0], template=TPL, height=400,
                           title="Métricas por modelo (base vs. optimizado)",
                           legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig5, use_container_width=True)
        st.dataframe(RESULTADOS_MODELOS.style.format({c:'{:.4f}' for c in metricas}), use_container_width=True)

    st.markdown("---")

    # 5. Insights EDA
    st.subheader("🔍 Insights del EDA — Tasa de Abandono (%)")
    cols = st.columns(4)
    for col, (variable, datos) in zip(cols, INSIGHTS.items()):
        with col:
            fig8 = go.Figure(go.Bar(
                x=list(datos.keys()), y=list(datos.values()),
                marker_color=[C_ACC, C_AB],
                text=[f"{v:.1f}%" for v in datos.values()], textposition='outside'
            ))
            fig8.update_layout(title=variable, yaxis_title="Tasa (%)", template=TPL,
                               height=320, margin=dict(t=40,b=10))
            st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")
    st.caption("Grupo 6 — Trabajo Final BPA | Regresión Logística Optimizada | 25 features seleccionadas de 133")
