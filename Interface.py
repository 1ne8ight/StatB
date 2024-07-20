import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)
# Chargement des données pour la normalisation
data = pd.read_csv("simulated_cancer_data.csv")

# Vérifier les colonnes du DataFrame
# st.write(data.columns)

# Définir les colonnes
gene_columns = [f'gene{i+1}' for i in range(2)]
protein_columns = [f'protein{i+1}' for i in range(5)]
clinical_columns = ['age', 'tumor_size', 'lymph_node_status', 'estrogen_receptor', 'progesterone_receptor', 'her2_status'] + [f'clinical_var{i}' for i in range(1, 5)]

# Vérifier la présence des colonnes
missing_cols = [col for col in gene_columns + protein_columns + clinical_columns if col not in data.columns]
if missing_cols:
    raise ValueError(f"Les colonnes suivantes sont manquantes: {missing_cols}")

# Normalisation des données
scaler = StandardScaler()
data[gene_columns + protein_columns + clinical_columns] = scaler.fit_transform(data[gene_columns + protein_columns + clinical_columns])

# Application Streamlit
st.title('Prédiction de la Réponse au Traitement du Cancer du Sein')

# Entrée des données par l'utilisateur
st.sidebar.header('Entrez les données du patient')

def user_input_features():
    age = st.sidebar.slider('Age', 20, 80, 50)
    tumor_size = st.sidebar.slider('Taille de la tumeur', 0.5, 10.0, 3.0)
    lymph_node_status = st.sidebar.selectbox('Statut des ganglions lymphatiques', [0, 1])
    estrogen_receptor = st.sidebar.selectbox('Récepteur d’œstrogènes', [0, 1])
    progesterone_receptor = st.sidebar.selectbox('Récepteur de progestérone', [0, 1])
    her2_status = st.sidebar.selectbox('Statut HER2', [0, 1])
    clinical_var1 = st.sidebar.slider('Variable clinique 1', -3.0, 3.0, 0.0)
    clinical_var2 = st.sidebar.slider('Variable clinique 2', -3.0, 3.0, 0.0)
    clinical_var3 = st.sidebar.slider('Variable clinique 3', -3.0, 3.0, 0.0)
    clinical_var4 = st.sidebar.slider('Variable clinique 4', -3.0, 3.0, 0.0)
    gene1 = st.sidebar.slider('Gene1', -3.0, 3.0, 0.0)
    gene2 = st.sidebar.slider('Gene2', -3.0, 3.0, 0.0)
    protein1 = st.sidebar.slider('Protein1', -3.0, 3.0, 0.0)
    protein2 = st.sidebar.slider('Protein2', -3.0, 3.0, 0.0)
    protein3 = st.sidebar.slider('Protein3', -3.0, 3.0, 0.0)
    protein4 = st.sidebar.slider('Protein4', -3.0, 3.0, 0.0)
    protein5 = st.sidebar.slider('Protein5', -3.0, 3.0, 0.0)

    data = {'age': age,
            'tumor_size': tumor_size,
            'lymph_node_status': lymph_node_status,
            'estrogen_receptor': estrogen_receptor,
            'progesterone_receptor': progesterone_receptor,
            'her2_status': her2_status,
            'clinical_var1': clinical_var1,
            'clinical_var2': clinical_var2,
            'clinical_var3': clinical_var3,
            'clinical_var4': clinical_var4,
            'gene1': gene1,
            'gene2': gene2,
            'protein1': protein1,
            'protein2': protein2,
            'protein3': protein3,
            'protein4': protein4,
            'protein5': protein5}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Normalisation des données d'entrée
input_df[gene_columns + protein_columns + clinical_columns] = scaler.transform(input_df[gene_columns + protein_columns + clinical_columns])

# Diviser les données en ensemble d'entraînement et de test
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Préparer les étiquettes
train_treatment_response = train_data['treatment_response']
test_treatment_response = test_data['treatment_response']

# Supprimer les colonnes non nécessaires
train_data = train_data.drop(columns=['patient_id', 'treatment_response'])
test_data = test_data.drop(columns=['patient_id', 'treatment_response'])

# Modèle Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(train_data, train_treatment_response)
lr_predictions = lr_model.predict(test_data)

# Calcul de la matrice de confusion pour Logistic Regression
lr_conf_matrix = confusion_matrix(test_treatment_response, lr_predictions)

# Affichage graphique de la matrice de confusion pour Logistic Regression
plt.figure(figsize=(8, 6))
sns.heatmap(lr_conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False,
            xticklabels=['Non Réponse', 'Réponse'], yticklabels=['Non Réponse', 'Réponse'])
plt.xlabel('Prédictions')
plt.ylabel('Vraies Valeurs')
plt.title('Matrice de Confusion - Model Bayesine Hiérarchique')
plt.show()
st.pyplot()

# Évaluation du modèle
lr_auc = roc_auc_score(test_treatment_response, lr_model.predict_proba(test_data)[:, 1])

st.subheader('Évaluation du modèle')
st.write(f'Score ROC du model Bayesien: {lr_auc:.2f}')

# Prédiction sur les données entrées
u_prediction_prob = lr_model.predict_proba(input_df)[:, 1][0]
u_prediction = 'Positive' if u_prediction_prob > 0.5 else 'Négatif'

st.subheader('Prédiction de la réponse au traitement')
st.write(f'Probabilité de réponse au traitement: {u_prediction_prob:.2f}')
st.write('Réponse prédite:', u_prediction)
