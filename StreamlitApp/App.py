import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from openpyxl import Workbook
import matplotlib.pyplot as plt
import plotly.express as px





def preprocess_data(df, is_training=False, training_features=None):
    df = pd.get_dummies(df, columns=['typeTransaction'])

    if is_training:
        features = ['montant', 'oldbalanceOrg', 'newbalanceOrig', 
                    'oldbalanceDest', 'newbalanceDest']
        features.extend([col for col in df.columns if 'typeTransaction' in col])

        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(features, "features.pkl")
    else:
        scaler = joblib.load("scaler.pkl")

        if training_features:
            missing_features = [col for col in training_features if col not in df.columns]
            for feature in missing_features:
                df[feature] = 0
            df = df[training_features]
        
        df[training_features] = scaler.transform(df[training_features])

    return df, training_features

def train_and_save_model(training_df):
    training_df, features = preprocess_data(training_df, is_training=True)

    X = training_df.drop(columns=['isFraud', 'step', 'NumeroCompte', 'Agence', 'NomClient', 'dateCreation', 'isFlaggedFraud'])
    y = training_df['isFraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "decision_tree_model.pkl")

    return model, features


@st.cache_data
def load_or_train_model(training_df=None):
    try:
        model = joblib.load("decision_tree_model.pkl")
        features = joblib.load("features.pkl")
    except (FileNotFoundError, EOFError):
        model, features = train_and_save_model(training_df)
    return model, features

def send_email_with_attachment(to_address, subject, body, attachment_path=None):
    from_address = "oumaima46590@gmail.com"
    password = "npdj mwuo xuzd sauv"

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_address, password)

    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = to_address
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    if attachment_path:
        attachment = open(attachment_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(attachment_path)}')
        msg.attach(part)

    server.send_message(msg)
    server.quit()

def save_fraud_clients_to_excel(fraud_clients):
    fraud_clients_excel = "fraud_clients.xlsx"
    fraud_clients.to_excel(fraud_clients_excel, index=False)
    return fraud_clients_excel


def send_fraud_alert(to_address, client_info):
    subject = "Alerte de Fraude: Client Détecté"
    body = f"Bonjour,\n\nUn client a été détecté comme frauduleux.\n\nDétails du client :\n{client_info}\n\nCordialement,"
    send_email_with_attachment(to_address=to_address, subject=subject, body=body)

def plot_histogram(data, column, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=30, color='blue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    st.pyplot(plt)


def plot_scatter(data, x_column, y_column, title):
    fig = px.scatter(data, x=x_column, y=y_column, color='isFraud',
                     labels={x_column: x_column, y_column: y_column},
                     title=title)
    st.plotly_chart(fig)
    

def plot_pie_chart(df, column, title):
    
    values = df[column].value_counts()
    labels = values.index
    sizes = values.values

    
    fig = px.pie(df, names=labels, values=sizes, title=title, labels={column: 'Type'})
    st.plotly_chart(fig)
    


def plot_fraud_distribution_by_type(df, type_column, fraud_column, title):
    
    fraud_transactions = df[df[fraud_column] == 1]
    
    
    fraud_counts = fraud_transactions[type_column].value_counts().reset_index()
    fraud_counts.columns = [type_column, 'Count']
    
    
    fig = px.bar(
        fraud_counts,
        x=type_column,
        y='Count',
        title=title,
        labels={type_column: 'Type de Transaction', 'Count': 'Nombre de Fraudes'},
        color=type_column
    )
    
    
    st.plotly_chart(fig)
    
    

# Interface Streamlit
st.set_page_config(page_title="Détection de Fraude", page_icon=":guardsman:", layout="wide")

def check_credentials(username, password):
    if username in users and users[username] == password:
        return True
    else:
        return False


if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False


def login():
    if st.session_state.logged_in:
        return True
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Nom d'utilisateur")
    password = st.sidebar.text_input("Mot de passe", type="password")
    if st.sidebar.button("Se connecter"):
        if check_credentials(username, password):
            st.session_state.logged_in = True
            st.sidebar.success("Vous êtes connecté avec succès!")
            return True
        else:
            st.sidebar.error("Identifiant ou mot de passe incorrect.")
            return False
    return False

users = {"admin": "admin", "Oumaima": "123"}



st.title("Application de Détection de Fraude")

training_data = pd.read_csv("dataset_transactions.csv", engine='python')


model, training_features = load_or_train_model(training_data)

if login():
    st.sidebar.empty()  

    
    option = st.sidebar.selectbox(
        "Choisissez une option",
        ["Prédiction pour un seul client", "Prédiction pour plusieurs clients"]
    )

    if option == "Prédiction pour un seul client":
        st.header("Prédiction pour un seul client")
        
        with st.form(key='client_form'):
            type_transaction = st.selectbox("Type de Transaction", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
            montant = st.number_input("Montant", value=0.0)
            numero_compte = st.text_input("Numéro de Compte")
            oldbalance_org = st.number_input("Ancien solde de l'organisateur", value=0.0)
            newbalance_orig = st.number_input("Nouveau solde de l'organisateur", value=0.0)
            oldbalance_dest = st.number_input("Ancien solde du destinataire", value=0.0)
            newbalance_dest = st.number_input("Nouveau solde du destinataire", value=0.0)
            nom_client = st.text_input("Nom du Client")
            date_creation = st.date_input("Date de Création")
        
            submit_button = st.form_submit_button(label="Prédire")
        
        if submit_button:
            data_single = pd.DataFrame({
                'typeTransaction': [type_transaction],
                'montant': [montant],
                'oldbalanceOrg': [oldbalance_org],
                'newbalanceOrig': [newbalance_orig],
                'oldbalanceDest': [oldbalance_dest],
                'newbalanceDest': [newbalance_dest],
                'NomClient': [nom_client],
                'dateCreation': [date_creation],
            })

            data_preprocessed, _ = preprocess_data(data_single, is_training=False, training_features=training_features)

            prediction = model.predict(data_preprocessed)[0]

            if prediction == 1:
                st.write("Le client est frauduleux.")
                client_info = data_single.to_string(index=False)
                to_address = "umastockage@gmail.com"
                send_fraud_alert(to_address, client_info)
                st.success(f"Une alerte de fraude a été envoyée à {to_address}.")
            else:
                st.write("Le client n'est pas frauduleux.")

    elif option == "Prédiction pour plusieurs clients":
        st.header("Prédiction pour plusieurs clients")

        uploaded_file = st.file_uploader("Téléchargez un fichier CSV pour la prédiction", type=["csv"])

        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)

            input_df_preprocessed, _ = preprocess_data(input_df, is_training=False, training_features=training_features)

            predictions = model.predict(input_df_preprocessed)

            input_df['isFraud'] = predictions

            fraud_clients = input_df[input_df['isFraud'] == 1]
            
            num_fraud_clients = fraud_clients.shape[0]
            st.write(f"Nombre de clients frauduleux détectés : {num_fraud_clients}")


            if not fraud_clients.empty:
                st.write("Clients détectés comme frauduleux :")

                
                st.dataframe(fraud_clients[['typeTransaction', 'montant', 'NumeroCompte', 'NomClient', 'dateCreation']].sort_values(by='montant', ascending=False))

                
                col1, col2 = st.columns(2)

                with col1:
                    st.write("Histogramme des Montants des Transactions Frauduleuses")
                    plot_histogram(fraud_clients, 'montant', 'Histogramme des Montants des Transactions Frauduleuses', 'Montant', 'Fréquence')

                with col2:
                    st.write("Scatter Plot des Transactions Frauduleuses")
                    plot_scatter(fraud_clients, 'montant', 'oldbalanceOrg', 'Scatter Plot des Transactions Frauduleuses')

                col3, col4 = st.columns(2)

                with col3:
                    st.write("Diagramme Circulaire des Fraudes")
                    plot_pie_chart(input_df, 'isFraud', 'Répartition des Transactions: Frauduleuses vs Non-Frauduleuses')

                with col4:
                    st.write("Répartition des Fraudes par Type de Transaction")
                    plot_fraud_distribution_by_type(input_df, 'typeTransaction', 'isFraud', 'Répartition des Fraudes par Type de Transaction')


                
                fraud_clients_excel = save_fraud_clients_to_excel(fraud_clients)
                to_address = "umastockage@gmail.com"
                subject = "Alertes de Fraude: Clients Multiples Détectés"
                body = "Bonjour,\n\nDes clients multiples ont été détectés comme frauduleux. Veuillez trouver les détails en pièce jointe.\n\nCordialement,"
                send_email_with_attachment(to_address, subject, body, fraud_clients_excel)
                st.success(f"Les résultats ont été envoyés par e-mail à {to_address}.")
            else:
                st.write("Aucun client frauduleux détecté.")
