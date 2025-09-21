import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU by default

import streamlit as st
from pathlib import Path
from PIL import Image
import shutil
import librosa
import numpy as np
import subprocess
import threading
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import plotly.express as px
from streamlit.runtime.scriptrunner import add_script_run_ctx
import dropbox, io

APP_KEY = os.environ.get("DROPBOX_APP_KEY", "yx077ued4cknfl4")
APP_SECRET = os.environ.get("DROPBOX_APP_SECRET", "nbwa3jp81lkq12c")
REFRESH_TOKEN = os.environ.get("DROPBOX_REFRESH_TOKEN", "USDPWRsHuyYAAAAAAAAAAa9iXWO_ggTnNFzLXejoKTJhE79oPs7YALwI_yEhtJoX")

dbx = dropbox.Dropbox(
    oauth2_refresh_token=REFRESH_TOKEN,
    app_key=APP_KEY,
    app_secret=APP_SECRET,
)
# -----------------------------
# LOGIN PAGE CONFIGURATION
# -----------------------------
st.set_page_config(page_title="NEUROCARENET Client Login", layout="wide")

# Dummy pre-approved accounts
PRE_APPROVED_USERS = {
    "Client1": "12345",
    "Client2": "12345",
    "Client3": "12345"
}

# Users awaiting approval (created during this session)
if "pending_users" not in st.session_state:
    st.session_state.pending_users = {}

# Login state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------
# LOGIN / SIGNUP / FORGOT PASSWORD PAGES
# -----------------------------
if not st.session_state.logged_in:
    st.image("image.png", use_container_width=False, width=300)
    st.title("NEUROCARENET Login")
    
    auth_option = st.radio("Select Option", ["Login", "Create Account", "Forgot Password"])

    if auth_option == "Login":
        username_or_email = st.text_input("Username or Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username_or_email in PRE_APPROVED_USERS and PRE_APPROVED_USERS[username_or_email] == password:
                st.success("Login successful!")
                st.session_state.logged_in = True
            elif username_or_email in st.session_state.pending_users:
                st.warning("Account pending approval. Please wait for admin verification.")
            else:
                st.error("Invalid username or password.")

    elif auth_option == "Create Account":
        st.subheader("Create a New Account (Admin Approval Required)")
        new_username = st.text_input("Username")
        reg_number = st.text_input("Registration Number")
        email = st.text_input("Email")
        verify_email = st.text_input("Verify Email")
        password = st.text_input("Password", type="password")
        verify_password = st.text_input("Verify Password", type="password")

        if st.button("Submit Account Request"):
            if not (new_username and reg_number and email and verify_email and password and verify_password):
                st.error("All fields are required.")
            elif email != verify_email:
                st.error("Emails do not match.")
            elif password != verify_password:
                st.error("Passwords do not match.")
            elif new_username in PRE_APPROVED_USERS or new_username in st.session_state.pending_users:
                st.error("Username already exists.")
            else:
                st.session_state.pending_users[new_username] = password
                st.info("Account created successfully! Awaiting admin approval. Email will be sent upon verification.")

    elif auth_option == "Forgot Password":
        st.subheader("Forgot Password")
        email = st.text_input("Enter your verified email address")
        if st.button("Send Reset Link"):
            if email:
                st.info(f"A password reset link will be sent to {email} .")
            else:
                st.warning("Please enter your email address.")

# -----------------------------
# MAIN APP: Only accessible if logged in
# -----------------------------
if st.session_state.logged_in:
    st.title("Welcome to NEUROCARENET Client Dashboard")
    # -------------------------------------------------
    # Dropbox Setup
    # -------------------------------------------------

    DROPBOX_CSV_PATH = "/patient_dataset2.csv"      # CSV location in Dropbox

    def load_csv_from_dropbox():
        try:
            metadata, res = dbx.files_download(DROPBOX_CSV_PATH)
            return pd.read_csv(io.BytesIO(res.content))
        except dropbox.exceptions.ApiError as e:
            st.error(f"Could not load CSV from Dropbox: {e}")
            return pd.DataFrame()

    def save_csv_to_dropbox(df: pd.DataFrame):
        try:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            dbx.files_upload(
                csv_buffer.getvalue().encode(),
                DROPBOX_CSV_PATH,
                mode=dropbox.files.WriteMode("overwrite")
            )
        except Exception as e:
            st.error(f"Could not save CSV to Dropbox: {e}")


    # -------------------------------------------------
    # Streamlit Config
    # -------------------------------------------------
    st.set_page_config(page_title='Federated Client (Dropbox CSV)', layout='wide')

    BASE = Path.cwd()
    DATA_DIR = BASE / "CLIENT_DATA"    # client dataset
    MODELS_DIR = BASE / "models"       # models folder
    MODELS_DIR.mkdir(exist_ok=True)
    IMAGE_MODEL_FILE = MODELS_DIR / "image_model.h5"
    AUDIO_MODEL_FILE = MODELS_DIR / "audio_model.h5"

    st.title("Federated Client — Dropbox CSV Integration")

    # -------------------------------------------------
    # Sidebar: Client Configuration + Page Navigation
    # -------------------------------------------------
    if "client_id" not in st.session_state:
        st.session_state.client_id = ""
    st.sidebar.header("Client configuration")
    st.session_state.client_id = st.sidebar.text_input("Client ID (e.g. client1)", value=st.session_state.client_id)
    SERVER_IP = st.sidebar.text_input("Server IP", value="172.16.70.233")
    IMAGE_PORT = st.sidebar.number_input("Image Server Port", value=8080)
    AUDIO_PORT = st.sidebar.number_input("Audio Server Port", value=8081)
    BM_PORT = st.sidebar.number_input("Biomarkers Server Port", value=8082)

    if not st.session_state.client_id:
        st.warning("Enter Client ID in the sidebar to proceed.")
        st.stop()

    page = st.sidebar.radio("Select Page", ["Client Data Verification", "MRI Preview", "Audio Preview", "Local Training / FL"])

    client_folder = DATA_DIR
    mri_path = client_folder / "MRI"
    audio_path = client_folder / "audio"

    # -------------------------------------------------
    # Utility: Build Image Model
    # -------------------------------------------------
    def build_image_model(num_classes):
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
        base.trainable = False
        inputs = Input(shape=(224,224,3))
        x = base(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # =================================================
    # PAGE 1: Client Data Verification
    # =================================================
    if page == "Client Data Verification":
        st.header("Dropbox Client Data Verification")

        df = load_csv_from_dropbox()
        if not df.empty:
            try:
                if "patient_id" not in df.columns:
                    st.error("CSV file must have a 'patient_id' column.")
                else:
                    patient_id = st.selectbox("Select Patient ID", df["patient_id"].unique())
                    patient_data = df[df["patient_id"] == patient_id].iloc[0]
                    st.subheader(f"Details for Patient ID: {patient_id}")

                    # ---------------- Feature Table ----------------
                    video_ids = [f"F{str(i).zfill(3)}" for i in range(1,21)]
                    audio_ids = [f"F{str(i).zfill(3)}" for i in range(21,41)]
                    clinician_ids = [f"F{str(i).zfill(3)}" for i in range(41,81)]

                    video_labels = [
                        "Facial expression (hypomimia)", "Blink rate", "Resting tremor", "Postural tremor",
                        "Bradykinesia (finger tapping)", "Bradykinesia (hand opening/closing)", "Arm swing reduction",
                        "Stride length", "Gait variability", "Freezing of gait", "Postural instability",
                        "Turn hesitation", "Facial asymmetry", "Drooling visible", "Head tremor", "Jaw tremor",
                        "Eye movements abnormal", "Saccade latency", "Stooped posture", "Falls observed"
                    ]

                    audio_labels = [
                        "Speech rate", "Articulation precision", "Pause frequency", "Pause duration", "Phonation ratio",
                        "Fundamental frequency variability", "Voice intensity variability", "Monotonicity", "Hypophonia",
                        "Slurred speech", "Word-finding pauses", "Naming latency", "Lexical diversity", "Semantic errors",
                        "Phonemic errors", "Discourse coherence", "Sentence repetition", "Story recall (immediate)",
                        "Story recall (delayed)", "Emotional prosody recognition"
                    ]

                    clinician_labels = [
                        "MoCA score", "MMSE score", "Delayed recall impairment", "Recognition memory errors",
                        "Clock drawing errors", "Trails B completion time", "Stroop interference errors",
                        "Set-shifting difficulty", "Planning impairment", "Orientation deficit", "IADL decline",
                        "ADL decline", "Apathy scale", "Depression scale", "REM sleep behavior disorder", "Hyposmia",
                        "Orthostatic hypotension", "Constipation", "Micrographia", "Rigidity on exam", "Tremor on exam",
                        "Hemiparesis on exam", "Aphasia on exam", "Dysarthria on exam", "Acute onset history",
                        "Carotid bruit", "Hypertension", "Diabetes", "Hyperlipidemia", "Atrial fibrillation",
                        "Prior TIA/stroke", "MRI white-matter lesions", "MRI lacunes", "Medial temporal atrophy",
                        "Small vessel disease burden", "Visuospatial neglect", "Line bisection error", "Cube copy errors",
                        "Frontal release signs", "Lewy body fluctuation history"
                    ]

                    all_ids = video_ids + audio_ids + clinician_ids
                    all_labels = video_labels + audio_labels + clinician_labels
                    values = [patient_data.get(fid, np.nan) for fid in all_ids]

                    feature_table = pd.DataFrame({
                        "FeatureID": all_ids,
                        "FeatureName": all_labels,
                        "Value": values
                    })
                    st.subheader("Feature Values Table")
                    st.dataframe(feature_table)

                    # ---------------- Video Features Bar Chart ----------------
                    fig_video = px.bar(
                        x=video_labels,
                        y=patient_data[video_ids].values,
                        color=video_labels,
                        labels={"x":"Feature", "y":"Score"},
                        title="Video-based Features"
                    )
                    st.plotly_chart(fig_video, use_container_width=True)

                    # ---------------- Audio Features Bar Chart ----------------
                    fig_audio = px.bar(
                        x=audio_labels,
                        y=patient_data[audio_ids].values,
                        color=audio_labels,
                        labels={"x":"Feature", "y":"Score"},
                        title="Audio-based Features"
                    )
                    st.plotly_chart(fig_audio, use_container_width=True)

                    # ---------------- Clinician Features Bar Chart ----------------
                    fig_clinician = px.bar(
                        x=clinician_labels,
                        y=patient_data[clinician_ids].values,
                        color=clinician_labels,
                        labels={"x":"Feature", "y":"Score"},
                        title="Clinician/Reported Biomarkers"
                    )
                    st.plotly_chart(fig_clinician, use_container_width=True)

                    # ---------------- Predicted Probabilities Pie Chart ----------------
                    prob_labels = ["Normal", "AD", "PD", "Stroke", "Tumour"]
                    prob_values = [
                        patient_data.get("P_Normal", 0),
                        patient_data.get("P_AD", 0),
                        patient_data.get("P_PD", 0),
                        patient_data.get("P_Stroke", 0),
                        patient_data.get("P_Tumour", 0)
                    ]
                    fig_prob = px.pie(
                        names=prob_labels,
                        values=prob_values,
                        title="Predicted Probabilities (Model Output)"
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

                    # ---------------- Description ----------------
                    most_likely_idx = np.argmax(prob_values)
                    description = f"The model predicts the patient is most likely: {prob_labels[most_likely_idx]} with probability {prob_values[most_likely_idx]:.2f}."
                    st.info(description)

                    # ---------------- Doctor Verification ----------------
                    st.subheader("Doctor Verification / Input")
                    st.write("Enter your assessment percentages for each category:")

                    default_vals = {
                        "D_Normal": patient_data.get("D_Normal", prob_values[0]),
                        "D_AD": patient_data.get("D_AD", prob_values[1]),
                        "D_PD": patient_data.get("D_PD", prob_values[2]),
                        "D_Stroke": patient_data.get("D_Stroke", prob_values[3]),
                        "D_Tumour": patient_data.get("D_Tumour", prob_values[4]),
                    }

                    col1, col2, col3, col4, col5 = st.columns(5)
                    D_Normal = col1.number_input("Normal (%)", min_value=0.0, max_value=100.0, value=float(default_vals["D_Normal"]))
                    D_AD = col2.number_input("AD (%)", min_value=0.0, max_value=100.0, value=float(default_vals["D_AD"]))
                    D_PD = col3.number_input("PD (%)", min_value=0.0, max_value=100.0, value=float(default_vals["D_PD"]))
                    D_Stroke = col4.number_input("Stroke (%)", min_value=0.0, max_value=100.0, value=float(default_vals["D_Stroke"]))
                    D_Tumour = col5.number_input("Tumour (%)", min_value=0.0, max_value=100.0, value=float(default_vals["D_Tumour"]))

                    if st.button("Confirm Doctor Diagnosis"):
                        df.loc[df["patient_id"] == patient_id, "D_Normal"] = D_Normal
                        df.loc[df["patient_id"] == patient_id, "D_AD"] = D_AD
                        df.loc[df["patient_id"] == patient_id, "D_PD"] = D_PD
                        df.loc[df["patient_id"] == patient_id, "D_Stroke"] = D_Stroke
                        df.loc[df["patient_id"] == patient_id, "D_Tumour"] = D_Tumour
                        save_csv_to_dropbox(df)
                        st.success("Doctor diagnosis saved successfully!")

                        doctor_values = [D_Normal, D_AD, D_PD, D_Stroke, D_Tumour]
                        fig_doctor = px.pie(
                            names=prob_labels,
                            values=doctor_values,
                            title="Doctor Verified Probabilities"
                        )
                        st.plotly_chart(fig_doctor, use_container_width=True)

                    # ---------------- Doctor Comments ----------------
                    st.subheader("Doctor Comments / Notes")
                    default_comment = patient_data.get("COMMENT", "")
                    comment_text = st.text_area("Enter your comments on the report:", value=default_comment, height=120)

                    if st.button("Save Comments"):
                        df.loc[df["patient_id"] == patient_id, "COMMENT"] = comment_text
                        save_csv_to_dropbox(df)
                        st.success("Doctor comments saved successfully!")

            except Exception as e:
                st.error("Error processing Dropbox CSV: " + str(e))
        else:
            st.info("No patient_data.csv found in Dropbox. Please upload it first.")

    # =================================================
    # PAGE 2, 3, 4 remain same (MRI Preview, Audio Preview, Local Training / FL)
    # =================================================

    # =================================================
    # PAGE 2, 3, 4 remain unchanged
    # =================================================
    # MRI Preview, Audio Preview, Local Training / FL sections remain as you provided above


    # =================================================
    # PAGE 2: MRI Preview
    # =================================================
    elif page == "MRI Preview":
        st.header(f"Client: {st.session_state.client_id} — MRI Dataset Preview")
        if mri_path.exists():
            st.subheader("MRI images")
            for label_folder in sorted([p for p in mri_path.iterdir() if p.is_dir()]):
                st.write(f"Class: {label_folder.name}")
                for idx, img_file in enumerate(sorted(label_folder.glob('*'))):
                    try:
                        img = Image.open(img_file)
                        st.image(img, caption=img_file.name, width=300)
                    except Exception as e:
                        st.write("Could not open image", img_file.name, e)
                        continue
                    options = ["Keep current"] + [p.name for p in mri_path.iterdir() if p.is_dir()]
                    key = f"img_{label_folder.name}_{img_file.name}_{idx}"   # unique key
                    new_label = st.selectbox(f"Relabel {img_file.name}", options=options, key=key)
                    if new_label != "Keep current":
                        dest = mri_path / new_label / img_file.name
                        shutil.move(str(img_file), str(dest))
                        st.rerun()
        else:
            st.info("No MRI folder found. Create CLIENT_DATA/MRI/<class>/ and add images.")

    # =================================================
    # PAGE 3: Audio Preview
    # =================================================
    elif page == "Audio Preview":
        st.header(f"Client: {st.session_state.client_id} — Audio Dataset Preview")
        if audio_path.exists():
            st.subheader("Audio recordings")
            for label_folder in sorted([p for p in audio_path.iterdir() if p.is_dir()]):
                st.write(f"Class: {label_folder.name}")
                for idx, audio_file in enumerate(sorted(label_folder.glob('*'))):
                    st.write(audio_file.name)
                    try:
                        st.audio(str(audio_file), format='audio/wav')
                    except Exception as e:
                        st.write("Could not play audio:", e)
                    options = ["Keep current"] + [p.name for p in audio_path.iterdir() if p.is_dir()]
                    key = f"aud_{label_folder.name}_{audio_file.name}_{idx}"  # unique key
                    new_label = st.selectbox(f"Relabel {audio_file.name}", options=options, key=key)
                    if new_label != "Keep current":
                        dest = audio_path / new_label / audio_file.name
                        shutil.move(str(audio_file), str(dest))
                        st.rerun()
        else:
            st.info("No audio folder found. Create CLIENT_DATA/audio/<class>/ and add .wav files.")

    # =================================================
    # PAGE 4: Local Training / Federated Learning
    # =================================================
    # =================================================
    # PAGE 4: Local Training / Federated Learning
    # =================================================
    elif page == "Local Training / FL":
        st.header("Local Training Options")

        st.markdown("---")
        st.subheader("Choose what to fine-tune locally:")

        col1, col2, col3 = st.columns(3)

        # ---------------- Image Fine-Tune ----------------
        with col1:
            if st.button("Fine-tune Image Model"):
                if mri_path.exists():
                    try:
                        st.info("Starting image model fine-tune...")
                        train_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
                            mri_path, target_size=(224,224), batch_size=4, class_mode='categorical', shuffle=True
                        )
                        num_classes = len(train_gen.class_indices)
                        if IMAGE_MODEL_FILE.exists():
                            try:
                                image_model = load_model(str(IMAGE_MODEL_FILE))
                            except Exception:
                                image_model = build_image_model(num_classes)
                                image_model.load_weights(str(IMAGE_MODEL_FILE))
                        else:
                            image_model = build_image_model(num_classes)
                        image_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
                        image_model.fit(train_gen, epochs=1)
                        save_path = MODELS_DIR / f"image_model_{st.session_state.client_id}.keras"
                        image_model.save(save_path)
                        st.success(f"Image model fine-tuned and saved to {save_path.name}")
                    except Exception as e:
                        st.error("Image training failed: " + str(e))
                else:
                    st.warning("MRI folder not found for this client.")

        # ---------------- Audio Fine-Tune ----------------
        with col2:
            if st.button("Fine-tune Audio Model"):
                if audio_path.exists():
                    try:
                        st.info("Starting audio model fine-tune...")
                        X, Y = [], []
                        classes = sorted([p.name for p in audio_path.iterdir() if p.is_dir()])
                        cls2idx = {c:i for i,c in enumerate(classes)}
                        for folder in audio_path.iterdir():
                            if not folder.is_dir(): continue
                            for wav in folder.glob('*'):
                                y, sr = librosa.load(wav, sr=22050)
                                spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                                spec_db = librosa.power_to_db(spec, ref=np.max)
                                if spec_db.shape[1] < 44:
                                    spec_db = np.pad(spec_db, ((0,0),(0,44-spec_db.shape[1])), mode='constant')
                                else:
                                    spec_db = spec_db[:, :44]
                                spec_db = np.expand_dims(spec_db, -1)
                                X.append(spec_db)
                                Y.append(cls2idx[folder.name])
                        X = np.array(X)
                        Y = to_categorical(np.array(Y), num_classes=len(classes))
                        if X.size > 0:
                            from tensorflow.keras.models import Sequential
                            from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
                            audio_model = Sequential([
                                Conv2D(16, (3,3), activation='relu', input_shape=X.shape[1:]),
                                MaxPooling2D((2,2)),
                                Conv2D(32, (3,3), activation='relu'),
                                MaxPooling2D((2,2)),
                                Flatten(),
                                Dense(128, activation='relu'),
                                Dropout(0.3),
                                Dense(len(classes), activation='softmax')
                            ])
                            audio_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
                            audio_model.fit(X, Y, epochs=1, batch_size=2)
                            save_path = MODELS_DIR / f"audio_model_{st.session_state.client_id}.keras"
                            audio_model.save(save_path)
                            st.success(f"Audio model fine-tuned and saved to {save_path.name}")
                        else:
                            st.warning("No audio data found for training.")
                    except Exception as e:
                        st.error("Audio training failed: " + str(e))
                else:
                    st.warning("Audio folder not found for this client.")

        # ---------------- Early Symptoms Fine-Tune ----------------
        with col3:
            if st.button("Fine-tune Early Symptoms Model"):
                import torch
                import torch.nn as nn
                import torch.optim as optim

                CLIENT_DATA_PATH = BASE / "CLIENT_DATA" / "es" / "patient_data.csv"
                MODEL_SAVE_PATH = MODELS_DIR / f"early_symptoms_model_{st.session_state.client_id}.pth"
                MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

                if not CLIENT_DATA_PATH.exists():
                    st.error(f"CSV not found: {CLIENT_DATA_PATH}")
                else:
                    st.info("Starting tabular early symptoms model fine-tune...")
                    df = pd.read_csv(CLIENT_DATA_PATH)
                    feature_cols = [c for c in df.columns if c.startswith("F")]
                    X = df[feature_cols].values.astype(np.float32)

                    if "Prediction_Label" in df.columns:
                        Y = df["Prediction_Label"].values
                    else:
                        st.error("CSV must contain Prediction_Label column")
                        st.stop()

                    # Encode labels
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    Y = le.fit_transform(Y)
                    Y = Y.astype(np.int64)

                    # Normalize
                    X = np.nan_to_num(X, nan=0.0)
                    max_val = np.max(X)
                    if max_val > 0: X = X / max_val

                    num_classes = len(np.unique(Y))
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    class TabularNN(nn.Module):
                        def __init__(self, input_dim, num_classes):
                            super().__init__()
                            self.net = nn.Sequential(
                                nn.Linear(input_dim, 128),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(128, 64),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(64, num_classes)
                            )
                        def forward(self, x):
                            return self.net(x)

                    model = TabularNN(input_dim=X.shape[1], num_classes=num_classes).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=1e-3)

                    epochs = 3
                    batch_size = 16
                    idx = np.arange(len(X))
                    for epoch in range(epochs):
                        np.random.shuffle(idx)
                        for i in range(0, len(idx), batch_size):
                            batch_idx = idx[i:i+batch_size]
                            xb = torch.tensor(X[batch_idx], dtype=torch.float32).to(device)
                            yb = torch.tensor(Y[batch_idx], dtype=torch.long).to(device)
                            optimizer.zero_grad()
                            out = model(xb)
                            loss = criterion(out, yb)
                            loss.backward()
                            optimizer.step()
                        #st.write(f"Epoch {epoch+1}/{epochs} done, last loss: {loss.item():.4f}")

                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    st.success(f"Early symptoms model fine-tuned and saved: {MODEL_SAVE_PATH.name}")

        # ---------------- Federated Learning GUI ----------------
        st.markdown("---")
        st.header("Federated Learning — Run from GUI")

        if "fl_logs" not in st.session_state:
            st.session_state.fl_logs = ""
        log_placeholder = st.empty()
        st.text_area("Federated Learning Logs", value=st.session_state.fl_logs, height=420, key="fl_log_area")

        if "fl_process" not in st.session_state:
            st.session_state.fl_process = None

        def run_fl_and_write_logs(cmd):
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                st.session_state.fl_process = proc
                for line in proc.stdout:
                    st.session_state.fl_logs += line
                    try:
                        log_placeholder.text(st.session_state.fl_logs)
                    except Exception:
                        pass
                proc.stdout.close()
                proc.wait()
                st.session_state.fl_logs += f"\n[Process exited with code {proc.returncode}]\n"
                try:
                    log_placeholder.text(st.session_state.fl_logs)
                except Exception:
                    pass
            except Exception as e:
                st.session_state.fl_logs += f"\n[FL thread exception] {e}\n"
                try:
                    log_placeholder.text(st.session_state.fl_logs)
                except Exception:
                    pass
            finally:
                st.session_state.fl_process = None

        # Buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Start FL (Image)"):
                st.session_state.fl_logs = ""
                cmd = [
                    "python3", "client_image_fl.py",
                    "--server", f"{SERVER_IP}:{IMAGE_PORT}",
                    "--client_id", st.session_state.client_id,
                    "--local_epochs", "1"
                ]
                thread = threading.Thread(target=run_fl_and_write_logs, args=(cmd,), daemon=True)
                add_script_run_ctx(thread)
                thread.start()

        with col2:
            if st.button("Start FL (Audio)"):
                st.session_state.fl_logs = ""
                cmd = [
                    "python3", "client_audio_fl.py",
                    "--server", f"{SERVER_IP}:{AUDIO_PORT}",
                    "--client_id", st.session_state.client_id,
                    "--local_epochs", "1"
                ]
                thread = threading.Thread(target=run_fl_and_write_logs, args=(cmd,), daemon=True)
                add_script_run_ctx(thread)
                thread.start()

        with col3:
            if st.button("Start FL (Biomarkers/Early symptoms)"):
                st.session_state.fl_logs = ""
                cmd = [
                    "python3", "client_es.py",
                    "--server", f"{SERVER_IP}:8082",
                    "--client_id", st.session_state.client_id,
                    "--csv_file", "patient_data.csv",
                    "--local_epochs", "1"
                ]
                thread = threading.Thread(target=run_fl_and_write_logs, args=(cmd,), daemon=True)
                add_script_run_ctx(thread)
                thread.start()

