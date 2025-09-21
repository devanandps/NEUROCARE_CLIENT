\
Client package — instructions
=============================

1) Unzip this package on the client machine where you want to run only that client's work.
2) Place your dataset (only this client's data) under the folder `CLIENT_DATA/` with structure:
   CLIENT_DATA/
     MRI/
       Alzheimers/
       Parkinsons/
       Normal/
       Tumour/
     audio/
       HC/
       PD/

3) Put your local h5 models (if you have them) inside the `models/` folder (same directory as this app):
   - models/image_model.h5
   - models/audio_model.h5  (optional)

4) Create virtualenv and install requirements:
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt

5) Run the Streamlit app locally (to preview relabel & optionally fine-tune locally):
   streamlit run client_app.py

6) To participate in federated learning rounds (connect to server), run:
   python client_image_fl.py --server <SERVER_IP>:8080 --client_id <CLIENT_ID> --local_epochs 1
   python client_audio_fl.py --server <SERVER_IP>:8081 --client_id <CLIENT_ID> --local_epochs 1

Note: clients only load data from their own CLIENT_DATA folder. The FL scripts will connect to the central server for federated averaging.

