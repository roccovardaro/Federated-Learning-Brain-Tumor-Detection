# 🧠 Federated Learning for Brain Tumor Detection

This project implements a Federated Learning (FL) system for brain tumor detection using MRI data. Built with the **Flower** framework, it simulates a decentralized environment in which clients train a shared global model locally without sharing sensitive data.

---

## 📁 Project Structure

```
.
├── data/                         # Input datasets
│   ├── brain_tumor_dataset/
│   ├── brain_tumor_dataset2/
│   └── data_test_server/
│
├── outputs/                      # Outputs per date and time
│   
│
├── src/
│   ├── conf/
│   │   └── config_server.yaml   # Central config file
│   ├── dataframe_history/       # CSV logs per client
│   ├── model/                   # Model definitions and test
│   │   └── trained_models/      # Saved models
│   └── utils/                   # Server/client and utilities
│       ├── server.py
│       ├── client.py
│       └── customStrategy.py
│
├── start_server.ps1             # Launch server (PowerShell)
├── start_client.ps1             # Launch one client
└── start_clients.ps1            # Launch multiple clients
```

---

## ⚙️ Requirements

- Python 3.8+
- Packages:
  - `flwr`
  - `tensorflow`
  - `numpy`, `pandas`, `matplotlib`

To install dependencies:
```bash
pip install -r requirements.txt
```
---

## 🚀 How to Run

### 1. Configure

Edit `src/conf/config_server.yaml`:
```yaml
num_rounds: 5 # number of FL rounds in the experiment
num_clients: 5  # number of total clients available (this is also the number of partitions we need to create)
num_clients_per_round_fit: 5 
batch_size: 20 # batch size to use by clients during training
fraction_fit: 0.3 # number of clients to involve in each fit round (fit  round = clients receive the trained_models from the server and do local training)
...
```

### 2. Launch the Server
```powershell
./start_server.ps1
```

### 3. Launch Clients
```powershell
./start_clients.ps1      # Launch N clients automatically
# or
./start_client.ps1 -id 1 # Launch a single client
```
---

## 🧠 Federated Learning Logic

- **Clients** receive the global model, perform local training, and return weights.
- **Server** aggregates weights using a `Strategy`  and updates the global model.

---

## 🔧 Custom Strategy – `CustomFedAvg`

This project defines a **custom strategy class** that extends Flower’s default `FedAvg` strategy to enhance functionality.

📄 **File**: `src/utils/customStrategy.py`

### 🔍 Key Features

- ✅ Inherits from `fl.server.strategy.FedAvg`
- ✅ Saves the **final global model** automatically at the end of the last round
- ✅ Uses your `create_modelCNN()` architecture and saves the trained model:

---

## 📊 Outputs

- 📈 PNG plots per client and global training
- 📁 CSV logs in `dataframe_history/`
- 💾 Trained models in `trained_models/`
- 🗃️ Output organized by date/time in `outputs/`

---

## 🔒 Privacy & Security

✔️ Data never leaves the local device  
✔️ Only model weights/gradients are exchanged  
🔐 Extension possible with:
- Differential Privacy
- Secure Aggregation (MPC)
- Homomorphic Encryption

---

## 🧪 Dataset

The system uses anonymized MRI datasets for binary classification:
- `brain_tumor_dataset`
- `brain_tumor_dataset2`

Data is split **non-IID** across clients to simulate real-world conditions.

---

## 🔁 Extendability

- Add new strategies: `FedProx`, `FedAdam`, `QFedAvg`, etc.
- Simulate mobile/IoT devices
- Integrate with TensorBoard for live monitoring
- Add web-based dashboard

---

## 📚 References

📄 Thesis: *Experimental Analysis of Federated Learning Applied to Medical Diagnosis*  
👨‍🎓 Author: Rocco Pio Vardaro – University of Calabria  
🔗 [Flower Documentation](https://flower.dev)

---

## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.

---

## 🧑‍💻 Author

**Rocco Pio Vardaro**  


