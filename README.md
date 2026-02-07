
## How to Run 

### 1 Start Ilúvatar and get the server IP
Run **Ilúvatar** (serverless inference platform) and note the server IP address or hostname where it is running.

- Ilúvatar repository:  
https://github.com/COS-IN/iluvatar-faas

---

### 2 Set the Ilúvatar server address in the config
Update the following configuration file:

```text
codespace/ckn_controller/ckn_config.py
```

### Set the server address
Update the server address in the configuration file:

```python
SERVER_ADDRESS = "xxx.xxx.xxx.xx:8079"
```
Replace xxx.xxx.xxx.xx with Ilúvatar server IP.

### Register functions
Register the built functions with Ilúvatar using the following command:

```bash
python3 register_function_rpc.py
```

### 3 Run the CKN service
Start the CKN service from the project root:

```bash
python ckn_service.py

```


### 4 Configure the workload generator (deadline and arrival interval)
Edit the workload generator configuration file:

```text
codespace/workload_generator/generator.py
```
Update the following parameters as needed:

```python
default_deadline_ms = 500
default_air_ms = 100

```


### 5 Run the workload generator
Navigate to the workload generator directory:

```text
codespace/workload_generator
```
Run the workload generator: 

```bash
python main.py
```
