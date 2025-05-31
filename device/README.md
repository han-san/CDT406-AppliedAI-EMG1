# Running model on BeagleBone Green
Files to run the model

## Prerequisites
- `libiio`
- `pylibiio`
- `systemd`

## Files
- `adc/adc.py` - continious collection of ADC readings
- `adc.service` - `systemd` service descriptor for ADC reading script
- `inference_res.py` - logging of inference results
- `model.service` - `systemd` service descriptor of the LSTM model
- `run.sh` - run to start the model

## Initial setup
### Prerequisites
- `EMG1` user with sudo privileges
### Setup
1. Create `setup` directory in `EMG1`'s home directory
```
$ su - EMG1
$ mkdir ~/setup
```
2. Copy `src` directory to `setup`
```
$ cp /path/to/repo/src ~/setup/
```
3. Install ADC service
```
# cp adc/adc.service /etc/systemd/system/
```
4. Install model service
```
$ cp model.service /home/EMG1/.config/systemd/user/
```
5. Move the `.tflite` model file to `setup`. Name it `model.tflite`
```
$ mv /path/to/model ~/setup/model.tflite
```

### File tree
```
setup
├── adc.py
├── inference_res.py
├── run.sh
├── model.tflite
└── src
```

## Running the model
To run the model execute
```
$ ./run.sh
```
To stop running the model do `Ctrl-C` (send the `SIGINT` signal). After the model is terminated you will see two log files:
- `model.log` - response times, `Waiting` - waiting and inference, `Inference` - only inference (preprocessing included)
- `states.log` - labeled voltages