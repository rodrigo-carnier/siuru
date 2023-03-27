# IoT anomaly detection (IoT-AD) internship

This repository contains materials and code from NII internship focused on creating a 
framework for IoT anomaly detection.

## System architecture

Below is a diagram of the core components in the IoT-AD pipeline, with arrows marking
the flow of data between components:

![System architecture diagram](graphics/iot-ad-system-overview.svg)

Elements in gray are not yet implemented. For details on the components, see the
_Repository Structure_ section below.

## Quickstart

The commands below are meant to be run on Ubuntu 20.04.

### Feature extractor

Since it is in C++, the code needs to be compiled using CMake, which can be installed
via Snap or package manager. 

In addition, the following dependencies are needed:
```bash
sudo apt install libpcap0.8 libpcap0.8-dev libpcap0.8-dbg
```

Build and install PcapPlusPlus as follows:

```bash
cd code/cpp-extract-features/PcapPlusPlus
cmake -S . -B build
cmake --build build
sudo cmake --install build
```

Then run from repository root:

```bash
mkdir cmake-build-debug && cd cmake-build-debug
cmake ..
cmake --build .
sudo setcap cap_net_raw+ep $(pwd)/pcap-feature-extraction
```

The last command is needed to give the executable permissions to listen on the network
interfaces. The path to the C++ executable is a command line argument to the main
anomaly detection program ``IoT-AD.py``. Whenever the executable is recompiled, the
permissions must also be assigned again.

### InfluxDB (only for reporting)

Install [InfluxDB](https://docs.influxdata.com/influxdb/v2.6/install/) for example as
a Docker container and follow the setup guide to create an organization and bucket.

Also create a directory where InfluxDB should store the data. In the commands below,
it is referred to as `</project/root>/influxdb`.

Start the image with:
```
docker run -p 8086:8086 \
--volume </project/root>/influxdb:/var/lib/influxdb2 \
influxdb:2.6.1 --reporting-disabled
```

From the interface that starts under http://localhost:8086 by default, generate a
token with read-write permissions to use in Grafana and when running `IoT-AD.py` with
reporting enabled.

### Grafana (only for reporting)

Install
[Grafana](https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/)
for example by following the guide for Ubuntu and Debian. Start the service:

```bash
sudo service grafana-server start
```

and follow the guide to set up an 
[InfluxDB data source](https://grafana.com/docs/grafana/latest/datasources/influxdb/).
If all goes well, Grafana should be able to connect to the InfluxDB instance you are
running.

### Python environment

I recommend to set up a Python virtual environment, e.g. pyenv. The Python
libraries needed by this project can then be installed by running from root:

```bash
cd code
pip install -r requirements.txt
```

### Training a model

Refer to the command line hints of ``code/IoT-AD.py`` for information on the available 
parameters, and the files under ``configurations/examples`` for sample configuration
files.

The example below assumes that we have stored the following:
1. data for anomaly detection under 
    `</project/root>/data/MQTTset/Data/PCAP/slowite.pcap`, 
    `</project/root>/data/MQTTset/Data/PCAP/malariaDoS.pcap` and
    `</project/root>/data/MQTTset/Data/PCAP/capture_custom_1h.pcap` (which is a custom
    segment from the full MQTTSet normal traffic file, ask me for reference)
2. built the C++ feature extractor using CMake under `</project/root>/cmake-build-debug`

As a result of successful training, we will have a random forest classifier stored under `</project/root>/models/example-flow-based-rf/flow-based-rf-train.pickle`.

```bash
cd code
python IoT-AD.py \
-c </project/root>/configurations/examples/flow-based-rf-train.json.jinja
```

### Running anomaly detection

Refer to the commandline hint of ``code/IoT-AD.py`` for information on the available 
parameters.

The sample command below assumes that we have the following:
1. data for anomaly detection similarly as for training
2. built the C++ feature extractor using CMake under `</project/root>/cmake-build-debug`
3. trained and stored the model under `</project/root>/models/example-flow-based-rf/flow-based-rf-train.pickle` (see previous section)
4. configured InfluxDB as seen below, including the generated token

```bash
cd code
python IoT-AD.py \
-c </project/root>/configurations/examples/flow-based-rf-test.json.jinja \
--influx-token <token>
```

## Repository structure

### code/cpp-extract-features

Contains the C++ feature extractor, which is very quick compared to Scapy (refer to 
``plots/benchmarking.png``).

### code/dataloaders

Contains a generic data loader interface and some implementations for dataloaders from
common datasets (MQTTSet / kaiyodai-ship, Mawi).

### code/Kafka

The Docker container that can listen on network interfaces and capture packet data.
As proof-of-concept, the following system was setup:

![System architecture diagram](graphics/kafka-pcap-demo.svg)

In the future, the container should offer access to all the IoT-AD functionalities
from this project.

### code/models

Some ML models used to test the anomaly detection pipeline.

### code/preprocessors

Packet preprocessor providing features for ML models.

### code/reporting

Reporting module sends prediction data to a logging or visualization endpoint.

In the future, this component would interface with a network controller that takes
actions based on the anomaly detection output.

### code/IoT-AD.py

The entry point to the IoT anomaly detection pipeline.

### configurations

Configuration files, which are required input for the IoT anomaly detection program.

The files are in JSON format and must define three pipeline elements: data source(s),
ML model, and output. [Jinja](https://palletsprojects.com/p/jinja/) is used to
support template variables, which the main program will replace with computed values
during runtime evaluation.

As a reference for the pipeline used to train a ML model, a copy of the processed
configuration file is stored in the same directory as the model after training.

To distinguish models by their creation date, include the `{{ timestamp }}`
template variable in the "model_name" field of the configuration file. The model name
and directory will then include a timestamp from the beginning of program execution.

### data

See README.md for references to some available datasets.

Data is automatically moved here when you run the
`code/stop_kafka_pcap.bash` script. Pcap files stored with timestamps.
While timestamps in pcap file names are in UST, packets inside have JST timestamps (+9).
