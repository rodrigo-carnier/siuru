#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>

#include <EthLayer.h>
#include <SystemUtils.h>
#include <TcpLayer.h>
#include <cstdio>
#include <cstring>
#include <iostream>

#include "IPv4Layer.h"
#include "IpAddress.h"
#include "Packet.h"
#include "PcapFileDevice.h"
#include "PcapLiveDeviceList.h"
#include "RawPacket.h"

auto& USAGE_STR = "Usage: ./pcap-feature-extraction [function]\n\n"
                  "Functions:\n"
                  "  stream-file <abspath>  Output feature stream from file\n"
                  "  stream-device <name>   Output feature stream from device\n"
                  "  test-devices           List available devices\n"
                  "  test-file <abspath>    Open file and read one packet\n";



// Declare a queue and associated mutex and condition variable
std::queue<std::string> packetQueue;
std::mutex queueMutex;
std::condition_variable queueCondVar;


// Function to add a packet to the queue
void enqueuePacket(const std::string& packet) {
    std::lock_guard<std::mutex> lock(queueMutex);
    packetQueue.push(packet);
    queueCondVar.notify_one();
}


// Function to continuously process packets from the queue
void processPackets() {
    while (true) {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCondVar.wait(lock, [&]() {
            return !packetQueue.empty() || packetQueue.back() == "STOP_SIGNAL";
        });

        while (!packetQueue.empty()) {
            std::string packet = packetQueue.front();
            // Process the packet (e.g., output to stdout)
            std::cout << "Processed packet: " << packet << std::endl;
            packetQueue.pop();
        }

        if (!packetQueue.empty() && packetQueue.back() == "STOP_SIGNAL") {
            // Signal to stop processing
            break;
        }
    }
}


/**
 * List available network devices.
 */
int test_pcap() {
  auto& devs = pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDevicesList();
  std::cout << "Available devices:" << std::endl;
  for (auto& dev : devs) {
    std::cout << " - " << dev->getName() << std::endl;
  }
  return (0);
}




/**
 * Opens a PCAP file and tries to parse the first packet as IPv4 packet.
 *
 * @param path Absolute path to the PCAP file to open.
 */
int test_file(const char* path) {
  pcpp::PcapFileReaderDevice reader(path);
  if (!reader.open())
  {
    std::cerr << "Error opening the pcap file" << std::endl;
    return 1;
  }

  pcpp::RawPacket rawPacket;
  if (!reader.getNextPacket(rawPacket))
  {
    std::cerr << "Couldn't read the first packet in the file" << std::endl;
    return 1;
  }

  pcpp::Packet parsedPacket(&rawPacket);

  if (parsedPacket.isPacketOfType(pcpp::IPv4)) {
    pcpp::IPv4Address srcIP = parsedPacket.getLayerOfType<pcpp::IPv4Layer>()->getSrcIPv4Address();
    pcpp::IPv4Address destIP = parsedPacket.getLayerOfType<pcpp::IPv4Layer>()->getDstIPv4Address();

    std::cout << "Source IP is '" << srcIP << "'; Dest IP is '" << destIP << "'" << std::endl;
  }
  else {
    std::cout << "Not an IPv4 packet" << std::endl;
  }

  reader.close();
  return 0;
}


/**
 * Prints data extracted from the packet in format:
 * <src_ip> <dst_ip> <src_port> <dst_port> <protocol> <JSON features>\n
 *
 * JSON features:
 *    ts - packet receipt timestamp in milliseconds
 *    ip_len - IP packet length
 *    tcp_len - TCP segment length
 *    tcp_flags - 0 or 1 for [CWR, ECE, URG, ACK, PSH, RST, SYN, FIN]
 */
static void packet_to_features(pcpp::RawPacket* rawPacket, pcpp::PcapLiveDevice* dev, void* cookie) {
  pcpp::Packet packet(rawPacket);
  auto* ip_layer = packet.getLayerOfType<pcpp::IPv4Layer>();
  if (ip_layer == nullptr) {
    return;
  }

  auto* tcp_layer = packet.getLayerOfType<pcpp::TcpLayer>();
  if (tcp_layer == nullptr) {
    return;
  }

  auto src_ip = ip_layer->getSrcIPv4Address().toString();
  auto dst_ip = ip_layer->getDstIPv4Address().toString();
  auto src_port = tcp_layer->getSrcPort();
  auto dst_port = tcp_layer->getDstPort();

  auto flag_cwr = tcp_layer->getTcpHeader()->cwrFlag;
  auto flag_ece = tcp_layer->getTcpHeader()->eceFlag;
  auto flag_urg = tcp_layer->getTcpHeader()->urgFlag;
  auto flag_ack = tcp_layer->getTcpHeader()->ackFlag;
  auto flag_psh = tcp_layer->getTcpHeader()->pshFlag;
  auto flag_rst = tcp_layer->getTcpHeader()->rstFlag;
  auto flag_syn = tcp_layer->getTcpHeader()->synFlag;
  auto flag_fin = tcp_layer->getTcpHeader()->finFlag;

  auto& TCP_PROTO = "tcp";

  // Timestamp (received in nanoseconds, forwarded in milliseconds).
  auto ts_ns = rawPacket->getPacketTimeStamp().tv_sec*1000000000L + rawPacket->getPacketTimeStamp().tv_nsec;

  printf(
    "%s,%s,%d,%d,%s,%ld,%zu,%zu,%hu,%hu,%hu,%hu,%hu,%hu,%hu,%hu,%zu\n",
    src_ip.c_str(),
    dst_ip.c_str(),
    src_port,
    dst_port,
    TCP_PROTO,
    ts_ns / 1000,
    ip_layer->getHeaderLen(),
    ip_layer->getDataLen(),
    flag_cwr,
    flag_ece,
    flag_urg,
    flag_ack,
    flag_psh,
    flag_rst,
    flag_syn,
    flag_fin,
    tcp_layer->getHeaderLen()
  );
}


/**
 * Stream a PCAP file and output packet features to stdout.
 *
 * @param path Absolute path to the PCAP file.
 */
void stream_file(const char* path) {
  pcpp::PcapFileReaderDevice reader(path);
  if (!reader.open())
  {
    fprintf(stderr, "Couldn't open pcap file at %s\n", path);
    return;
  }
  pcpp::RawPacket rawPacket;
  bool hasNext = reader.getNextPacket(rawPacket);
  while (hasNext) {
    packet_to_features(&rawPacket, nullptr, nullptr);
    hasNext = reader.getNextPacket(rawPacket);
  }
}



/**
 * Stream packets from a network device continuously
 * and enqueue packets
 * TODO add packet filter as argument
 *
 * @param device_name
 */

void stream_device(const std::string& device_name, int captureDurationSeconds, int maxPackets) {
    pcpp::PcapLiveDevice* dev = pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDeviceByName(device_name);
    if (dev == nullptr) {
        std::cerr << "Cannot find interface [" << device_name << "]" << std::endl;
        return;
    }
    std::cout << "Found interface [" << device_name << "]" << std::endl;

    if (!dev->open()) {
        std::cerr << "Cannot open device" << std::endl;
        return;
    }
    std::cout << "Opened device [" << device_name << "]" << std::endl;

    auto startTime = std::chrono::steady_clock::now();
    int packetCount = 0;

    try {
        dev->startCapture([&](pcpp::RawPacket* rawPacket, pcpp::PcapLiveDevice* dev, void* cookie) {
            pcpp::Packet packet(rawPacket);
            auto packetString = packet.toString();  // Convert packet to string or use other serialization
            enqueuePacket(packetString);
            packetCount++;

            // Check if the maximum number of packets is reached
            if (maxPackets > 0 && packetCount >= maxPackets) {
                enqueuePacket("STOP_SIGNAL");  // Signal to stop capturing
            }

            // Check if the specified capture duration is reached
            auto currentTime = std::chrono::steady_clock::now();
            auto elapsedSeconds = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
            if (captureDurationSeconds > 0 && elapsedSeconds >= captureDurationSeconds) {
                enqueuePacket("STOP_SIGNAL");  // Signal to stop capturing
            }
        }, nullptr);

        // Wait for a signal to stop capturing
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCondVar.wait(lock, [&]() {
            return !packetQueue.empty() && packetQueue.back() == "STOP_SIGNAL";
        });
    }
    catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    dev->stopCapture();
}

//////////////////////////////////////////////////

int main(int argc, char *argv[]) {
    if (argc == 1) {
        std::cout << USAGE_STR;
        return 0;
    }

    for (auto i = 0; i < argc; i++) {
        if (strcmp(argv[i], "stream-file") == 0) {
            if (i == argc - 1) {
                std::cout << USAGE_STR;
                exit(1);
            }
            i++;
            stream_file(argv[i]);
            exit(0);
        }
        if (strcmp(argv[i], "test-devices") == 0) {
            test_pcap();
            exit(0);
        }
        if (strcmp(argv[i], "test-file") == 0) {
            if (i == argc - 1) {
                std::cout << USAGE_STR;
                exit(1);
            }
            i++;
            test_file(argv[i]);
            exit(0);
        }
        if (strcmp(argv[i], "stream-device") == 0) {
            if (i == argc - 1) {
                std::cout << USAGE_STR;
                exit(1);
            }
            i++;
            stream_device(argv[i]);
            exit(0);
        }
    }
}