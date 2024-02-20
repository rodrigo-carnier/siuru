from scapy.all import sniff

def packet_callback(packet):
    # Callback function to process each captured packet
    print(packet.show())

# Capture packets on a specific interface (e.g., 'eth0') with a given callback
sniff(iface='wlp0s20f3', prn=packet_callback, store=0)