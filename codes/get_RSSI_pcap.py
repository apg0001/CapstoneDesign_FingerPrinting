import pcap
import dpkt

def packet_handler(ts, pkt):
    eth = dpkt.ethernet.Ethernet(pkt)
    if isinstance(eth.data, dpkt.ieee80211.IEEE80211):
        print(f"Captured 802.11 packet: {eth}")

sniffer = pcap.pcap(name="en0")
for timestamp, packet in sniffer:
    packet_handler(timestamp, packet)