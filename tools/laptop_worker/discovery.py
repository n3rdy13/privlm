"""Finding PrivateLM devices on the Wi-Fi (Bonjour / mDNS).

Two ways, because Wi-Fi networks differ:

- browse(): the usual multicast search ("Find nearby"). Names are collected
  while browsing and looked up afterwards: looking one up inside zeroconf's
  callback blocks its thread, and with two phones the second was missed.
- query_host(): asks one device directly at its IP address (a unicast mDNS
  query to port 5353). Phones answer it even on networks that drop
  multicast between some devices (2.4 vs 5 GHz, mesh nodes, guest
  isolation), which is when browse() finds only some of them.

Each result is {"instance", "host", "port", "props"}.
"""

import socket
import time
from typing import Callable, Dict, List, Optional

LogFn = Optional[Callable[[str], None]]


def _txt_props(data: bytes) -> Dict[str, str]:
    """TXT record bytes (length-prefixed key=value strings) to a dict."""
    props, i = {}, 0
    while i < len(data):
        n = data[i]
        item = data[i + 1:i + 1 + n].decode("utf-8", "replace")
        i += 1 + n
        if "=" in item:
            key, value = item.split("=", 1)
            props[key] = value
    return props


def browse(service_type: str, seconds: float = 4.0, log: LogFn = None) -> List[dict]:
    from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, Zeroconf  # type: ignore

    names = set()

    class Listener:
        def add_service(self, zc, type_, name):
            names.add(name)

        def update_service(self, zc, type_, name):
            names.add(name)

        def remove_service(self, zc, type_, name):
            pass

    zc = Zeroconf(ip_version=IPVersion.V4Only)
    results = []
    try:
        browser = ServiceBrowser(zc, service_type, Listener())
        time.sleep(seconds)
        browser.cancel()
        for name in sorted(names):
            info = ServiceInfo(service_type, name)
            if not info.request(zc, 3000):
                if log:
                    log(f"Found {name} but it didn't answer with its address")
                continue
            host = next((a for a in info.parsed_addresses() if ":" not in a), None)
            if not host:
                if log:
                    log(f"Found {name} but it has no IPv4 address")
                continue
            props = {(k.decode() if isinstance(k, bytes) else k): (v.decode("utf-8", "replace") if v else "")
                     for k, v in info.properties.items()}
            results.append({"instance": name, "host": host, "port": info.port, "props": props})
    finally:
        zc.close()
    if log:
        log(f"Find nearby: {len(results)} found ({', '.join(r['host'] for r in results) or 'none'})")
    return results


def query_host(host: str, service_type: str, timeout: float = 2.0, log: LogFn = None) -> List[dict]:
    """Asks the device at `host` for its `service_type` services directly."""
    from zeroconf import DNSIncoming, DNSOutgoing, DNSQuestion, const  # type: ignore

    def ask(sock, name: str, rtype: int):
        out = DNSOutgoing(const._FLAGS_QR_QUERY, False)
        out.add_question(DNSQuestion(name, rtype, const._CLASS_IN))
        for packet in out.packets():
            sock.sendto(packet, (host, 5353))

    records = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        # From an ordinary port: responders answer such "legacy" queries
        # straight back to us instead of on the multicast group.
        sock.bind(("", 0))
        sock.settimeout(0.3)

        def collect(deadline):
            while time.time() < deadline:
                try:
                    data, addr = sock.recvfrom(9000)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if addr[0] != host:
                    continue
                msg = DNSIncoming(data)
                if msg.valid and msg.is_response():
                    records.extend(msg.answers())

        try:
            ask(sock, service_type, const._TYPE_PTR)
        except OSError as e:
            if log:
                log(f"Couldn't ask {host}: {e}")
            return []
        collect(time.time() + timeout)

        instances = {r.alias for r in records if r.type == const._TYPE_PTR and r.name == service_type}
        # Ask for anything the first answer left out.
        missing = False
        for inst in instances:
            have = {r.type for r in records if r.name == inst}
            for rtype in (const._TYPE_SRV, const._TYPE_TXT):
                if rtype not in have:
                    ask(sock, inst, rtype)
                    missing = True
        if missing:
            collect(time.time() + timeout)

    results = []
    for inst in sorted(instances):
        srv = next((r for r in records if r.type == const._TYPE_SRV and r.name == inst), None)
        txt = next((r for r in records if r.type == const._TYPE_TXT and r.name == inst), None)
        if not srv:
            continue
        results.append({"instance": inst, "host": host, "port": srv.port,
                        "props": _txt_props(txt.text) if txt else {}})
    if log:
        log(f"Asked {host} directly: {len(results)} service(s)")
    return results
