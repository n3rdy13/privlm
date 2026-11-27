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
MDNS_GROUP = "224.0.0.251"


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


def _legacy_query(dest: str, service_type: str, timeout: float, log: LogFn,
                  only_from: Optional[str] = None) -> List[dict]:
    """Sends a PTR query to `dest` (a device, or the mDNS group) from an
    ordinary port. Responders answer such "legacy" queries straight back to
    that port, so the answers can't be taken by another program on port 5353
    (on Windows, its own mDNS service shares that port and can swallow them)."""
    from zeroconf import DNSIncoming, DNSOutgoing, DNSQuestion, const  # type: ignore

    records = []   # (source address, record)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("", 0))
        sock.settimeout(0.3)
        if dest == MDNS_GROUP:
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 255)
            local = lan_address()
            if local != "127.0.0.1":
                try:
                    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(local))
                except OSError:
                    pass

        def ask(to: str, name: str, rtype: int):
            out = DNSOutgoing(const._FLAGS_QR_QUERY, False)
            out.add_question(DNSQuestion(name, rtype, const._CLASS_IN))
            for packet in out.packets():
                sock.sendto(packet, (to, 5353))

        def collect(deadline):
            while time.time() < deadline:
                try:
                    data, addr = sock.recvfrom(9000)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if only_from and addr[0] != only_from:
                    continue
                msg = DNSIncoming(data)
                if msg.valid and msg.is_response():
                    records.extend((addr[0], r) for r in msg.answers())

        try:
            ask(dest, service_type, const._TYPE_PTR)
        except OSError as e:
            if log:
                log(f"Couldn't ask {dest}: {e}")
            return []
        collect(time.time() + timeout)
        if dest == MDNS_GROUP:
            ask(dest, service_type, const._TYPE_PTR)  # a second chance for slow phones
            collect(time.time() + 1.0)

        # Each instance, from the device that announced it.
        instances = {}
        for src, r in records:
            if r.type == const._TYPE_PTR and r.name == service_type:
                instances.setdefault(r.alias, src)
        missing = False
        for inst, src in instances.items():
            have = {r.type for s, r in records if r.name == inst}
            for rtype in (const._TYPE_SRV, const._TYPE_TXT):
                if rtype not in have:
                    ask(src, inst, rtype)
                    missing = True
        if missing:
            collect(time.time() + timeout)

    results = []
    for inst, src in sorted(instances.items()):
        srv = next((r for s, r in records if r.type == const._TYPE_SRV and r.name == inst), None)
        txt = next((r for s, r in records if r.type == const._TYPE_TXT and r.name == inst), None)
        if not srv:
            continue
        results.append({"instance": inst, "host": src, "port": srv.port,
                        "props": _txt_props(txt.text) if txt else {}})
    return results


def query_host(host: str, service_type: str, timeout: float = 2.0, log: LogFn = None) -> List[dict]:
    """Asks the device at `host` for its `service_type` services directly."""
    results = _legacy_query(host, service_type, timeout, log, only_from=host)
    if log:
        log(f"Asked {host} directly: {len(results)} service(s)")
    return results


def legacy_browse(service_type: str, timeout: float = 2.5, log: LogFn = None) -> List[dict]:
    """Asks the whole Wi-Fi from an ordinary port: every device answers us
    directly, which works where browse() loses answers."""
    results = _legacy_query(MDNS_GROUP, service_type, timeout, log)
    if log:
        log(f"Asked the Wi-Fi directly: {len(results)} found ({', '.join(r['host'] for r in results) or 'none'})")
    return results


def lan_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("192.168.1.1", 9))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def scan_subnet(port: int, timeout: float = 0.5, log: LogFn = None) -> List[str]:
    """Addresses on this computer's /24 network with `port` open: finds
    devices even where the Wi-Fi blocks discovery traffic altogether."""
    from concurrent.futures import ThreadPoolExecutor

    local = lan_address()
    if local == "127.0.0.1":
        return []
    prefix = local.rsplit(".", 1)[0]
    hosts = [f"{prefix}.{i}" for i in range(1, 255) if f"{prefix}.{i}" != local]

    def check(host):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return host if s.connect_ex((host, port)) == 0 else None

    with ThreadPoolExecutor(max_workers=64) as pool:
        found = [h for h in pool.map(check, hosts) if h]
    if log:
        log(f"Scanned {prefix}.0/24 for port {port}: {', '.join(found) or 'none'}")
    return found


def find_all(service_type: str, port: int, known_hosts=(), log: LogFn = None) -> List[dict]:
    """Every device offering `service_type`, by every way there is: the usual
    search, a direct question to the whole Wi-Fi, a direct question to each
    known address, and a scan of the network for `port`. Devices found only by
    the scan come back with empty props when they don't answer mDNS."""
    found: Dict[tuple, dict] = {}

    def add(items):
        for f in items:
            found.setdefault((f["host"], f["port"]), f)

    try:
        add(browse(service_type, 3.0, log=log))
    except Exception as e:  # noqa: BLE001 - other ways follow
        if log:
            log(f"Wi-Fi search failed: {e}")
    try:
        add(legacy_browse(service_type, log=log))
    except Exception as e:  # noqa: BLE001
        if log:
            log(f"Direct Wi-Fi question failed: {e}")
    scanned = set(scan_subnet(port, log=log))
    for host in sorted(set(known_hosts) | scanned):
        if any(h == host for h, _ in found):
            continue
        try:
            answers = query_host(host, service_type, log=log)
        except Exception:  # noqa: BLE001
            answers = []
        if answers:
            add(answers)
        elif host in scanned:  # its port is open, it just didn't answer mDNS
            found.setdefault((host, port), {"instance": host, "host": host, "port": port, "props": {}})
    return list(found.values())
