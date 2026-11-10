import Flutter
import Foundation
import Metal
import UIKit

/// Phone cluster on iOS: this device can be a worker (its GPU serves a
/// master's model layers) or a master (it runs a model with part of its
/// layers on workers). Same channel and methods as the Android
/// ClusterChannel.
final class ClusterChannel: NSObject, NetServiceDelegate {
    private let channel: FlutterMethodChannel
    private var service: NetService?
    private var browser: WorkerBrowser?
    private var ramMb = 0
    private var bandwidth = 0.0
    private var error: String?

    init(messenger: FlutterBinaryMessenger) {
        channel = FlutterMethodChannel(name: "llama_flutter_android/cluster", binaryMessenger: messenger)
        super.init()
        channel.setMethodCallHandler { [weak self] call, result in
            self?.handle(call, result: result)
        }
        NSLog("[LlamaCluster] Memory for a worker: %ld MB (app limit %ld MB)",
              Self.usableMemoryMb(), LlamaClusterWorker.availableMemoryMb())
    }

    private func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        let args = call.arguments as? [String: Any] ?? [:]
        switch call.method {
        case "startWorker":
            let port = args["port"] as? Int ?? 50052
            startWorker(port: port)
            result(nil)
        case "stopWorker":
            // ggml's server can't be stopped inside a running app.
            result(FlutterError(code: "IOS_STOP",
                                message: "Close PrivateLM to stop sharing this device.",
                                details: nil))
        case "workerStatus":
            let running = LlamaClusterWorker.isRunning()
            if !running, error == nil, let exitError = LlamaClusterWorker.lastError() {
                error = exitError
            }
            var status: [String: Any] = [
                "state": error != nil ? "error" : (running ? "running" : "stopped"),
                "heldMb": LlamaClusterWorker.footprintMb(),
                "ramMb": ramMb,
                "bandwidth": bandwidth,
                "id": Self.workerId,
            ]
            if let error = error { status["error"] = error }
            if running, let peer = LlamaClusterWorker.connectedPeer() { status["peer"] = peer }
            result(status)
        case "setKeepScreenOn":
            UIApplication.shared.isIdleTimerDisabled = args["on"] as? Bool ?? false
            result(nil)
        case "localFreeRamMb":
            result(LlamaClusterWorker.availableMemoryMb())
        case "benchmark":
            DispatchQueue.global(qos: .userInitiated).async {
                let gbps = LlamaClusterWorker.memoryBandwidth(withThreads: Int32(self.performanceCores()))
                DispatchQueue.main.async { result(gbps) }
            }
        case "requestBatteryUnrestricted":
            result(nil)  // Android only
        case "lastLoadBuffers":
            result(LlamaIosWrapper.lastLoadBuffers())
        case "discover":
            let seconds = Double(args["timeoutMs"] as? Int ?? 4000) / 1000
            browser?.cancel()
            let browser = WorkerBrowser()
            self.browser = browser
            browser.search(for: seconds) { [weak self] found in
                if self?.browser === browser { self?.browser = nil }
                // This device's own worker isn't one it can use.
                result(found.filter { $0["id"] as? String != Self.workerId })
            }
        case "probe":
            let endpoint = args["endpoint"] as? String ?? ""
            DispatchQueue.global(qos: .userInitiated).async {
                let ok = LlamaIosWrapper.probeEndpoint(endpoint, timeoutMs: 2000)
                DispatchQueue.main.async { result(ok) }
            }
        case "setCluster":
            let endpoints = args["endpoints"] as? [String] ?? []
            let weights = (args["weights"] as? [NSNumber]) ?? []
            let localWeight = (args["localWeight"] as? NSNumber)?.floatValue ?? 1
            DispatchQueue.global(qos: .userInitiated).async {
                // llama.cpp aborts the whole app on an unreachable worker, so
                // refuse the cluster unless every worker answers right now.
                let unreachable = endpoints.filter {
                    !LlamaIosWrapper.probeEndpoint($0, timeoutMs: 2000)
                }
                DispatchQueue.main.async {
                    if !unreachable.isEmpty {
                        result(FlutterError(code: "UNREACHABLE",
                                            message: "Can't reach \(unreachable.joined(separator: ", "))",
                                            details: nil))
                        return
                    }
                    LlamaIosWrapper.setClusterEndpoints(endpoints, weights: weights,
                                                        localWeight: localWeight)
                    result(nil)
                }
            }
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    private func startWorker(port: Int) {
        guard !LlamaClusterWorker.isRunning() else { return }
        error = nil
        // iOS suspends apps that leave the screen, which would cut the master
        // off mid-token: keep the screen on while this device is a worker.
        UIApplication.shared.isIdleTimerDisabled = true

        DispatchQueue.global(qos: .userInitiated).async {
            // Measure first, before the model takes memory.
            self.ramMb = Self.usableMemoryMb()
            self.bandwidth = LlamaClusterWorker.memoryBandwidth(withThreads: Int32(self.performanceCores()))
            let cache = NSSearchPathForDirectoriesInDomains(.cachesDirectory, .userDomainMask, true)[0]
                + "/cluster_cache"
            let failure = LlamaClusterWorker.start(onPort: Int32(port),
                                                   threads: Int32(self.performanceCores()),
                                                   cacheDir: cache)
            DispatchQueue.main.async {
                if let failure = failure {
                    self.error = failure
                    UIApplication.shared.isIdleTimerDisabled = false
                } else {
                    self.advertise(port: port)
                }
            }
        }
    }

    /// Bonjour, so masters can find this device with "Find nearby".
    private func advertise(port: Int) {
        let service = NetService(domain: "local.", type: "_privatelm._tcp.",
                                 name: "PrivateLM \(UIDevice.current.name)", port: Int32(port))
        let txt: [String: Data] = [
            "ram": Data("\(ramMb)".utf8),
            "bw": Data(String(format: "%.1f", locale: Locale(identifier: "en_US_POSIX"), bandwidth).utf8),
            "model": Data(UIDevice.current.model.utf8),
            "v": Data("rpc7".utf8),
            "id": Data(Self.workerId.utf8),
        ]
        service.setTXTRecord(NetService.data(fromTXTRecord: txt))
        service.delegate = self
        service.publish()
        self.service = service
    }

    /// What this device can hold as a worker. The increased-memory-limit
    /// entitlement lifts iOS's per-app limit to about all of RAM (8.2 GB on
    /// an 8 GB iPad), but the worker keeps its layers in GPU memory, so the
    /// GPU's recommended working set (5.7 GB on an M1) is the real ceiling.
    static func usableMemoryMb() -> Int {
        let appLimit = LlamaClusterWorker.availableMemoryMb()
        guard let gpu = MTLCreateSystemDefaultDevice() else { return appLimit }
        let gpuMb = Int(gpu.recommendedMaxWorkingSetSize / (1024 * 1024))
        return min(appLimit, gpuMb)
    }

    /// This device's lasting worker ID, so a master can find it again when
    /// its IP address changes (another Wi-Fi network, a hotspot, DHCP).
    static var workerId: String {
        let key = "privatelm.cluster.workerId"
        if let id = UserDefaults.standard.string(forKey: key) { return id }
        let id = UUID().uuidString.lowercased()
        UserDefaults.standard.set(id, forKey: key)
        return id
    }

    private func performanceCores() -> Int {
        var count: Int32 = 0
        var size = MemoryLayout<Int32>.size
        if sysctlbyname("hw.perflevel0.physicalcpu", &count, &size, nil, 0) == 0, count > 0 {
            return Int(count)
        }
        return max(2, ProcessInfo.processInfo.activeProcessorCount / 2)
    }
}

/// Master: finds workers on the Wi-Fi by the Bonjour service they advertise
/// (Android's NSD and the iOS worker above use the same type and TXT keys).
/// Use on the main thread.
private final class WorkerBrowser: NSObject, NetServiceBrowserDelegate, NetServiceDelegate {
    private let browser = NetServiceBrowser()
    private var services: [NetService] = []
    private var completion: (([[String: Any]]) -> Void)?

    /// Browses for `seconds`, then gives services still resolving a moment
    /// more. Calls `completion` once, on the main thread.
    func search(for seconds: TimeInterval, completion: @escaping ([[String: Any]]) -> Void) {
        self.completion = completion
        browser.delegate = self
        browser.searchForServices(ofType: "_privatelm._tcp.", inDomain: "local.")
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds) {
            self.browser.stop()
            let resolving = self.services.contains { Self.ipv4Address(of: $0) == nil }
            DispatchQueue.main.asyncAfter(deadline: .now() + (resolving ? 2 : 0)) { self.finish() }
        }
    }

    /// Ends a search early with what it has found so far.
    func cancel() {
        browser.stop()
        finish()
    }

    private func finish() {
        guard let completion = completion else { return }
        self.completion = nil
        let found = services.compactMap(Self.describe)
        for service in services {
            service.stop()
            service.delegate = nil
        }
        services.removeAll()
        completion(found)
    }

    func netServiceBrowser(_ browser: NetServiceBrowser, didFind service: NetService, moreComing: Bool) {
        services.append(service)  // a service must be kept to resolve
        service.delegate = self
        service.resolve(withTimeout: 3)
    }

    func netServiceBrowser(_ browser: NetServiceBrowser, didRemove service: NetService, moreComing: Bool) {
        services.removeAll { $0 == service }
    }

    func netServiceBrowser(_ browser: NetServiceBrowser, didNotSearch errorDict: [String: NSNumber]) {
        NSLog("[LlamaCluster] Discovery failed to start: %@", errorDict)
        finish()
    }

    /// The map the Dart side reads (ClusterWorker.fromMap), or nil if the
    /// service didn't resolve to an IPv4 address.
    private static func describe(_ service: NetService) -> [String: Any]? {
        guard let host = ipv4Address(of: service), service.port > 0 else { return nil }
        var txt: [String: String] = [:]
        if let data = service.txtRecordData() {
            for (key, value) in NetService.dictionary(fromTXTRecord: data) {
                txt[key] = String(data: value, encoding: .utf8)
            }
        }
        var worker: [String: Any] = ["name": service.name, "host": host, "port": service.port]
        if let ram = txt["ram"].flatMap({ Int($0) }) { worker["ramMb"] = ram }
        if let bandwidth = txt["bw"].flatMap({ Double($0) }) { worker["bandwidth"] = bandwidth }
        if let model = txt["model"] { worker["model"] = model }
        if let id = txt["id"] { worker["id"] = id }
        if let version = txt["v"] { worker["version"] = version }
        return worker
    }

    private static func ipv4Address(of service: NetService) -> String? {
        for address in service.addresses ?? [] {
            let host = address.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> String? in
                guard let base = raw.baseAddress, raw.count >= MemoryLayout<sockaddr_in>.size else { return nil }
                let socketAddress = base.assumingMemoryBound(to: sockaddr.self)
                guard socketAddress.pointee.sa_family == sa_family_t(AF_INET) else { return nil }
                var name = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                guard getnameinfo(socketAddress, socklen_t(raw.count), &name, socklen_t(name.count),
                                  nil, 0, NI_NUMERICHOST) == 0 else { return nil }
                return String(cString: name)
            }
            if let host = host { return host }
        }
        return nil
    }
}
