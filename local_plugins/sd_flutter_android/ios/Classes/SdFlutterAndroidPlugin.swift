import Flutter
import UIKit

public class SdFlutterAndroidPlugin: NSObject, FlutterPlugin {
  private var channel: FlutterMethodChannel!
  private let wrapper = SdIosWrapper()

  public static func register(with registrar: FlutterPluginRegistrar) {
    let channel = FlutterMethodChannel(name: "sd_flutter_android", binaryMessenger: registrar.messenger())
    let instance = SdFlutterAndroidPlugin()
    instance.channel = channel
    registrar.addMethodCallDelegate(instance, channel: channel)
  }

  public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {
    case "getPlatformVersion":
      result("iOS " + UIDevice.current.systemVersion)

    case "initModel":
      guard let args = call.arguments as? [String: Any],
            let path = args["path"] as? String else {
        result(FlutterError(code: "INVALID_ARGUMENT", message: "Path is null", details: nil))
        return
      }
      DispatchQueue.global(qos: .userInitiated).async { [weak self] in
        guard let self = self else { return }
        let initResult = self.wrapper.loadModel(path)
        DispatchQueue.main.async {
          result(initResult)
        }
      }

    case "generateImage":
      guard let args = call.arguments as? [String: Any],
            let prompt = args["prompt"] as? String,
            let steps = args["steps"] as? Int else {
        result(FlutterError(code: "INVALID_ARGUMENT", message: "Invalid arguments", details: nil))
        return
      }

      self.wrapper.onProgress = { step, total in
        DispatchQueue.main.async {
          self.channel.invokeMethod("onProgress", arguments: [
            "step": step,
            "total": total
          ])
        }
      }

      DispatchQueue.global(qos: .userInitiated).async { [weak self] in
        guard let self = self else { return }
        let data = self.wrapper.generateImage(prompt, steps: Int32(steps))
        DispatchQueue.main.async {
          if let data = data {
            result(FlutterStandardTypedData(bytes: data))
          } else {
            result(FlutterError(code: "GENERATION_FAILED", message: "Native generation returned null", details: nil))
          }
        }
      }

    case "generateImageEx":
      // The phone cluster's image worker: size, seed, negative prompt, CFG
      // and sampler, with the size and seed actually used in the reply.
      guard let args = call.arguments as? [String: Any],
            let prompt = args["prompt"] as? String else {
        result(FlutterError(code: "INVALID_ARGUMENT", message: "Invalid arguments", details: nil))
        return
      }
      let steps = args["steps"] as? Int ?? 20
      let width = args["width"] as? Int ?? 512
      let height = args["height"] as? Int ?? 512
      let seed = (args["seed"] as? NSNumber)?.int64Value ?? -1
      let negative = args["negativePrompt"] as? String ?? ""
      let cfg = (args["cfgScale"] as? NSNumber)?.floatValue ?? -1
      let sampler = args["sampler"] as? String ?? ""
      let initImage = (args["initImage"] as? FlutterStandardTypedData)?.data
      let strength = (args["strength"] as? NSNumber)?.floatValue ?? 0.75
      let loraPaths = args["loraPaths"] as? [String] ?? []
      let loraWeights = (args["loraWeights"] as? [NSNumber]) ?? []

      self.wrapper.onProgress = { step, total in
        DispatchQueue.main.async {
          self.channel.invokeMethod("onProgress", arguments: ["step": step, "total": total])
        }
      }

      DispatchQueue.global(qos: .userInitiated).async { [weak self] in
        guard let self = self else { return }
        let out = self.wrapper.generateImage(prompt, steps: Int32(steps), width: Int32(width),
                                             height: Int32(height), seed: seed,
                                             negativePrompt: negative, cfgScale: cfg, sampler: sampler,
                                             initImage: initImage, strength: strength,
                                             loraPaths: loraPaths, loraWeights: loraWeights)
        DispatchQueue.main.async {
          if let out = out, let data = out["data"] as? Data {
            result([
              "rgb": FlutterStandardTypedData(bytes: data),
              "width": out["width"] ?? width,
              "height": out["height"] ?? height,
              "seed": out["seed"] ?? seed,
            ])
          } else {
            result(FlutterError(code: "GENERATION_FAILED", message: "Native generation returned null", details: nil))
          }
        }
      }

    case "unloadModel":
      wrapper.unloadModel()
      result(nil)

    default:
      result(FlutterMethodNotImplemented)
    }
  }
}
