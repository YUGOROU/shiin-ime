import CoreML
import Foundation

final class ShiinInferenceEngine {
    static let shared = ShiinInferenceEngine()
    private init() { loadVocab() }

    private let FIXED_T = 52
    private let PAD = 0, SOS = 1, EOS = 2, VSZ = 29

    private var encoderModel: MLModel?
    private var decoderModel: MLModel?
    private var c2i = [Character: Int]()
    private var i2c = [Int: String]()
    private let lock = NSLock()
    private var modelsLoaded = false

    // MARK: - Public API

    func warmUp() {
        tryLoadModels()
    }

    /// Returns top-3 romaji strings for the given consonant sequence.
    func predict(_ consonants: String) -> [String] {
        tryLoadModels()
        guard let encoder = encoderModel, let decoder = decoderModel else { return [] }

        let filtered = consonants.lowercased().filter { c2i[$0] != nil }
        guard !filtered.isEmpty else { return [] }

        var tokens = [SOS]
        for ch in filtered { tokens.append(c2i[ch]!) }
        tokens.append(EOS)
        guard tokens.count <= FIXED_T else { return [] }

        let (srcLeft, maskLeft) = makePaddedArrays(tokens)
        guard let (encOut, hInit) = runEncoder(encoder,
                                               srcLeft: srcLeft,
                                               mask: maskLeft) else { return [] }
        return diverseBeamSearch(decoder, encOut: encOut, hInit: hInit, mask: maskLeft)
    }

    // MARK: - Model loading

    private func loadVocab() {
        guard let url = Bundle(for: Self.self).url(forResource: "vocab", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rawC2I = json["c2i"] as? [String: Int] else { return }
        for (k, v) in rawC2I {
            if let ch = k.first { c2i[ch] = v }
            i2c[v] = k
        }
    }

    private func tryLoadModels() {
        lock.lock()
        defer { lock.unlock() }
        guard !modelsLoaded else { return }
        modelsLoaded = true
        encoderModel = try? loadModel(named: "encoder")
        decoderModel = try? loadModel(named: "decoder_step")
    }

    private func loadModel(named name: String) throws -> MLModel {
        let bundle = Bundle(for: Self.self)
        guard let pkgURL = bundle.url(forResource: name, withExtension: "mlpackage") else {
            throw ModelError.notFound(name)
        }

        let modelVersion = "v5"  // bump when encoder/decoder interface changes
        let cacheDir = FileManager.default
            .urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("ShiinIME/models-\(modelVersion)")
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        let compiledURL = cacheDir.appendingPathComponent("\(name).mlmodelc")

        if !FileManager.default.fileExists(atPath: compiledURL.path) {
            let tmp = try MLModel.compileModel(at: pkgURL)
            try FileManager.default.moveItem(at: tmp, to: compiledURL)
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        return try MLModel(contentsOf: compiledURL, configuration: config)
    }

    enum ModelError: Error {
        case notFound(String)
    }

    // MARK: - Inference helpers

    private func makePaddedArrays(_ tokens: [Int])
        -> (srcLeft: MLMultiArray, maskLeft: MLMultiArray)
    {
        let srcLeft  = try! MLMultiArray(shape: [1, FIXED_T as NSNumber], dataType: .int32)
        let maskLeft = try! MLMultiArray(shape: [1, FIXED_T as NSNumber], dataType: .float32)

        for i in 0..<FIXED_T {
            srcLeft[i]  = 0
            maskLeft[i] = 0.0
        }
        for (j, tok) in tokens.enumerated() {
            srcLeft[j]  = NSNumber(value: tok)  // left-aligned [SOS, ..., EOS, PAD...]
            maskLeft[j] = 1.0
        }
        return (srcLeft, maskLeft)
    }

    private func runEncoder(_ model: MLModel,
                            srcLeft: MLMultiArray,
                            mask: MLMultiArray) -> (MLMultiArray, MLMultiArray)? {
        guard let inp = try? MLDictionaryFeatureProvider(dictionary: [
                  "src_left":  srcLeft,
                  "attn_mask": mask]),
              let out    = try? model.prediction(from: inp),
              let encOut = out.featureValue(for: "enc_out")?.multiArrayValue,
              let hInit  = out.featureValue(for: "h_init")?.multiArrayValue else { return nil }
        return (encOut, hInit)
    }

    private func diverseBeamSearch(_ model: MLModel,
                                    encOut: MLMultiArray, hInit: MLMultiArray,
                                    mask: MLMultiArray,
                                    numGroups: Int = 3,
                                    beamPerGroup: Int = 3,
                                    diversity: Float = 0.8,
                                    maxLen: Int = 72) -> [String] {
        struct Beam { let lp: Float; let toks: [Int]; let h: MLMultiArray }

        var groups: [[Beam]] = (0..<numGroups).map { _ in
            [Beam(lp: 0, toks: [SOS], h: copyArray(hInit))]
        }
        var doneGroups: [[(lp: Float, toks: [Int])]] = Array(repeating: [], count: numGroups)

        for _ in 0..<maxLen {
            var prevSelected: [Int] = []

            for g in 0..<numGroups {
                var next: [Beam] = []
                var thisSelected: [Int] = []

                for b in groups[g] {
                    if b.toks.last == EOS {
                        doneGroups[g].append((b.lp / Float(max(b.toks.count - 1, 1)), b.toks))
                        continue
                    }
                    let tokArr = try! MLMultiArray(shape: [1], dataType: .int32)
                    tokArr[0] = NSNumber(value: b.toks.last!)
                    guard let inp = try? MLDictionaryFeatureProvider(dictionary: [
                                  "tok": tokArr, "h": b.h,
                                  "enc_out": encOut, "attn_mask": mask]),
                          let out     = try? model.prediction(from: inp),
                          let logits  = out.featureValue(for: "logits")?.multiArrayValue,
                          let hNewRaw = out.featureValue(for: "h_new")?.multiArrayValue
                    else { continue }
                    let hNew = copyArray(hNewRaw)

                    var lps = logSoftmax(logits)

                    if g > 0 {
                        var counts = [Int: Int]()
                        for t in prevSelected { counts[t, default: 0] += 1 }
                        for (t, c) in counts where t < VSZ {
                            lps[t] -= diversity * Float(c)
                        }
                    }

                    for (idx, lp) in (0..<VSZ).map({ ($0, lps[$0]) })
                                               .sorted(by: { $0.1 > $1.1 })
                                               .prefix(beamPerGroup) {
                        next.append(Beam(lp: b.lp + lp, toks: b.toks + [idx], h: hNew))
                        thisSelected.append(idx)
                    }
                }
                next.sort { $0.lp > $1.lp }
                groups[g] = Array(next.prefix(beamPerGroup))
                prevSelected.append(contentsOf: thisSelected)
            }
        }

        // Best beam from each group first, then remaining by score
        var primary: [(lp: Float, toks: [Int])] = []
        var secondary: [(lp: Float, toks: [Int])] = []
        for g in 0..<numGroups {
            var all = doneGroups[g]
            for b in groups[g] { all.append((b.lp / Float(max(b.toks.count - 1, 1)), b.toks)) }
            all.sort { $0.lp > $1.lp }
            if let best = all.first { primary.append(best) }
            secondary.append(contentsOf: all.dropFirst())
        }
        secondary.sort { $0.lp > $1.lp }

        var seen = Set<String>()
        var result: [String] = []
        for item in primary + secondary {
            guard result.count < 3 else { break }
            let s = decodeTokens(item.toks)
            if !s.isEmpty, seen.insert(s).inserted { result.append(s) }
        }
        return result
    }

    private func copyArray(_ src: MLMultiArray) -> MLMultiArray {
        let dst = try! MLMultiArray(shape: src.shape, dataType: src.dataType)
        let bytes = src.count * 4  // float32 = 4 bytes
        memcpy(dst.dataPointer, src.dataPointer, bytes)
        return dst
    }

    private func logSoftmax(_ arr: MLMultiArray) -> [Float] {
        let n = arr.count
        var v = (0..<n).map { Float(truncating: arr[$0]) }
        let mx = v.max() ?? 0
        v = v.map { $0 - mx }
        let logSumExp = log(v.map { exp($0) }.reduce(0, +))
        return v.map { $0 - logSumExp }
    }

    private func decodeTokens(_ toks: [Int]) -> String {
        toks.dropFirst().compactMap { t -> String? in
            guard t != EOS, t != PAD, t != SOS else { return nil }
            return i2c[t]
        }.joined()
    }
}
