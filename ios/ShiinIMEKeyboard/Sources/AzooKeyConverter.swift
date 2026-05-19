import Foundation
import KanaKanjiConverterModuleWithDefaultDictionary

/// Wraps AzooKeyKanaKanjiConverter for romaji → kana/kanji conversion.
final class AzooKeyConverter {
    static let shared = AzooKeyConverter()
    private init() {}

    private lazy var converter = KanaKanjiConverter.withDefaultDictionary()
    private let storageURL = FileManager.default
        .urls(for: .cachesDirectory, in: .userDomainMask)[0]

    /// Returns deduped kanji/kana candidates from multiple romaji inputs.
    func convertAll(_ romajiCandidates: [String], total: Int = 9) -> [String] {
        var out = [String]()
        var seen = Set<String>()
        for romaji in romajiCandidates {
            for text in convert(romaji) {
                if seen.insert(text).inserted { out.append(text) }
                if out.count >= total { return out }
            }
        }
        return out
    }

    // MARK: - Private

    private func convert(_ romaji: String, nBest: Int = 6) -> [String] {
        var composing = ComposingText()
        composing.insertAtCursorPosition(romaji, inputStyle: .roman2kana)
        let options = ConvertRequestOptions(
            N_best: nBest,
            requireJapanesePrediction: true,
            requireEnglishPrediction: false,
            keyboardLanguage: .ja_JP,
            englishCandidateInRoman2KanaInput: false,
            fullWidthRomanCandidate: false,
            halfWidthKanaCandidate: false,
            learningType: .nothing,
            maxMemoryCount: 0,
            shouldResetMemory: false,
            memoryDirectoryURL: storageURL,
            sharedContainerURL: storageURL,
            textReplacer: .withDefaultEmojiDictionary(),
            specialCandidateProviders: KanaKanjiConverter.defaultSpecialCandidateProviders,
            metadata: .init(versionString: "ShiinIME 1.0")
        )
        return converter.requestCandidates(composing, options: options)
            .mainResults
            .map { $0.text }
    }
}
