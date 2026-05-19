import UIKit

final class KeyboardViewController: UIInputViewController {

    // MARK: - State

    private var inputBuffer = "" {
        didSet { bufferLabel.text = inputBuffer.isEmpty ? "" : "▶ \(inputBuffer)"; scheduleInference() }
    }
    private var candidates = [String]() { didSet { rebuildCandidateBar() } }
    private var inferenceTask: Task<Void, Never>?
    private var hasMarkedText = false
    private let haptic = UIImpactFeedbackGenerator(style: .light)

    // MARK: - Views

    private let candidateScrollView = UIScrollView()
    private let candidateStack = UIStackView()
    private let bufferLabel = UILabel()
    private let keyboardStack = UIStackView()
    private var bufferBar: UIView!
    private var heightConstraint: NSLayoutConstraint?

    // MARK: - Layout constants

    private enum Layout {
        static let candidateBarH: CGFloat = 44
        static let bufferBarH: CGFloat    = 28
        static let keyH: CGFloat          = 42
        static let hSpace: CGFloat        = 6
        static let vSpace: CGFloat        = 11
        static let rowPadTop: CGFloat     = 12
        static let rowPadBot: CGFloat     = 3
        static let sidePad: CGFloat       = 3
        static var keysH: CGFloat { 4 * keyH + 3 * vSpace + rowPadTop + rowPadBot }
        static var totalH: CGFloat { candidateBarH + bufferBarH + keysH }
    }

    // MARK: - Key tag (used to identify key buttons in dark-mode walk)
    private static let keyTag = 1001

    // MARK: - Keyboard layout

    private let rows: [[String]] = [
        ["q","w","e","r","t","y","u","i","o","p"],
        ["a","s","d","f","g","h","j","k","l"],
        ["z","x","c","v","b","n","m","⌫"],
        ["🌐","space","return"],
    ]

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        haptic.prepare()
        buildUI()
        Task.detached(priority: .background) {
            ShiinInferenceEngine.shared.warmUp()
        }
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        updateHeight()
    }

    override func viewSafeAreaInsetsDidChange() {
        super.viewSafeAreaInsetsDidChange()
        updateHeight()
    }

    // MARK: - Height

    private func updateHeight() {
        heightConstraint?.isActive = false
        // Account for home indicator on Face ID devices
        let safeExtra = max(view.safeAreaInsets.bottom - Layout.rowPadBot, 0)
        heightConstraint = view.heightAnchor.constraint(
            equalToConstant: Layout.totalH + safeExtra)
        heightConstraint!.priority = .defaultHigh
        heightConstraint!.isActive = true
    }

    // MARK: - UI construction

    private func buildUI() {
        view.backgroundColor = keyboardBgColor()

        bufferBar = makeBufferBar()
        buildCandidateBar()
        buildKeys()

        let main = UIStackView(arrangedSubviews: [candidateScrollView, bufferBar, keyboardStack])
        main.axis = .vertical
        main.spacing = 0
        main.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(main)
        NSLayoutConstraint.activate([
            main.topAnchor.constraint(equalTo: view.topAnchor),
            main.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            main.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            main.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            candidateScrollView.heightAnchor.constraint(equalToConstant: Layout.candidateBarH),
            bufferBar.heightAnchor.constraint(equalToConstant: Layout.bufferBarH),
        ])
    }

    private func buildCandidateBar() {
        candidateScrollView.backgroundColor = topBarBgColor()
        candidateScrollView.showsHorizontalScrollIndicator = false

        let sep = UIView()
        sep.backgroundColor = .separator
        sep.translatesAutoresizingMaskIntoConstraints = false
        candidateScrollView.addSubview(sep)

        candidateStack.axis = .horizontal
        candidateStack.spacing = 6
        candidateStack.alignment = .center
        candidateStack.translatesAutoresizingMaskIntoConstraints = false
        candidateScrollView.addSubview(candidateStack)

        NSLayoutConstraint.activate([
            candidateStack.topAnchor.constraint(
                equalTo: candidateScrollView.contentLayoutGuide.topAnchor),
            candidateStack.bottomAnchor.constraint(
                equalTo: candidateScrollView.contentLayoutGuide.bottomAnchor),
            candidateStack.leadingAnchor.constraint(
                equalTo: candidateScrollView.contentLayoutGuide.leadingAnchor, constant: 8),
            candidateStack.trailingAnchor.constraint(
                equalTo: candidateScrollView.contentLayoutGuide.trailingAnchor, constant: -8),
            candidateStack.heightAnchor.constraint(
                equalTo: candidateScrollView.frameLayoutGuide.heightAnchor),
            sep.leadingAnchor.constraint(equalTo: candidateScrollView.leadingAnchor),
            sep.trailingAnchor.constraint(equalTo: candidateScrollView.trailingAnchor),
            sep.bottomAnchor.constraint(equalTo: candidateScrollView.bottomAnchor),
            sep.heightAnchor.constraint(equalToConstant: 0.5),
        ])
    }

    private func makeBufferBar() -> UIView {
        bufferLabel.font = .monospacedSystemFont(ofSize: 12, weight: .regular)
        bufferLabel.textColor = .secondaryLabel
        bufferLabel.translatesAutoresizingMaskIntoConstraints = false

        let bar = UIView()
        bar.backgroundColor = topBarBgColor()
        bar.addSubview(bufferLabel)
        NSLayoutConstraint.activate([
            bufferLabel.leadingAnchor.constraint(equalTo: bar.leadingAnchor, constant: 10),
            bufferLabel.trailingAnchor.constraint(equalTo: bar.trailingAnchor, constant: -10),
            bufferLabel.centerYAnchor.constraint(equalTo: bar.centerYAnchor),
        ])
        return bar
    }

    private func buildKeys() {
        keyboardStack.axis = .vertical
        keyboardStack.spacing = Layout.vSpace
        keyboardStack.isLayoutMarginsRelativeArrangement = true
        keyboardStack.layoutMargins = UIEdgeInsets(
            top: Layout.rowPadTop, left: Layout.sidePad,
            bottom: Layout.rowPadBot, right: Layout.sidePad)

        keyboardStack.addArrangedSubview(makeLetterRow(rows[0]))
        keyboardStack.addArrangedSubview(makeLetterRow(rows[1]))
        keyboardStack.addArrangedSubview(makeLetterRow(rows[2]))
        keyboardStack.addArrangedSubview(makeBottomRow())
    }

    private func makeLetterRow(_ keys: [String]) -> UIStackView {
        let row = UIStackView()
        row.axis = .horizontal
        row.distribution = .fillEqually
        row.spacing = Layout.hSpace
        row.heightAnchor.constraint(equalToConstant: Layout.keyH).isActive = true
        for k in keys { row.addArrangedSubview(makeKey(k)) }
        return row
    }

    private func makeBottomRow() -> UIStackView {
        let row = UIStackView()
        row.axis = .horizontal
        row.distribution = .fill
        row.spacing = Layout.hSpace
        row.heightAnchor.constraint(equalToConstant: Layout.keyH).isActive = true

        let globe = makeKey("🌐")
        let space = makeKey("space")
        let ret   = makeKey("return")

        // Globe/Return: fixed proportions of available width; Space fills the rest.
        // UIScreen.main.bounds.width is reliable for full-width portrait keyboard.
        let availW = UIScreen.main.bounds.width - 2 * (Layout.sidePad + Layout.hSpace)
        globe.widthAnchor.constraint(equalToConstant: availW * 0.13).isActive = true
        ret.widthAnchor.constraint(equalToConstant: availW * 0.19).isActive = true
        space.setContentHuggingPriority(.defaultLow - 1, for: .horizontal)

        row.addArrangedSubview(globe)
        row.addArrangedSubview(space)
        row.addArrangedSubview(ret)
        return row
    }

    private func makeKey(_ label: String) -> UIButton {
        let special = ["⌫","space","return","🌐"].contains(label)
        let btn = UIButton(type: .custom)
        btn.tag = Self.keyTag
        btn.setTitle(displayLabel(label), for: .normal)
        btn.titleLabel?.font = (label.count == 1 && label != "⌫")
            ? .systemFont(ofSize: 17, weight: .light)
            : .systemFont(ofSize: 15, weight: .regular)
        btn.backgroundColor = special ? specialKeyColor() : letterKeyColor()
        btn.setTitleColor(.label, for: .normal)
        btn.layer.cornerRadius = 5
        btn.layer.masksToBounds = false
        applyShadow(to: btn)

        btn.addTarget(self, action: #selector(keyDown(_:)), for: .touchDown)
        btn.addTarget(self, action: #selector(keyUp(_:)),
                      for: [.touchUpInside, .touchUpOutside, .touchCancel])
        btn.addTarget(self, action: #selector(keyTapped(_:)), for: .touchUpInside)
        return btn
    }

    private func applyShadow(to btn: UIButton) {
        let dark = traitCollection.userInterfaceStyle == .dark
        btn.layer.shadowColor   = UIColor.black.cgColor
        btn.layer.shadowOpacity = dark ? 0 : 0.35
        btn.layer.shadowOffset  = CGSize(width: 0, height: 1)
        btn.layer.shadowRadius  = 0
    }

    private func displayLabel(_ key: String) -> String {
        switch key {
        case "space":  return "space"
        case "return": return "return"
        default:       return key
        }
    }

    // MARK: - Key press handling

    @objc private func keyDown(_ sender: UIButton) {
        haptic.impactOccurred()
        let special = ["⌫","space","return","🌐"].contains(sender.title(for: .normal))
        sender.backgroundColor = special ? pressedSpecialColor() : pressedLetterColor()
    }

    @objc private func keyUp(_ sender: UIButton) {
        let special = ["⌫","space","return","🌐"].contains(sender.title(for: .normal))
        sender.backgroundColor = special ? specialKeyColor() : letterKeyColor()
    }

    @objc private func keyTapped(_ sender: UIButton) {
        guard let label = sender.title(for: .normal) else { return }
        switch label {
        case "⌫":
            if inputBuffer.isEmpty {
                // バッファが空のとき: 未確定テキストがあれば破棄、なければ1文字削除
                if hasMarkedText { clearMarkedText() }
                else { textDocumentProxy.deleteBackward() }
            } else {
                clearMarkedText()         // 即座に未確定をクリア
                inputBuffer.removeLast()  // didSet → 推論 → 新しい未確定をセット
            }
        case "🌐":
            advanceToNextInputMode()
        case "return":
            commit(nil); textDocumentProxy.insertText("\n")
        case "space":
            if hasMarkedText {
                // 未確定テキスト（トップ候補）をそのまま確定
                textDocumentProxy.unmarkText()
                hasMarkedText = false
                inputBuffer = ""
                candidates = []
            } else if let first = candidates.first {
                commit(first)
            } else if !inputBuffer.isEmpty {
                textDocumentProxy.insertText(inputBuffer); inputBuffer = ""
            } else {
                textDocumentProxy.insertText(" ")
            }
        default:
            clearMarkedText()   // 新しいキー入力で即座にクリア
            inputBuffer += label
        }
    }

    private func commit(_ text: String?) {
        if hasMarkedText {
            // setMarkedText → unmarkText で未確定テキストを目的の文字列に差し替えて確定
            let finalText = text ?? ""
            textDocumentProxy.setMarkedText(
                finalText,
                selectedRange: NSRange(location: finalText.utf16.count, length: 0))
            textDocumentProxy.unmarkText()
            hasMarkedText = false
        } else if let text {
            textDocumentProxy.insertText(text)
        }
        inputBuffer = ""
        candidates = []
    }

    private func clearMarkedText() {
        guard hasMarkedText else { return }
        textDocumentProxy.setMarkedText("", selectedRange: NSRange(location: 0, length: 0))
        textDocumentProxy.unmarkText()
        hasMarkedText = false
    }

    // MARK: - Candidate bar

    private func rebuildCandidateBar() {
        for v in candidateStack.arrangedSubviews { v.removeFromSuperview() }
        for text in candidates {
            let btn = UIButton(type: .system)
            btn.setTitle(text, for: .normal)
            btn.titleLabel?.font = .systemFont(ofSize: 17)
            btn.setTitleColor(.label, for: .normal)
            btn.backgroundColor = UIColor.systemGray5
            btn.layer.cornerRadius = 6
            btn.contentEdgeInsets = UIEdgeInsets(top: 4, left: 12, bottom: 4, right: 12)
            btn.addTarget(self, action: #selector(candidateTapped(_:)), for: .touchUpInside)
            candidateStack.addArrangedSubview(btn)
        }
    }

    @objc private func candidateTapped(_ sender: UIButton) {
        guard let text = sender.title(for: .normal) else { return }
        commit(text)
    }

    // MARK: - Inference

    private func scheduleInference() {
        inferenceTask?.cancel()
        guard !inputBuffer.isEmpty else {
            candidates = []
            clearMarkedText()
            return
        }
        let buf = inputBuffer
        inferenceTask = Task {
            let romaji = await Task.detached(priority: .userInitiated) {
                ShiinInferenceEngine.shared.predict(buf)
            }.value
            guard !Task.isCancelled else { return }
            let kanji = await Task.detached(priority: .userInitiated) {
                AzooKeyConverter.shared.convertAll(romaji)
            }.value
            guard !Task.isCancelled else { return }
            await MainActor.run {
                self.candidates = kanji
                if let top = kanji.first, !top.isEmpty {
                    // トップ候補を未確定テキスト（下線付き）としてセット
                    self.textDocumentProxy.setMarkedText(
                        top,
                        selectedRange: NSRange(location: top.utf16.count, length: 0))
                    self.hasMarkedText = true
                } else {
                    self.clearMarkedText()
                }
            }
        }
    }

    // MARK: - Colors

    private func keyboardBgColor() -> UIColor {
        UIColor { $0.userInterfaceStyle == .dark
            ? UIColor(white: 0.21, alpha: 1)
            : UIColor(red: 0.82, green: 0.84, blue: 0.86, alpha: 1) }
    }
    private func topBarBgColor() -> UIColor {
        UIColor { $0.userInterfaceStyle == .dark
            ? UIColor(white: 0.16, alpha: 1)
            : .systemBackground }
    }
    private func letterKeyColor() -> UIColor {
        UIColor { $0.userInterfaceStyle == .dark
            ? UIColor(white: 0.38, alpha: 1)
            : .white }
    }
    private func specialKeyColor() -> UIColor {
        UIColor { $0.userInterfaceStyle == .dark
            ? UIColor(white: 0.27, alpha: 1)
            : UIColor(white: 0.68, alpha: 1) }
    }
    private func pressedLetterColor() -> UIColor {
        UIColor { $0.userInterfaceStyle == .dark
            ? UIColor(white: 0.52, alpha: 1)
            : UIColor(white: 0.80, alpha: 1) }
    }
    private func pressedSpecialColor() -> UIColor {
        UIColor { $0.userInterfaceStyle == .dark
            ? UIColor(white: 0.40, alpha: 1)
            : UIColor(white: 0.55, alpha: 1) }
    }

    // MARK: - Dark mode

    override func traitCollectionDidChange(_ previousTraitCollection: UITraitCollection?) {
        super.traitCollectionDidChange(previousTraitCollection)
        guard traitCollection.hasDifferentColorAppearance(comparedTo: previousTraitCollection) else { return }
        view.backgroundColor = keyboardBgColor()
        candidateScrollView.backgroundColor = topBarBgColor()
        bufferBar?.backgroundColor = topBarBgColor()
        walkKeys(keyboardStack)
    }

    private func walkKeys(_ v: UIView) {
        if let btn = v as? UIButton, btn.tag == Self.keyTag,
           let title = btn.title(for: .normal) {
            let special = ["⌫","space","return","🌐"].contains(title)
            btn.backgroundColor = special ? specialKeyColor() : letterKeyColor()
            applyShadow(to: btn)
        }
        for sub in v.subviews { walkKeys(sub) }
    }
}
