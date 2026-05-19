import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Image(systemName: "keyboard")
                    .font(.system(size: 60))
                    .foregroundColor(.accentColor)

                Text("子音のみ日本語 IME")
                    .font(.title.bold())

                Text("子音だけ入力→ローマ字→かな漢字 変換")
                    .font(.subheadline)
                    .foregroundColor(.secondary)

                Divider()

                VStack(alignment: .leading, spacing: 12) {
                    Label("使い方", systemImage: "info.circle")
                        .font(.headline)

                    Text("1. 設定 → 一般 → キーボード → キーボードを追加")
                    Text("2. 「Shiin IME Keyboard」を追加")
                    Text("3. テキスト入力欄で 🌐 を長押しして切り替え")
                    Text("4. 子音（ktsh → 何を食べたいし…）を打鍵")
                    Text("5. 候補バーの変換結果をタップして確定")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Color(.secondarySystemBackground))
                .cornerRadius(12)

                Spacer()
            }
            .padding()
            .navigationTitle("Shiin IME")
        }
    }
}
