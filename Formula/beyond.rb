class Beyond < Formula
  desc "AI-native terminal with block-oriented model and GPU-accelerated rendering"
  homepage "https://github.com/PewterZz/Beyond"
  version "0.1.0"
  license "GPL-3.0-or-later"

  on_macos do
    on_arm do
      url "https://github.com/PewterZz/Beyond/releases/download/v#{version}/beyonder-v#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER"
    end

    on_intel do
      url "https://github.com/PewterZz/Beyond/releases/download/v#{version}/beyonder-v#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/PewterZz/Beyond/releases/download/v#{version}/beyonder-v#{version}-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "PLACEHOLDER"
    end

    on_intel do
      url "https://github.com/PewterZz/Beyond/releases/download/v#{version}/beyonder-v#{version}-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "PLACEHOLDER"
    end
  end

  def install
    bin.install "beyonder"
  end

  def caveats
    <<~EOS
      Beyond requires:
        - A GPU that supports wgpu (Metal on macOS, Vulkan/DX12 on Linux)
        - Ollama running locally for agent features: ollama serve
    EOS
  end

  test do
    assert_match "Beyond", shell_output("#{bin}/beyonder --version 2>&1", 0)
  end
end
