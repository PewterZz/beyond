class Beyondtty < Formula
  desc "AI-native terminal with block-oriented model and GPU-accelerated rendering"
  homepage "https://github.com/PewterZz/Beyond"
  version "0.1.1"
  license "GPL-3.0-or-later"

  on_macos do
    on_arm do
      url "https://github.com/PewterZz/Beyond/releases/download/v#{version}/beyondtty-v#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER"
    end

    on_intel do
      url "https://github.com/PewterZz/Beyond/releases/download/v#{version}/beyondtty-v#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "PLACEHOLDER"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/PewterZz/Beyond/releases/download/v#{version}/beyondtty-v#{version}-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "PLACEHOLDER"
    end

    on_intel do
      url "https://github.com/PewterZz/Beyond/releases/download/v#{version}/beyondtty-v#{version}-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "PLACEHOLDER"
    end
  end

  def install
    bin.install "beyondtty"
    bin.install "beyonder"
  end

  def caveats
    <<~EOS
      Beyond requires:
        - A GPU that supports wgpu (Metal on macOS, Vulkan/DX12 on Linux)
        - At least one LLM provider running for agent features (Ollama, llama.cpp, or MLX)
    EOS
  end

  test do
    assert_match "Beyond", shell_output("#{bin}/beyondtty --version 2>&1", 0)
  end
end
