#!/bin/sh
# Installs the latest Antares release on macOS or Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/Madushan996/Antares-Programming-Language/main/install.sh | sh
#
# Antares goes into ~/.antares/bin (set ANTARES_INSTALL to choose another
# folder), and that folder is added to your PATH in ~/.zshrc or ~/.bashrc.
# Set ANTARES_NO_MODIFY_PATH=1 to leave your shell files alone.
set -eu

REPO="Madushan996/Antares-Programming-Language"
DIR="${ANTARES_INSTALL:-$HOME/.antares}/bin"

case "$(uname -s)" in
    Linux) os=linux ;;
    Darwin) os=macos ;;
    *)
        echo "There is no ready-made Antares download for $(uname -s)." >&2
        echo "You can build it from source: https://github.com/$REPO#build-from-source" >&2
        exit 1
        ;;
esac
case "$(uname -m)" in
    x86_64 | amd64) arch=x86_64 ;;
    arm64 | aarch64) arch=arm64 ;;
    *)
        echo "There is no ready-made Antares download for $(uname -m) processors." >&2
        echo "You can build it from source: https://github.com/$REPO#build-from-source" >&2
        exit 1
        ;;
esac

name="antares-$os-$arch"
url="https://github.com/$REPO/releases/latest/download/$name.tar.gz"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "Downloading $url"
curl -fsSL "$url" -o "$tmp/antares.tar.gz"
tar -xzf "$tmp/antares.tar.gz" -C "$tmp"
mkdir -p "$DIR"
mv -f "$tmp/$name/antares" "$DIR/antares"
chmod +x "$DIR/antares"
echo "Installed $("$DIR/antares" version) to $DIR/antares"

case ":$PATH:" in
    *":$DIR:"*) exit 0 ;;
esac

line="export PATH=\"$DIR:\$PATH\""
rc=""
case "${SHELL:-}" in
    */zsh) rc="$HOME/.zshrc" ;;
    */bash) if [ "$os" = macos ]; then rc="$HOME/.bash_profile"; else rc="$HOME/.bashrc"; fi ;;
esac

if [ -n "$rc" ] && [ -z "${ANTARES_NO_MODIFY_PATH:-}" ]; then
    if ! grep -qsF "$line" "$rc"; then
        printf '\n# Antares\n%s\n' "$line" >> "$rc"
    fi
    echo "Added $DIR to your PATH in $rc."
    echo "Open a new terminal (or run: source $rc), then try: antares version"
else
    echo "To use antares from anywhere, add this line to your shell's startup file:"
    echo "    $line"
fi
