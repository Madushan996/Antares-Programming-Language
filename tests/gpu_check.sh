#!/bin/bash
# Checks Antares's GPU support on a Linux machine with an NVIDIA GPU, such
# as Google Colab or Kaggle: builds Antares from this folder, runs every GPU
# test, compares speed with the CPU, and prints a report.
#
# In a Colab or Kaggle notebook cell (on Kaggle, turn on Internet in the
# notebook's settings and pick a GPU accelerator first):
#
#   !git clone -b claude/cuda https://github.com/Madushan996/Antares-Programming-Language antares
#   !bash antares/tests/gpu_check.sh

cd "$(dirname "$0")/.." || exit 1
report=()
note() { report+=("$1"); echo; echo "== $1"; }

echo "== The machine"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi: no NVIDIA GPU found"
ls /usr/local/cuda/lib64/libnvrtc.so* 2>/dev/null | head -3

# Rust, to build Antares (Colab and Kaggle don't have it)
if ! command -v cargo >/dev/null 2>&1; then
    [ -x "$HOME/.cargo/bin/cargo" ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "== Building (about 2 minutes)"
if cargo build --release 2>&1 | tail -3; then :; fi
A=./target/release/antares
if [ ! -x "$A" ]; then
    note "BUILD FAILED"
    printf '%s\n' "${report[@]}"
    exit 1
fi

printf 'show gpu()\n' > /tmp/antares_gpu_name.ant
gpu=$("$A" /tmp/antares_gpu_name.ant 2>&1)
note "Antares sees the GPU as: $gpu"

echo "== Rust GPU tests"
if out=$(cargo test --release --lib gpu 2>&1); then
    note "Rust GPU tests: $(echo "$out" | grep 'test result' | head -1)"
else
    echo "$out" | grep -E "panicked|assert|FAILED|error" | head -20
    note "Rust GPU tests: FAILED (details above)"
fi

echo "== Antares tests"
for t in gpu deep_learning check features; do
    if out=$("$A" tests/$t.ant 2>&1); then
        note "tests/$t.ant: ok"
    else
        echo "$out" | tail -5
        note "tests/$t.ant: FAILED: $(echo "$out" | grep -m1 Problem)"
    fi
done

echo "== Speed"
cat > /tmp/antares_train.ant <<'EOF'
seed(1)
x = randn(1024, 784)
labels = floor(rand(1024) * 10)
learn w1 = he(784, 1024)
learn b1 = zeros(1024)
learn w2 = he(1024, 1024)
learn b2 = zeros(1024)
learn w3 = xavier(1024, 10)
step = () => cross_entropy(gelu(gelu(x @ w1 + b1) @ w2 + b2) @ w3, labels)
descend step() by 0.001 using adamw clip 1     # once to warm up
t0 = clock()
repeat 10 { descend step() by 0.001 using adamw clip 1 }
show "loss {fixed(sum(step()), 4)} in {round((clock() - t0) * 100) / 100} s"
EOF
(echo "use gpu"; cat /tmp/antares_train.ant) > /tmp/antares_train_gpu.ant
cpu=$("$A" /tmp/antares_train.ant 2>&1)
gpu_run=$("$A" /tmp/antares_train_gpu.ant 2>&1)
note "Training 784-1024-1024-10 for 10 steps, CPU: $cpu"
note "Training 784-1024-1024-10 for 10 steps, GPU: $gpu_run"
bench=$(cargo test --release --lib benchmark -- --ignored --nocapture 2>&1 | grep -E "x [0-9]+:")
note "Matrix products:"$'\n'"$bench"

echo
echo "================ REPORT (paste this back) ================"
printf '%s\n' "${report[@]}"
