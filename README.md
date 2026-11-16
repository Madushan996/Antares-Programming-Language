# Antares

[![CI](https://github.com/Madushan996/Antares-Programming-Language/actions/workflows/ci.yml/badge.svg)](https://github.com/Madushan996/Antares-Programming-Language/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**The programming language for AI.** Train neural networks, talk to
language models, and build AI agents in one small, friendly language.
Tensors, automatic differentiation, the building blocks of transformers, and
calls to Claude, OpenAI and local models are part of the language itself, not
libraries you install.

> **Status: experimental preview.** Antares is ready to try, learn and
> experiment with. The language may still change between versions, so programs
> written today may need small updates later. Feedback and bug reports are very
> welcome in [Issues](https://github.com/Madushan996/Antares-Programming-Language/issues).

📘 **Full documentation:** [Antares Language Reference (PDF)](docs/Antares_Language_Reference.pdf)
covers the whole language, every built-in function, and complete examples.

📗 **New to programming?** [Learn Antares: A Beginner's Guide (PDF)](docs/Learn_Antares.pdf)
teaches programming from the very first step, with exercises and answers.

**Train a model.** `learn` marks what may change; `descend` improves it.
Layers, losses and optimisers are built in, and matrix maths uses every core:

```
learn w1 = he(2, 32)
learn w2 = xavier(32, 2)

repeat 500 {
    loss = cross_entropy(gelu(x @ w1) @ w2, labels)
    descend loss by 0.01 using adamw clip 1
}
```

**Ask a model.** Answers come back as real values in the shape you ask for:

```
model claude = "anthropic:claude-opus-5-5"

person = ask claude "Who wrote Gitanjali?" as {name: text, born: number}
show person.born + 100
```

**Build an agent.** Give a model your own functions as tools, or built-in
ones that run commands and read the web, and it calls them:

```
to percent(amount, rate) = amount * rate / 100

model helper = "ollama:qwen2.5" with {tools: [percent, shell, read_page]}
show ask helper "What is 17% of 2450? And which file here is the biggest?"
```

- **AI in the language.** Conversations, tools, structured answers, images,
  streaming and embeddings for language models; tensors, autodiff,
  attention, layer norm, cross-entropy and AdamW for training your own.
  `examples/tiny_gpt.ant` is a working GPT-style transformer in about 100 lines.
- **Trains on the GPU.** `use gpu` sends training to the graphics card, with
  nothing to install: on any Windows PC, and on NVIDIA GPUs on Linux,
  including Google Colab and Kaggle. 53 times faster than the CPU on Colab's
  Tesla T4.
- **Agents that act.** `shell` runs commands and `read_page` reads the web,
  as tools a model can use by itself; Antares asks you before each command.
- **Graphics that look good.** Games in a window with smooth shapes,
  see-through colours, gradients, PNG pictures, real fonts, soft shadows and
  glowing light, drawn by Antares itself with nothing to install.
- **3D worlds.** A camera, shapes and `.obj` models, sunlight with soft
  shadows, lamps, sky, fog, and realistic shiny and metal materials, drawn
  on the graphics card.
- **Any model.** Claude, OpenAI and services like it, or free and private
  models on your own computer with Ollama.
- **One file, no dependencies.** The whole language is a single small program
  written in Rust with zero external crates. It starts instantly.
- **Reads like plain verbs.** `learn`, `descend`, `ask`, `show`, `repeat`,
  `each`, `when`, `to`, `give`.
- **Friendly errors.** Every problem names the line, shows it, and says what
  to do in plain words.

## Contents

- [Install](#install)
- [Your first program](#your-first-program)
- [Building something: a step-by-step example](#building-something-a-step-by-step-example)
- [Everyday commands](#everyday-commands)
- [Making an executable to share](#making-an-executable-to-share)
- [Troubleshooting](#troubleshooting)
- [The language in five minutes](#the-language-in-five-minutes)
- [Deep learning](#deep-learning) · [Language models](#language-models) · [Games in a window](#games-in-a-window) · [3D](#3d) · [Sound](#sound) · [Built-in functions](#built-in-functions)

## Install

You don't need Rust or any other tools to write and run Antares programs.
Pick one of these three ways.

### Option 1: one-line installer (easiest)

**Windows**: open **PowerShell** and run:

```powershell
irm https://raw.githubusercontent.com/Madushan996/Antares-Programming-Language/main/install.ps1 | iex
```

**macOS or Linux**: open **Terminal** and run:

```bash
curl -fsSL https://raw.githubusercontent.com/Madushan996/Antares-Programming-Language/main/install.sh | sh
```

The installer downloads the latest release, puts `antares` in your user
folder (`%LOCALAPPDATA%\Programs\Antares` on Windows, `~/.antares/bin` on
macOS and Linux), and adds it to your PATH. **Close the terminal and open a
new one**, then check it works:

```bash
antares version
```

You should see `Antares 0.13.1`.

### Option 2: download it yourself

1. Open the [latest release](https://github.com/Madushan996/Antares-Programming-Language/releases/latest).
2. Download the file for your computer:

   | Your computer | File to download |
   |---|---|
   | Windows (64-bit) | `antares-windows-x86_64.zip` |
   | Mac with Apple Silicon (M1, M2, M3, M4…) | `antares-macos-arm64.tar.gz` |
   | Mac with an Intel processor | `antares-macos-x86_64.tar.gz` |
   | Linux, most PCs | `antares-linux-x86_64.tar.gz` |
   | Linux on ARM (Raspberry Pi 4/5 with 64-bit OS, ARM servers) | `antares-linux-arm64.tar.gz` |

   Not sure which Mac you have? Apple menu → **About This Mac**: "Chip: Apple M…"
   means Apple Silicon; "Processor: Intel…" means Intel.
3. Unpack it (double-click the file). Inside is the `antares` program
   (`antares.exe` on Windows), with the README and license.
4. Move the program somewhere permanent and add that folder to your PATH, so
   you can type `antares` in any folder:
   - **Windows:** move `antares.exe` to e.g. `C:\Antares`. Press Start, type
     "environment variables", open **Edit environment variables for your
     account**, select **Path** → **Edit** → **New**, and add `C:\Antares`.
   - **macOS / Linux:** run `mkdir -p ~/.antares/bin && mv antares ~/.antares/bin/`,
     then add `export PATH="$HOME/.antares/bin:$PATH"` to `~/.zshrc` (Mac) or
     `~/.bashrc` (Linux).
5. **macOS only:** the first time, macOS may say the app "can't be opened
   because Apple cannot check it". Run this once to allow it:
   `xattr -d com.apple.quarantine ~/.antares/bin/antares`
6. Open a new terminal and run `antares version`.

### Option 3: build from source

<a id="build-from-source"></a>
For contributors, or computers without a ready-made download. You need
[Rust](https://rustup.rs) and Git.

```bash
git clone https://github.com/Madushan996/Antares-Programming-Language.git
```

```bash
cd Antares-Programming-Language
```

```bash
cargo build --release
```

The program is now at `target/release/antares` (`antares.exe` on Windows).
Add that folder to your PATH as in Option 2, or run it by its full path.

## Your first program

1. Make a folder for your work and open a terminal in it.
2. Create a file called `hello.ant` with any text editor (Notepad, TextEdit
   in plain-text mode, VS Code…) and type:

   ```
   name = "world"
   show "Hello, {name}!"

   to square(x) = x * x
   show "7 squared is", square(7)
   ```

3. Run it:

   ```bash
   antares hello.ant
   ```

   ```
   Hello, world!
   7 squared is 49
   ```

To try lines one at a time, run `antares` on its own. This opens an
interactive session; type `exit` to leave.

To see more, download the [examples folder](examples) (or clone the repository)
and run them: `antares examples/xor.ant` trains a small neural network, and
`antares examples/snake.ant` is a game you play in the terminal.

## Building something: a step-by-step example

A real program usually has several files: a main program, some helper
functions in their own file, and data. Here is a complete small project, a
model that learns to predict house prices, to show how the pieces fit.

**1. Create a project folder:**

```
house-prices/
├── main.ant        the program you run
├── helpers.ant     functions used by main.ant
└── houses.json     the data
```

**2. `houses.json`** holds the data (size in hundreds of square metres, price in
thousands):

```json
{"size": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0], "price": [150, 210, 250, 290, 350, 450]}
```

**3. `helpers.ant`** holds reusable functions:

```
# Squared error between a guess and the right answers
to mse(guess, target) = mean((guess - target) ^ 2)

to rounded(x) = round(x * 100) / 100
```

**4. `main.ant`** brings in the helpers with `use`, loads the data, learns,
saves what it learned, and answers a question:

```
use "helpers"                         # runs helpers.ant from this folder

data = load(program_folder() + "/houses.json")
size = data.size
price = data.price

learn per_unit = 0
learn base = 0

repeat 2000 {
    descend mse(size * per_unit + base, price) by 2 using adam
}

show "price = {rounded(per_unit)} x size + {rounded(base)}"
save({per_unit: per_unit, base: base}, program_folder() + "/model.json")

answer = input("Size of your house (e.g. 1.3): ")
show "Estimated price:", rounded(number(answer) * per_unit + base), "thousand"
```

**5. Run it** from inside the folder:

```bash
antares main.ant
```

```
price = 200 x size + 50
Size of your house (e.g. 1.3): 1.3
Estimated price: 310 thousand
```

**6. Share it.** Send the folder to anyone with Antares installed, or turn it
into a single program that runs without Antares (next sections).

Tips for bigger projects:

- `use "name"` loads `name.ant` from the same folder, once. All files share
  one set of global names, so give helper files distinctive names.
- `program_folder()` gives the folder of the running program, so data files
  are found even when the program is started from elsewhere.
- Wrap anything that can fail (files, model calls) in `try { … } else problem { … }`.
- The full manual, [docs/Antares_Language_Reference.pdf](docs/Antares_Language_Reference.pdf),
  explains every feature and built-in function.

## Everyday commands

| Command | What it does |
|---|---|
| `antares program.ant` | Run a program |
| `antares` | Interactive session (type `exit` to leave) |
| `antares build program.ant` | Make a stand-alone executable to share |
| `antares version` | Show the installed version |
| `antares help` | List the commands |

To use language models, set your Anthropic API key first:
`setx ANTHROPIC_API_KEY "your-key"` on Windows (then open a new terminal), or
`export ANTHROPIC_API_KEY="your-key"` on macOS/Linux.

## Making an executable to share

`antares build` turns a program into an executable that runs on computers
**without** Antares installed:

```bash
antares build main.ant
```

This creates `main.exe` (Windows) or `main` (macOS/Linux) next to `main.ant`.
It is made in one of two ways:

| | Packed (`--pack`) | Compiled (`--native`) |
|---|---|---|
| Needs Rust | No | Yes, from [rustup.rs](https://rustup.rs) |
| Build time | Under a second | 30 to 40 seconds (a minute or two the first time) |
| Speed | Same as `antares program.ant` | Much faster (see [Compiling](#compiling)) |

Without either option, `antares build` compiles when Rust is installed and
packs otherwise (it says so when it packs).

- A packed program is a copy of `antares` with your program, the files it
  brings in with `use`, and any `--embed` files attached at the end.
- A compiled program downloads the Antares source code for your version on
  its first build.
- Files the program reads (like `houses.json`) must be shipped next to the
  executable, or packed inside it with `--embed houses.json`.
- To share a compiled executable with people who have older computers, add
  `--cpu x86-64`. See [Compiling](#compiling) for details.

## Troubleshooting

| Problem | What to do |
|---|---|
| `antares: command not found` / `'antares' is not recognized` | Open a **new** terminal after installing. If it still fails, the install folder isn't on your PATH; see Option 2, step 4 |
| macOS: "cannot be opened because Apple cannot check it" | Run `xattr -d com.apple.quarantine "$(which antares)"` |
| `Couldn't open "program.ant"` | You're in a different folder. Use `cd` to go to the folder with the file, or give the full path |
| `antares build --native` says it needs Rust | Install Rust from [rustup.rs](https://rustup.rs) and open a new terminal, or build with `--pack` instead |
| A model call says to set `ANTHROPIC_API_KEY` | Set your API key as shown in [Everyday commands](#everyday-commands) |
| Something else | Open an [issue](https://github.com/Madushan996/Antares-Programming-Language/issues) with the program and the full message |

## The language in five minutes

### Values

```
age = 36                  # number
name = "Ada"              # text
ready = yes               # yes / no
empty = nothing
v = [1, 2, 3]             # a list of numbers IS a tensor
m = [[1, 2], [3, 4]]      # a matrix
mixed = ["a", 1, yes]     # a list
person = {name: "Ada", age: 36}   # a map
```

Text can contain values: `"Hello {name}, next year you are {age + 1}"`.
Write `\{` for a plain brace. Comments start with `#`. `+=`, `-=`, `*=` and
`/=` change a value by an amount: `age += 1`, `counts[word] += 1`.

### Maths

`+ - * / % ^` work on numbers and on whole tensors, element by element, and
smaller shapes stretch to fit larger ones (`m + [10, 20]` adds to every row).
`@` is matrix multiplication. Indexes start at 0; negative indexes count from
the end: `v[0]`, `v[-1]`, `m[1, 0]`.

A range `start:end` takes a part (the end is not included, and either side
can be left out):

```
v[1:3]        # items 1 and 2
v[:2]         # the first two
v[-2:]        # the last two
m[:, 0]       # every row, first column
m[1:, :2]     # from row 1 on, first two columns
"antares"[:3] # "ant" (works on text and lists too)
```

`sum(m)` adds everything; `sum(m, 0)` adds down the rows and `sum(m, 1)`
across the columns. `mean` works the same way.

### Integers and bits

Ordinary numbers are decimals. For bit manipulation there are 64-bit integers:

```
mask = 0xFF00                 # a 0x literal is an integer
n = int(12)                   # so is int(x)
row = ints(64)                # a row of 64 integers, all 0

n & 10      n | 3      n xor 5      ~n        # and, or, exclusive or, flip
1 << 4      n >> 2                            # shifts (>> brings in zeros)
popcount(mask)   lowest_bit(mask)   highest_bit(mask)
div(n, 5)                                     # division dropping the remainder
```

An integer combined with a whole number stays an integer (`n + 1`), `/`
always gives an ordinary number, and combining an integer with a fraction is
refused (use `number(n)` first). Arithmetic wraps around at 64 bits. Bit
operations bind tighter than comparisons, so `a & b != 0` means
`(a & b) != 0`. Compiled programs keep integers in machine registers.

### Functions and pipelines

```
to square(x) = x * x

to describe(n) {
    when n < 0 { give "negative" }      # give = return early
    when n == 0 { "zero" } else { "positive" }   # last value is the result
}

double = x => x * 2                     # a small unnamed function
```

`->` passes a value into the next function as its first input, so code reads
in the order it happens:

```
[3, 1, 2] -> sort -> map(x => x * 10) -> sum      # 60
```

A line that starts with `->` continues the line above.

Functions written with `to` can be used anywhere in the file, even above
where they are written. Inside a function, `x = ...` always makes a local
variable. To replace a variable that lives outside the function, say so with
`change`:

```
count = 0
to bump() { change count += 1 }
```

### Decisions and loops

```
when score > 90 { show "great" } else when score > 50 { show "ok" } else { show "hmm" }

repeat 3 { show "hi" }
repeat 10 as step { show step }         # step counts 1, 2, ... 10
each item in items { show item }
while n > 0 { n = n - 1 }               # stop leaves a loop, skip jumps to the next round
```

`and`, `or`, `not` combine yes/no values.

### Changing lists and maps

```
person.age = 37
person.skills[0] = "maths"
grid[0, 1] = 9
words = push(words, "new")
```

Values are never shared by accident: giving a list to a function or another
name gives it its own copy (made lazily, only if one side changes it).

### Splitting a program across files

```
use "nn"                  # runs nn.ant from the same folder, once
```

Everything the other file defines becomes available. Using the same file
twice does nothing the second time. A problem inside a used file is reported
with that file's name and line.

### When things go wrong

```
try {
    answer = ask claude "..."
} else problem {
    show "That failed: {problem}"
    answer = "unknown"
}
```

If anything inside `try` fails, the `else` block runs instead, with the
message in the name you chose (the name is optional). `fail("message")`
raises a problem of your own, and `check(condition, "message")` does so when
the condition is not met.

A top-level variable or a function's input can be given the range its
numbers must stay in. Every value it is given is then checked:

```
board = ints(64) within 0 to 15
to put(piece within 0 to 15, square within 0 to 63) {
    board[square] = piece
}
put(3, 64)      # problem: square only holds numbers from 0 to 63, not 64
```

## Deep learning

`learn` declares a value the program is allowed to adjust. `descend`
adjusts every `learn` variable that a number depends on, to make that
number smaller. That is the whole training API.

```
seed(7)
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

learn w1 = randn(2, 8) * 0.5
learn b1 = zeros(8)
learn w2 = randn(8, 1) * 0.5
learn b2 = zeros(1)

to net(input) = input @ w1 + b1 -> tanh -> (h => h @ w2 + b2) -> sigmoid

repeat 1000 {
    loss = mean((net(x) - y) ^ 2)
    descend loss by 0.05 using adam     # leave out `using adam` for plain gradient descent
}

show net(x) -> round
```

`grad(loss, w)` returns the gradient without changing anything.

### Keeping what was learned

```
save({w1: w1, b1: b1}, "model.json")     # any value: tensors, maps, lists

saved = load("model.json")
learn w1 = saved.w1                      # carry on training, or just use it
```

`exists("model.json")` says whether a file is there.
`examples/train_save.ant` trains a network on the first run and reuses it on
later runs.

Gradients flow through: `+ - * / ^ @`, indexing, slices, `exp log tanh
sigmoid relu sqrt abs sin cos`, `gelu silu softplus leaky_relu`, `sum mean
softmax log_softmax t reshape`, `take concat layer_norm dropout`, and the
losses.

### Building blocks

| Kind | Functions |
|---|---|
| Activations | `relu`, `gelu`, `silu`, `tanh`, `sigmoid`, `softplus`, `leaky_relu`, `softmax`, `log_softmax` |
| Losses | `cross_entropy(scores, answers)` (softmax built in; answers as class numbers or probabilities), `mse(guess, target)`, `bce(probability, target)` |
| Layers | `take(table, positions)` (rows by position: an embedding lookup), `concat(tensors [, dimension])`, `layer_norm(x)`, `dropout(x, p)`, `causal_mask(n)` (for attention) |
| Starting weights | `he(inputs, outputs)` for relu and gelu, `xavier(inputs, outputs)` for tanh and sigmoid |
| Results | `accuracy(scores, answers)`, `top(values, k)` (positions of the largest), `sample(probabilities [, temperature])` |

### Optimisers

```
descend loss by 0.1                          # plain gradient descent
descend loss by 0.05 using momentum
descend loss by 0.01 using adam
descend loss by 0.01 using adamw clip 1      # weight decay, and gradients limited to size 1
```

`clip` works with any method: when the gradients' overall size is larger
than the limit, they are scaled down to it, which keeps training steady.

### Data

```
rows = read_csv("houses.csv")                # a list of maps, numbers already numbers
x = columns(rows, ["size", "rooms"])         # a table: one row per house
y = columns(rows, "price")                   # one number per house

each batch in batches(shuffle(range(len(y))), 32) {
    loss = mse(take(x, batch) @ w, take(y, batch))
    descend loss by 0.01 using adam
}
```

### Training on the GPU

`use gpu` at the top of a program sends tensor maths to the graphics card:

```
use gpu
learn w1 = he(784, 1024)
...                                      # the rest of the program is unchanged
```

Big matrix products, element-wise maths and activations on large tensors,
their gradients, and the optimisers then run on the GPU, and the numbers stay
there between steps. Small sums stay on the CPU, where they are faster.
`use cpu` switches back, and `gpu()` gives the graphics card's name (or
nothing).

- It works on any Windows PC with a graphics card or built-in graphics,
  through Direct3D 11, which Windows includes, and on Linux with an NVIDIA
  GPU, through CUDA: Google Colab, Kaggle, and PCs with an NVIDIA card and
  its driver and CUDA toolkit. Nothing else needs installing.
- The GPU works in 32-bit numbers, so results agree with the CPU's to about
  6 digits. That is why it is turned on with `use gpu` rather than
  automatically: programs without it print exactly what they did before.
- Without a usable GPU (and on macOS, for now), `use gpu` says so once and
  everything stays on the CPU, so a program runs anywhere.
- The first program to use the GPU waits about a second while its programs
  for the graphics card are compiled; they are kept, so later programs start
  at once.

Against the CPU of the same machine:

| Task | Colab CPU | Colab Tesla T4 | Laptop CPU (4 cores) | Laptop Intel HD 520 |
|---|---|---|---|---|
| Training a 784-1024-1024-10 classifier, 10 steps | 29.0 s | 0.55 s | 29.5 s | 1.89 s |
| A 1024 x 1024 matrix product | 378 ms | 7.9 ms | 360 ms | 49 ms |

### On Google Colab and Kaggle

Colab and Kaggle give you a free NVIDIA GPU in the browser. Choose a GPU
runtime (Colab: *Runtime, Change runtime type, T4 GPU*; Kaggle: a GPU
accelerator, with *Internet* on in the notebook's settings), then in a cell:

```
!curl -fsSL https://raw.githubusercontent.com/Madushan996/Antares-Programming-Language/main/install.sh | sh
```

Write a program in a cell (`%%writefile` saves it as a file), with `use gpu`
at the top:

```
%%writefile train.ant
use gpu
...
```

and run it:

```
!~/.antares/bin/antares train.ant
```

Files on the notebook's machine disappear when the session ends, so `save`
models to Google Drive on Colab (mount it with
`from google.colab import drive; drive.mount('/content/drive')` in a Python
cell) or to `/kaggle/working` on Kaggle. For API keys, use the notebook's
Secrets panel and copy the key into the environment from a Python cell
(`os.environ["ANTHROPIC_API_KEY"] = ...`); programs started with `!` then see
it. `bash tests/gpu_check.sh`, from a copy of this repository, checks the GPU
support on such a machine and compares it with the CPU.

### A tiny GPT

`examples/tiny_gpt.ant` is a GPT-style transformer, the kind of model behind
modern chatbots, in about 100 lines: token and position embeddings,
self-attention with a causal mask, a feed-forward layer, layer norm and
residual connections. It learns a page of verse in a few seconds and then
writes new lines one character at a time. `examples/classify.ant` teaches a
network to tell two spirals apart.

## Language models

```
model claude = "anthropic:claude-opus-5-5" with {system: "Be brief."}

answer = ask claude "Explain {topic} in one sentence."
count  = ask claude "How many moons does Mars have?" as number
sure   = ask claude "Is 91 a prime number?" as yes/no
ideas  = ask claude "Give me five names for a cat." as list
person = ask claude "Describe Ada Lovelace." as {name: text, born: number, known_for: list of text}
```

`as` turns the reply into a real value you can calculate with: `number`,
`yes/no`, `text`, `list` (one item per line), `list of <shape>`, or a map
shape like `{name: text, born: number}`, which can nest. Shapes use the
model's structured output, so the answer always fits.

### Models

| Write | Model | Needs |
|---|---|---|
| `"anthropic:claude-opus-5-5"` | Claude | `ANTHROPIC_API_KEY` |
| `"openai:gpt-5"` | OpenAI | `OPENAI_API_KEY` |
| `"openai:<name>"` with `{url: "..."}` | Any service with OpenAI's interface (LM Studio, Groq, vLLM, ...) | its key in `OPENAI_API_KEY`, if it needs one |
| `"ollama:llama3.2"` | A free model on your own computer, with [Ollama](https://ollama.com) | Ollama running, and `ollama pull llama3.2` |
| `"echo"` | A pretend model that repeats the prompt, for trying things offline | nothing |

Options after `with`: `system` (text), `max_tokens` (number), `effort`
(`"low"` to `"max"`, Claude only), `stream` (yes/no), `tools` (a list), and
`url` (the service's address).

### Conversations

```
chat = conversation(claude)
ask chat "My name is Ada."
show ask chat "What is my name?"         # it remembers
show history(chat)                       # [{role: "user", text: "My name is Ada."}, ...]
```

`conversation(chat)` starts a new conversation from a copy of an old one.

### Tools: models that call your functions

```
to weather(city) = "sunny and 31 degrees in {city}"

model helper = "anthropic:claude-opus-5-5" with {tools: [weather]}
show ask helper "Should I take an umbrella in Kandy today?"
```

The model decides when to call a tool. Antares runs your function with the
inputs the model chose and gives it the result, as many times as it needs,
then you get the final answer. Describe a tool to help the model use it well:
`{run: weather, about: "Today's weather for a city", inputs: {city: "text"}}`
(inputs are `"number"`, `"text"`, `"yes/no"` or `"list"`).
`examples/agent.ant` is a complete agent.

### Commands, the web, and agents that act

```
r = shell("git status")                  # {output, errors, code}
show r.output

show read_page("https://example.com")    # a web page's readable text
found = json(fetch("https://en.wikipedia.org/w/api.php",
                   {query: {action: "query", list: "search", srsearch: "Sigiriya", format: "json"}}))

model coder = "ollama:qwen2.5-coder" with {tools: [shell, read, write, files]}
ask coder "Run the tests and fix what fails."
```

`shell(command, {folder, input, timeout})` runs a command (in `cmd` on Windows,
`sh` elsewhere); at the timeout it is stopped with everything it started. `fetch(url, {query, method, headers, body, timeout})` gets
anything from the web as text; `page_text` and `page_links` take a page apart,
and `read_page` fetches a page as text in one step. `shell`, `read_page`,
`fetch`, `read`, `write` and `files` can be tools just as they are. Before a
model runs a command or writes a file, Antares shows what it wants to do and
asks `Allow it? (y/n)`; `confirm: yes` in a tool's map asks first for your own
tools too, and `{run: shell, confirm: no}` turns it off.
`examples/researcher.ant` answers questions from Wikipedia, and
`examples/coder.ant` is a programming helper that runs commands to check its
work.

### Images, streaming and embeddings

```
show ask claude "What is in this picture?" with image("photo.jpg")

model talker = "anthropic:claude-opus-5-5" with {stream: yes}
ask talker "Tell me a short story."      # the reply appears word by word

model embedder = "openai:text-embedding-3-small"
notes = embed(embedder, ["The museum is closed on Mondays.", "Bus 138 goes to Colombo."])
scores = similarity(embed(embedder, "When can I visit the museum?"), notes)
show top(scores, 1)                      # [0]: the first note is the closest
```

`image(file)` reads a .png, .jpg, .gif or .webp file; put several in a list
after `with`. `embed` gives one row of numbers per text, so that texts with
similar meanings get similar rows, and `similarity` compares them.
Anthropic doesn't make embeddings, so use OpenAI, Ollama (for example
`"ollama:nomic-embed-text"`) or `"echo"`. `examples/search.ant` answers
questions from your own notes this way, and `examples/chat.ant` is a chatbot.

Requests are sent with the `curl` program that comes with Windows, macOS and
Linux, and API keys never appear in the list of running programs.

## Games in a window

On Windows, Antares can open a window and draw in it, with nothing else
installed: smooth shapes, see-through colours, gradients, pictures from PNG
files, real fonts, soft shadows and lighting effects, or chunky pixel art.
Small games are scaled up to fit the screen and stay sharp:

```
window(160, 120, "Dodge")                # 160 x 120 pixels, scaled to fit
x = 72
while window_open() {
    when held("left") { x -= 2 }
    when held("right") { x += 2 }
    fill("navy")
    rect(x, 100, 16, 6, "orange")
    label("Hello!", 4, 4, "yellow")
    frame()                              # show it; 60 frames a second
}
```

And a scene with light:

```
window(640, 360, "Glow")
ship = picture("ship.png")
while window_open() {
    fill(gradient(0, 0, 0, 360, ["#0b1d3a", "#f08a5d"]))
    blend("add")                         # light adds up
    circle(320, 180, 90, radial(320, 180, 90, ["#ffd27f", "#ffd27f00"]))
    blend()
    shadow(4, 6, 10)                     # soft shadows from here on
    draw(ship, 280, 150, {angle: 20, scale: 2})
    shadow()
    font("Segoe UI", 28, "bold")
    label("Hello!", 20, 20, "white")
    glow(1)                              # bright parts shine
    frame()
}
```

| Function | What it does |
|---|---|
| `window(width, height [, title [, scale]])` | Opens a window of `width` by `height` pixels, scaled to fit the screen (or by `scale`) |
| `window_open()` | `yes` until the player closes the window |
| `frame()` | Shows what you've drawn, then waits, so the game runs at 60 frames a second; gives the seconds since the last frame |
| `fullscreen([yes or no])` | Fills the screen, or goes back to a window |
| `fill([colour])` | Paints the whole picture (black if no colour is given) |
| `rect(x, y, width, height [, colour [, radius]])` | A filled rectangle, its top-left corner at `x`, `y`, with rounded corners if a radius is given |
| `circle(x, y, radius [, colour])`, `ellipse(x, y, rx, ry [, colour])` | A filled circle or ellipse |
| `ring(x, y, radius, thickness [, colour])` | The outline of a circle |
| `polygon(points [, colour])` | A filled shape through `[[x, y], [x, y], ...]` |
| `line(x1, y1, x2, y2 [, colour [, width]])`, `path(points [, colour [, width]])` | Lines of any thickness |
| `pixel(x, y [, colour])`, `colour_at(x, y)` | Sets or reads one pixel |
| `label(text, x, y [, colour [, size]])` | Text: the built-in pixel font, or the font chosen with `font` |
| `font([name [, size [, style]]])`, `measure(text [, size])` | Chooses a font (`"Segoe UI"`, 24, `"bold"`); the size a label takes |
| `sprite(rows, x, y, colours [, size])` | Pixel art written as rows of text |
| `picture(file)`, `picture(width, height)` | Reads a PNG file; makes a new picture to draw on |
| `draw(picture, x, y [, options])` | Draws a picture: options `scale`, `width`, `height`, `angle`, `alpha`, `flip`, `part`, `tint`, `smooth` |
| `canvas([picture])` | Draws on a picture instead of the window; `canvas()` goes back |
| `gradient(x1, y1, x2, y2, colours)`, `radial(x, y, radius, colours)` | Colours changing smoothly, used anywhere a colour goes |
| `hsl(hue, saturation, lightness [, alpha])`, `mix(colour1, colour2, amount)` | Makes colours |
| `blend([mode])` | `"add"` (light), `"multiply"` (darken), `"screen"` or `"normal"` |
| `smooth(yes or no)` | Smooth edges, or hard pixel edges for pixel art |
| `shadow([across, down, softness [, colour]])` | Soft shadows for what is drawn next |
| `glow(amount)`, `blur(radius)`, `vignette(amount)` | Effects on the whole picture |
| `held(key)` | `yes` while a key is held down |
| `pressed(key)` | `yes` if a key went down since the last frame |
| `mouse()` | `{x, y, left, right}`: where the mouse is, in the game's pixels, and which buttons are held |
| `screenshot(file)` | Saves the picture as a PNG file |

- Colours are names (`"red"`, `"gold"`, `"skyblue"`, `"lightgray"` and the
  other usual web colour names, and `"transparent"`), codes (`"#ff8800"`,
  `"#f80"`, `"#ff880080"` with an alpha) or `[red, green, blue]` with each
  from 0 to 255, and an optional fourth part, the alpha, from 0
  (see-through) to 1 (solid). The colour is optional, and shapes and text
  are white without one.
- Positions can have fractions, and shapes have smooth edges; `smooth(no)`
  gives hard pixel edges for the classic look. `examples/night.ant` shows
  gradients, light, shadows and fonts together.
- Pictures, and drawing on them, work on every system, so programs on Linux
  and macOS can make PNG files; windows are Windows-only for now.
- Keys are named `"left"`, `"right"`, `"up"`, `"down"`, `"space"`,
  `"enter"`, `"escape"`, `"a"` to `"z"`, `"0"` to `"9"`, `"shift"`, `"ctrl"`
  and `"f1"` to `"f12"`; `"click"` and `"right click"` are the mouse buttons.
  `key()` and `wait()` work in a window too.
- A sprite is pixel art written as text, one character a pixel, with a map
  saying which character is which colour. Any other character shows what's
  behind: `sprite([".r.", "rrr"], x, y, {r: "red"}, 4)` draws it four times
  as big.
- Closing the window ends the game. A game made with `antares build` and
  started by double-clicking hides its console window, and shows a problem
  that stops it in a message box. With `--icon`, the window shows the icon too.
- `examples/catch.ant` is a complete game, with sound: catch the falling
  stars and dodge the rocks.

Windows only for now; on macOS and Linux, `window()` says so.

## 3D

Antares draws 3D worlds into the same window, on the graphics card
(through Direct3D 11, which every Windows PC has; Windows' own software
renderer stands in on a PC without one). Light behaves the way real light
does: the sun casts soft shadows, the sky lights everything from above,
lamps glow, and surfaces are rough or polished, plain or metal. Edges are
smooth, and fog fades the distance.

```
window(960, 540, "3D")
t = 0
while window_open() {
    t += frame()
    camera(0, 3, 8, 0, 0.5, 0)               # where from, and looking at what
    sky("#4f86d9", "#d8e6f3")
    sun(0.5, 1, 0.3)
    plane(0, 0, 0, 40, 40, "#6f9a4f")        # the ground
    cube(0, 0.75, 0, 1.5, {colour: "red", turn: t * 45})
    sphere(2.5, 1, 0, 1, {colour: "white", metal: 1, shine: 0.9})
    render()                                 # draw it; 2D goes on top
    label("Hello, 3D", 20, 20)
}
```

| Function | What it does |
|---|---|
| `camera(x, y, z, look_x, look_y, look_z [, {fov, near, far}])` | Where the camera is, and what it looks at (`y` is up) |
| `cube`, `box`, `sphere`, `cylinder`, `cone`, `plane` | Shapes, placed by their middle: `cube(x, y, z, size [, look])` |
| `mesh(file)`, `mesh(points, triangles)` | A model from an `.obj` file, or built from corners and triangles |
| `mesh(model, points)` | The same model with its corners moved, every frame if you like: water, flags, cloth |
| `draw(model, x, y, z [, look])` | Draws a model |
| `sun(x, y, z [, colour [, strength]])` | Sunlight from that direction, with soft shadows |
| `lamp(x, y, z [, colour [, strength [, reach]]])` | A light at a point (up to 8) |
| `sky(top [, horizon [, ground]])`, `fog([colour,] near, far)` | The sky, and fog |
| `shadows(yes or no)`, `render()`, `to_screen(x, y, z)` | Shadows on or off; draws the scene; where a 3D point shows on screen |

A look is a colour or a map: `{colour, shine, metal, glow, ripples, picture,
tiles, turn, scale, shadow}`; `ripples: 1` makes a shiny surface look like
water. `examples/world.ant` is a village to roll a ball
around, by day and by night.

## Sound

Games can play sound files, tones, tunes written as text, ready-made effects
and background music, all at once. Antares mixes the sounds itself, so
nothing else is needed. Sound works in a window or in the terminal:

```
effect("coin")                           # a ready-made effect
beep(440, 0.2)                           # a tone: frequency, seconds
tune("C E G C5:2", 0.15)                 # a tune written as text
sound("splash.wav")                      # a sound file
music("C - E - G - E -", 0.2, "triangle")    # a tune that loops
```

| Function | What it does |
|---|---|
| `sound(file [, volume])` | Plays a WAV file once. Sounds can overlap |
| `music(file)` | Plays a WAV file over and over, in the background |
| `music(notes, step [, wave])` | Plays a tune over and over, in the background |
| `music()` | Stops the music |
| `beep(frequency, seconds [, wave])` | A tone. 440 is the note A |
| `tune(notes [, step [, wave]])` | Plays a tune written as text, once; gives back its length in seconds |
| `effect(name)` | A ready-made sound effect: `"coin"`, `"jump"`, `"hit"`, `"laser"`, `"explosion"`, `"powerup"` or `"select"` |
| `volume(level)` | How loud everything is, from 0 (silent) to 1 (full) |
| `silence()` | Stops every sound, and the music |
| `playing()` | How many sounds are still playing (not counting music) |

- A tune is notes separated by spaces: `C`, `F#`, `Bb3`, `A5` (octave 4 if
  none is given) and `-` for a rest. Each lasts one step (0.15 seconds unless
  given), or several with `:n`, like `C:2`.
- Waves give tones their character: `"square"` (the default, the classic
  game sound), `"sine"` (soft), `"triangle"`, `"saw"` and `"noise"`.
- Sound files are WAV files (8, 16, 24 or 32-bit, any rate, mono or stereo).
  To pack one into a built game, build with `--embed splash.wav` and play
  `sound("embedded:splash.wav")`.
- Sounds start at most 60 ms after they are asked for. A program waits for
  its last sounds to finish before it ends, so `beep(...)` as its last line is
  heard.
- Windows only for now. Elsewhere, and on computers without a sound device,
  the sound functions do nothing, so a game still runs, silently.

## Built-in functions

| Area | Functions |
|---|---|
| Make tensors | `zeros ones rand randn range onehot seed` |
| Shape | `shape reshape t len` |
| Reduce | `sum mean max min argmax` |
| Maths | `exp log sqrt abs sin cos floor round trunc clip tanh sigmoid relu softmax` |
| Fast row arithmetic | `sum_rows update_rows rows sum_product screlu screlu_dot` |
| Integers | `int ints popcount lowest_bit highest_bit div rand_bits within` |
| Compact numbers | `compact` (whole numbers stored in 2 bytes instead of 8, for the row built-ins), `expand` |
| Learning | `grad` (plus the `learn` and `descend` statements) |
| Deep learning | `gelu silu softplus leaky_relu log_softmax cross_entropy mse bce take concat layer_norm dropout causal_mask he xavier accuracy top sample` |
| Data | `read_csv columns shuffle batches` |
| Models | `conversation history image embed similarity` (plus the `model` and `ask` statements) |
| GPU | `gpu` (plus the `use gpu` and `use cpu` statements) |
| Lists and maps | `push insert remove find unique map filter reduce sort sort_by reverse slice has keys values join` |
| Text | `upper lower trim split lines replace find text number fixed pad code char` |
| Data and files | `save load exists program_folder json to_json read write append delete files read_binary input input_waiting` |
| The command line | `args` (what was typed after the program's name), `env` (a setting of the computer) |
| Commands and the web | `shell fetch read_page page_text page_links` |
| Games and live screens | `key` (the key being pressed, or nothing), `wait` (pause, in seconds), `draw` (redraw the screen in place), `clear` |
| Games in a window | `window window_open frame fullscreen fill rect circle ellipse ring polygon line path pixel colour_at label font measure sprite picture draw canvas gradient radial hsl mix blend smooth shadow blur glow vignette held pressed mouse screenshot` |
| 3D | `camera cube box sphere cylinder cone plane mesh draw sun lamp sky fog shadows render to_screen` |
| Sound | `sound music beep tune effect volume silence playing` |
| Other | `kind clock now check fail` |

## Speed

Antares translates each function once into compact instructions and runs
them in a tight loop. Measured on one Windows laptop, best of 7 runs
(`python bench/bench.py` repeats this on your machine):

| Test | Antares | Python 3.12 | Node 24 |
|---|---|---|---|
| `fib(30)`, 1.3 million function calls | 0.28 s | 0.35 s | 0.013 s |
| 5 million additions in a loop | 0.17 s | 0.96 s | 0.007 s |
| `while` loop with a condition, 3 million rounds | 0.31 s | 0.93 s | 0.012 s |
| 3 million list lookups | 0.55 s | 0.92 s | 0.008 s |
| Multiplying 300x300 matrices three times | 0.03 s | 14.7 s (plain lists) | 0.15 s |

Antares is faster than Python on all of these (1.3 to 5.5 times on ordinary
code). It is much slower than Node on ordinary code, because Node turns hot
code into machine code while it runs and Antares does not. Tensor maths runs
as compiled code, which is why the matrix test beats both. Large matrix
multiplications, and their gradients, are shared across all the CPU's cores:
training a 256-512-10 network takes 5.5 times less time on 4 cores than on
one, with exactly the same results. With `use gpu`, training runs on the
graphics card instead (see [Training on the GPU](#training-on-the-gpu)).

## Compiling

`antares build program.ant --native` turns a program into a native executable
placed next to the source file (this is what `antares build` does when Rust is
installed). The program behaves the same as when interpreted;
`python tests/compare_compiled.py` checks this on the test suites and examples,
and `python tests/compare_compiled.py --pack` does the same for packed programs.

The compiler looks at the whole program and works out which variables only
ever hold a plain number, a yes/no, or a plain row of numbers. Those become
machine numbers and direct memory, and functions called by name become direct
calls. Everything else (text, lists, maps, tensors being learned, functions
passed around as values) keeps working through the same runtime the
interpreter uses.

| Test | Interpreted | Compiled | Python 3.12 | Node 24 |
|---|---|---|---|---|
| `fib(30)` | 0.28 s | 0.009 s | 0.35 s | 0.013 s |
| Chess move generation (perft), positions per second | 25,000 | 5,000,000 to 6,000,000 | | |
| Chess engine search, positions per second | 4,000 | 275,000 to 292,000 | | (C++ original: 490,000 generic, 1,160,000 avx2) |

Things to know:

- It needs Rust (`cargo`) installed, because the compiler writes Rust code and
  builds it. A build takes about 30 to 40 seconds. Without Rust, use `--pack`
  (see [Making an executable to share](#making-an-executable-to-share)): the
  program then runs at interpreter speed, but needs nothing to build.
- Packed programs work on Windows, macOS and Linux: every change is checked
  to make sure packed programs behave exactly like interpreted ones on all
  three.
- The compiled program is built against the Antares source code. `antares
  build` uses the folder named by the `ANTARES_HOME` environment variable if
  set, then the folder this copy of Antares was built from if it is still
  there; otherwise it downloads the source of the matching release from
  GitHub once and keeps it in your cache folder.
- By default the program is built for the processor of the machine that
  compiles it, so it may not run on an older one. Choose another target with
  `antares build program.ant out.exe --cpu x86-64` (any 64-bit PC),
  `x86-64-v2` (SSE4.2) or `x86-64-v3` (AVX2).
- A global given one literal number at the top level and never changed
  (`SIZE = 64`) is treated as a constant, and a list made only of such
  numbers is built once. Both make arithmetic like `x % SIZE` much cheaper.
- Whole-number counters (`repeat ... as i`, `each i in range(...)`, and
  variables only ever changed by `i = i + 1` and the like) are machine
  integers, so positions like `row[i + 1]` cost only a range check.
- A top-level row made once with `ints(n)` or `zeros(n)`, with a size known
  when compiling, and afterwards only used as `t[i]` and `t[i] = v`, becomes a
  fixed table, the fastest kind of row. Using the row as a whole anywhere
  (`len(t)`, passing `t` to a function, `each x in t`) turns this off.
- The innermost `each i in range(...)`, `repeat ... as i` or counting `while`
  loop (`while i < n { ...; i = i + 1 }`) that uses rows and fixed tables only
  as `row[i + c]` checks positions once, before it starts.
- A fixed table can be cleared by giving it `ints(n)` / `zeros(n)` of the same
  size again, and `ints([...])` of constant numbers makes one that starts with
  them. A top-level variable given, once, arithmetic on constants given before
  it (`AH = HIST + 1`) is a constant too.
- Rows of a compact matrix are read and changed by position:
  `m[d] = update_rows(m[s], w, ...)`, `m[d] = m[s]` and
  `screlu_dot(m[r], w[k], cap)` run in place on the matrix in a compiled
  program, with no row built in between.
- A position into a fixed table that is certainly inside it needs no check.
  Where the compiler can't work that out, give a range:
  `board = ints(64) within 0 to 15`, `count = int(0) within 0 to 255`,
  `to put(p within 0 to 15, sq within 0 to 63)`. Values are checked where
  they are given (an input's check moves to the calls, and is left out when
  the value is known to fit), and uses as positions are then free.
- One value of unknown kind can slow down everything it touches. If a number
  comes out of a map, a list or JSON, wrap it in `number(...)` so the compiler
  knows it is a number. In the chess engine, one such lookup made the whole
  board slow until it was wrapped.
- A number variable that is read before it is given a value reads as 0 in a
  compiled program, where the interpreter reports a problem.
- `--embed file` packs a file into the program. The program reads it under
  the name `"embedded:<file name>"` with `read_binary`, and `exists` says
  whether it is there, so the same source can fall back to a file on disk
  when run interpreted. The chess engine embeds its 25 MB network this way.
- `--icon file.ico` gives a Windows executable its icon (when compiling with
  the GNU Rust toolchain it needs `windres`; packing needs nothing).
- To see why something is slow, build with the environment variable
  `ANTARES_EXPLAIN=1`. The compiler then names every variable that ended up
  as a general value and what it was given. A common cause is two files using
  the same global name for different things.

## How it is built

| File | What it does |
|---|---|
| `src/lexer.rs` | Source text to tokens |
| `src/parser.rs` | Tokens to a tree; resolves every name to a slot up front |
| `src/aot.rs` | The compiler behind `antares build --native`: type inference and code generation |
| `src/pack.rs` | `antares build --pack`: attaches a program to a copy of `antares`, and finds it again at startup |
| `src/rt.rs` | Support code linked into compiled programs |
| `src/vm.rs` | The fast engine: translates functions to instructions and runs them |
| `src/interp.rs` | Values, indexing, and the fallback evaluator for rarer constructs |
| `src/tensor.rs` | Tensors, broadcasting, reverse-mode autodiff, the optimisers, and matrix maths on every core |
| `src/nn.rs` | Deep learning and data: activations, losses, layers, starting weights, sampling, CSV and batches |
| `src/gpu.rs` | The GPU: Direct3D 11 compute shaders for matrix products, element-wise maths, gradients and optimisers, and the shader cache |
| `src/gpu_cuda.rs` | The same on Linux with NVIDIA GPUs (Colab, Kaggle), through CUDA: kernels compiled with libnvrtc |
| `src/builtins.rs` | The built-in functions |
| `src/gfx.rs` | Games in a window: drawing, the built-in font, PNG screenshots, and the Windows window and its keyboard and mouse |
| `src/sound.rs` | Sound: the mixer, WAV files, tones, tunes and effects, and Windows sound output |
| `src/llm.rs`, `src/json.rs` | Language models (Claude, OpenAI-style, Ollama): conversations, tools, shapes, images, streaming, embeddings; and JSON |
| `src/web.rs` | Commands (`shell`) and the web (`fetch` through `curl`, readable text and links from pages) |

## A real program: a chess engine

The NARC Genesys 5.1 chess engine has been re-written in Antares: about
2,000 lines covering bitboard move generation, the neural-network evaluation
and the full search, verified against the original C++ engine. It lives in
its own project, separate from this repository. Compiled,
it searches about 285,000 positions per second, roughly 60% of the speed of
the original C++ generic build.

## Current limits

This is version 0.13. Not there yet: giving a value to a slice
(`v[1:3] = 0`), the GPU on macOS and on non-NVIDIA Linux, and game windows,
3D and sound on macOS and Linux.

## Rebuilding the documentation

The language reference is generated from [`docs/src/manual.md`](docs/src/manual.md)
and the beginner's book from [`docs/src/tutorial.md`](docs/src/tutorial.md):

```bash
python docs/src/build.py
python docs/src/build.py tutorial
```

It needs Python with `markdown`, `pygments` and `pymupdf`, and Chrome or Edge.
Every example in the book shows what Antares really prints; check that they
still do after changing the language with:

```bash
python docs/src/check_examples.py
```

## License

Antares is released under the [MIT License](LICENSE).

