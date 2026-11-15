# Antares

[![CI](https://github.com/Madushan996/Antares-Programming-Language/actions/workflows/ci.yml/badge.svg)](https://github.com/Madushan996/Antares-Programming-Language/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A small programming language for AI. Tensors, automatic differentiation and
language-model calls are part of the language itself, not libraries.

> **Status: experimental preview (0.1).** Antares is ready to try, learn and
> experiment with. The language may still change between versions, so programs
> written today may need small updates later. Feedback and bug reports are very
> welcome in [Issues](https://github.com/Madushan996/Antares-Programming-Language/issues).

📘 **Full documentation:** [Antares Language Reference (PDF)](docs/Antares_Language_Reference.pdf)
covers the whole language, every built-in function, and complete examples.

📗 **New to programming?** [Learn Antares: A Beginner's Guide (PDF)](docs/Learn_Antares.pdf)
teaches programming from the very first step, with exercises and answers.

- **One file, no dependencies.** The whole language is a single ~680 KB
  program written in Rust with zero external crates. It starts instantly.
- **Reads like plain verbs.** `learn`, `descend`, `ask`, `show`, `repeat`,
  `each`, `when`, `to`, `give`.
- **Friendly errors.** Every problem names the line, shows it, and says what
  to do in plain words.

```
learn slope = 0
learn offset = 0

repeat 400 {
    loss = mean((slope * x + offset - y) ^ 2)
    descend loss by 0.05
}
```

```
model claude = "anthropic:claude-opus-5-5"
count = ask claude "How many moons does Mars have?" as number
```

## Contents

- [Install](#install)
- [Your first program](#your-first-program)
- [Building something: a step-by-step example](#building-something-a-step-by-step-example)
- [Everyday commands](#everyday-commands)
- [Making an executable to share](#making-an-executable-to-share)
- [Troubleshooting](#troubleshooting)
- [The language in five minutes](#the-language-in-five-minutes)
- [Machine learning](#machine-learning) · [Language models](#language-models) · [Built-in functions](#built-in-functions)

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

You should see `Antares 0.6.0`.

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

## Machine learning

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

Gradients flow through: `+ - * / ^ @`, indexing, `exp log tanh sigmoid relu
sqrt abs sin cos`, `sum mean softmax t reshape`, and slices.

## Language models

```
model claude = "anthropic:claude-opus-5-5" with {system: "Be brief."}

answer = ask claude "Explain {topic} in one sentence."
count  = ask claude "How many moons does Mars have?" as number
sure   = ask claude "Is 91 a prime number?" as yes/no
ideas  = ask claude "Give me five names for a cat." as list
```

`as number`, `as yes/no` and `as list` turn the reply into a real value you
can calculate with. Set the `ANTHROPIC_API_KEY` environment variable first.
Requests are sent with the `curl` program that comes with Windows, macOS and
Linux. Options after `with`: `system` (text) and `max_tokens` (number).

`model fake = "echo"` makes a pretend model that repeats the prompt, for
trying things offline (see `examples/ask.ant`).

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
| Lists and maps | `push insert remove find unique map filter reduce sort sort_by reverse slice has keys values join` |
| Text | `upper lower trim split lines replace find text number fixed pad code char` |
| Data and files | `save load exists program_folder json to_json read write append delete files read_binary input input_waiting` |
| The command line | `args` (what was typed after the program's name), `env` (a setting of the computer) |
| Games and live screens | `key` (the key being pressed, or nothing), `wait` (pause, in seconds), `draw` (redraw the screen in place), `clear` |
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
as compiled code, which is why the matrix test beats both.

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
- Packed programs are checked on Windows. On macOS, attaching data to a
  program may upset its code signature; compile with `--native` there if a
  packed program won't start.
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
| `src/tensor.rs` | Tensors, broadcasting, reverse-mode autodiff, SGD and Adam |
| `src/builtins.rs` | The built-in functions |
| `src/llm.rs`, `src/json.rs` | Model calls and JSON |

## A real program: a chess engine

The NARC Genesys 5.1 chess engine has been re-written in Antares: about
2,000 lines covering bitboard move generation, the neural-network evaluation
and the full search, verified against the original C++ engine. It lives in
its own project, separate from this repository. Compiled,
it searches about 285,000 positions per second, roughly 60% of the speed of
the original C++ generic build.

## Current limits

This is version 0.6. Not there yet: giving a value to a slice
(`v[1:3] = 0`), tools and multi-turn chat for models, model providers other
than Anthropic, and GPU support.

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

