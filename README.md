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

## Build and run

```bash
cargo build --release
```

```bash
./target/release/antares examples/hello.ant
```

To compile a program to a fast native executable (see "Compiling" below):

```bash
antares build program.ant
```

Run `antares` with no file for an interactive session. Programs use the
`.ant` extension. `tests/check.ant`, `tests/features.ant`, `tests/engine.ant` and `tests/integers.ant` are self-check suites.

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
Write `\{` for a plain brace. Comments start with `#`.

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
to bump() { change count = count + 1 }
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
| Integers | `int ints popcount lowest_bit highest_bit div rand_bits` |
| Compact numbers | `compact` (whole numbers stored in 2 bytes instead of 8, for the row built-ins), `expand` |
| Learning | `grad` (plus the `learn` and `descend` statements) |
| Lists and maps | `push map filter reduce sort reverse slice has keys values join` |
| Text | `upper lower trim split lines replace text number` |
| Data | `save load exists program_folder json to_json read write read_binary input input_waiting` |
| Games and live screens | `key` (the key being pressed, or nothing), `wait` (pause, in seconds), `draw` (redraw the screen in place), `clear` |
| Other | `kind clock check fail` |

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

`antares build program.ant` turns a program into a native executable placed
next to the source file. The program behaves the same as when interpreted;
`python tests/compare_compiled.py` checks this on the test suites and examples.

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

- It needs Rust (`cargo`) installed and the Antares source folder, because
  the compiler writes Rust code and builds it. A build takes about 30 to 40
  seconds.
- By default the program is built for the processor of the machine that
  compiles it, so it may not run on an older one. Choose another target with
  `antares build program.ant out.exe --cpu x86-64` (any 64-bit PC),
  `x86-64-v2` (SSE4.2) or `x86-64-v3` (AVX2).
- A global given one literal number at the top level and never changed
  (`SIZE = 64`) is treated as a constant, and a list made only of such
  numbers is built once. Both make arithmetic like `x % SIZE` much cheaper.
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
- To see why something is slow, build with the environment variable
  `ANTARES_EXPLAIN=1`. The compiler then names every variable that ended up
  as a general value and what it was given. A common cause is two files using
  the same global name for different things.

## How it is built

| File | What it does |
|---|---|
| `src/lexer.rs` | Source text to tokens |
| `src/parser.rs` | Tokens to a tree; resolves every name to a slot up front |
| `src/aot.rs` | The compiler behind `antares build`: type inference and code generation |
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

This is version 0.1. Not there yet: giving a value to a slice
(`v[1:3] = 0`), tools and multi-turn chat for models, model providers other
than Anthropic, and GPU support.

Known platform gaps:

- `key()` and `input_waiting()` only work on Windows so far; on macOS and
  Linux they always report no key, so live terminal games cannot be steered.
- `antares build` must be run with an `antares` binary built from a source
  folder on the same machine, because it compiles against that folder.

## Rebuilding the documentation

The PDF is generated from [`docs/src/manual.md`](docs/src/manual.md):

```bash
python docs/src/build.py
```

It needs Python with `markdown`, `pygments` and `pymupdf`, and Chrome or Edge.

## License

Antares is released under the [MIT License](LICENSE).

