# Introduction {#intro}

Antares is the programming language for artificial intelligence. Tensors, automatic differentiation, the building blocks of deep learning (attention, layer normalisation, cross-entropy, AdamW) and calls to large language models (Claude, OpenAI and local models) are part of the language itself, not libraries you have to install and import. A model that learns, a GPT-style transformer, a chatbot that remembers, and an agent that calls your own functions are each a few lines of plain, readable code.

## Design principles

**One file, no dependencies.** The whole language is a single program of about 680 KB, written in Rust with zero external crates. It starts instantly and can be copied to another machine as one file.

**Reads like plain verbs.** The core statements are ordinary English words: `learn`, `descend`, `ask`, `show`, `repeat`, `each`, `when`, `to`, `give`. A newcomer can usually guess what a program does before learning the rules.

**Friendly errors.** Every problem names the line, shows it, and says what to do in plain words. Misspelled names come with a suggestion.

**Fast enough to be real.** Functions are translated once into compact instructions for a fast virtual machine, and `antares build` compiles a whole program to a native executable. A full chess engine has been written in Antares and runs at about 60% of the speed of its C++ original.

## A first taste

Fitting a straight line to data is a complete machine-learning program:

```antares
learn slope = 0
learn offset = 0

repeat 400 {
    loss = mean((slope * x + offset - y) ^ 2)
    descend loss by 0.05
}
```

`learn` marks a value the program may adjust. `descend` adjusts every learnable value that `loss` depends on so that `loss` gets smaller. There is no separate framework, optimiser object or gradient tape: that is the whole training interface.

Asking a language model is just as direct, and the answer can come back as a real value:

```antares
model claude = "anthropic:claude-opus-5-5"
count = ask claude "How many moons does Mars have?" as number
show count + 1
```

## About this manual

This manual describes **Antares 0.13.1**. Chapters 2 to 9 cover the core language: values, operators, control flow, functions, errors and modules. Chapters 10 to 15 cover what makes Antares different: tensors, learning, language models and agents, integers, files, commands and the web, and games in the terminal and in a window, in 2D and 3D, with sound. Chapter 16 explains performance and the native compiler. Chapter 17 is a complete reference of built-in functions, and the appendices give a grammar summary and complete example programs.

Code is shown in boxes like this. Output is shown in a lighter box underneath where it helps:

```antares
show "Hello, world!"
```

```output
Hello, world!
```

# Getting Started {#start}

## Installing Antares

Ready-made downloads for Windows, macOS and Linux are published on the project's GitHub releases page, github.com/Madushan996/Antares-Programming-Language. The quickest way to install is the one-line installer.

On Windows, in PowerShell:

```shell
irm https://raw.githubusercontent.com/Madushan996/Antares-Programming-Language/main/install.ps1 | iex
```

On macOS or Linux:

```shell
curl -fsSL https://raw.githubusercontent.com/Madushan996/Antares-Programming-Language/main/install.sh | sh
```

The installer puts `antares` in your user folder and adds it to your PATH. Open a new terminal afterwards and check it with `antares version`. The README on GitHub also explains how to download and install by hand.

## Building from source

Antares can also be built from source with Rust's `cargo` tool. From the Antares folder:

```shell
cargo build --release
```

The finished program is `target/release/antares` (`antares.exe` on Windows). Copy it anywhere on your `PATH`, or run it in place. The release build uses full link-time optimisation and is stripped, so it stays small.

## Running a program

Antares programs are plain text files with the `.ant` extension.

```shell
antares examples/hello.ant
```

`antares run program.ant` does the same thing. The program runs from top to bottom; there is no `main` function to write.

## The interactive session

Run `antares` with no file to get an interactive session (a REPL). Type a line and press Enter; the value of each expression is printed. If a line opens a bracket or brace that is not yet closed, the prompt changes from `>` to `.` and Antares keeps reading until it is balanced, so functions and loops can be typed over several lines. Type `exit` or `quit` to leave.

```session
Antares 0.13.1 — type a line and press Enter. `exit` to leave.
> v = [1, 2, 3]
> v * 10
[10, 20, 30]
> to square(x) = x * x
> square(12)
144
```

## Command-line reference

| Command | What it does |
|---|---|
| `antares` | Start an interactive session |
| `antares program.ant` | Run a program |
| `antares run program.ant` | The same |
| `antares build program.ant [output] [options]` | Make an executable to share: compiled, or packed when Rust isn't installed (Chapter 16) |
| `antares version` | Show the version (also `--version`, `-v`) |
| `antares help` | Show a short summary of commands (also `--help`, `-h`) |

Options for `build`:

| Option | Meaning |
|---|---|
| `--cpu native` | Build for this computer's processor (the default) |
| `--cpu x86-64` | Build for any 64-bit PC |
| `--cpu x86-64-v2` | Processors with SSE4.2 |
| `--cpu x86-64-v3` | Processors with AVX2 |
| `--embed file` | Pack a file into the executable (may be repeated) |
| `--icon file.ico` | The icon Windows shows for the executable (see below) |

## Your first program

Create `hello.ant`:

```antares
name = "world"
show "Hello, {name}!"

to square(x) = x * x
show square(7)

a = [1, 2, 3]
show a * 2 + 1
```

```output
Hello, world!
49
[3, 5, 7]
```

## Self-checks and examples

The `examples` folder holds short programs that show each part of the language; Appendix B reproduces several of them. The `tests` folder holds self-check suites written in Antares itself. Each prints one line when every check passes:

```shell
antares tests/check.ant
antares tests/features.ant
antares tests/engine.ant
antares tests/integers.ant
```

```output
all checks passed
all feature checks passed
all engine checks passed
all integer checks passed
```

# Lexical Structure {#lexical}

## Lines and statements

A program is a sequence of statements, normally one per line. Blocks are written with braces `{ }`. A block may share a line with its statement or spread over several lines:

```antares
when ready { show "go" }

when ready {
    show "go"
}
```

A line that starts with `->` continues the line above, which lets long pipelines be laid out one step per line (see Section 5.8). Inside parentheses and square brackets, an expression may run over several lines.

## Comments

A comment starts with `#` and runs to the end of the line.

```antares
rate = 0.05      # how big each learning step is
```

## Names

Names (identifiers) start with a letter or underscore and continue with letters, digits or underscores: `x`, `learning_rate`, `w1`, `_tmp`. Names are case-sensitive.

## Keywords

These words have a fixed meaning and cannot be used as names:

| | | | | | |
|---|---|---|---|---|---|
| `and` | `as` | `ask` | `by` | `change` | `descend` |
| `each` | `else` | `give` | `in` | `learn` | `model` |
| `no` | `not` | `nothing` | `or` | `repeat` | `show` |
| `skip` | `stop` | `to` | `try` | `use` | `using` |
| `when` | `while` | `with` | `xor` | `yes` | |

## Number literals

Ordinary numbers are 64-bit floating-point values. They may have a fraction and an exponent, and underscores may be used to group digits:

```antares
42      3.14159      1e-9      2.5E3      1_000_000
```

A literal starting with `0x` is a hexadecimal **64-bit integer** (Chapter 13). Up to 16 hex digits are allowed, and underscores may be used here too:

```antares
0xFF      0x8000_0000_0000_0000
```

## Text literals

Text is written in double quotes. These escapes are recognised:

| Escape | Meaning |
|---|---|
| `\n` | New line |
| `\t` | Tab |
| `\"` | A double quote |
| `\\` | A backslash |
| `\{` | A plain opening brace (not the start of a value) |

**Interpolation.** Any expression inside `{ }` in a text literal is evaluated and inserted:

```antares
name = "Ada"
age = 36
show "Hello {name}, next year you are {age + 1}"
show "Shouting: {upper(name)}"
```

```output
Hello Ada, next year you are 37
Shouting: ADA
```

To write a literal brace, for example inside JSON, escape the opening brace with `\{`. A text literal can also span several lines.

# Values and Types {#values}

## The kinds of value

Every value has a kind. The built-in function `kind(v)` returns its name as text:

| Kind | Examples | Notes |
|---|---|---|
| `number` | `36`, `-0.5`, `1e-3` | 64-bit floating point |
| `integer` | `0xFF`, `int(12)` | 64-bit whole number, for bit work (Chapter 13) |
| `text` | `"Ada"` | Unicode text |
| `yes/no` | `yes`, `no` | Booleans |
| `nothing` | `nothing` | The absence of a value |
| `tensor` | `[1, 2, 3]`, `[[1, 2], [3, 4]]` | A list or grid of numbers |
| `list` | `["a", 1, yes]` | Any values, in order |
| `map` | `{name: "Ada", age: 36}` | Named fields, in insertion order |
| `function` | `square`, `x => x * 2` | Named or unnamed functions |
| `model` | `model claude = ...` | A language model (Chapter 12) |
| `integer row` | `ints(64)` | A fixed row of 64-bit integers |
| `compact numbers` | `compact(table)` | Whole numbers stored in 2 bytes each |

## Numbers, yes/no and nothing

```antares
age = 36
ready = yes
done = no
empty = nothing
```

`yes` and `no` are the only boolean values. `when`, `while`, `and`, `or` and `not` expect them. A function that finishes without producing a value gives `nothing`.

## Tensors: lists of numbers

In Antares a list made only of numbers **is** a tensor. There is no separate array type to convert to:

```antares
v = [1, 2, 3]             # a vector, shape [3]
m = [[1, 2], [3, 4]]      # a matrix, shape [2, 2]
show shape(m)
show m * 10
```

```output
[2, 2]
[[10, 20],
 [30, 40]]
```

All rows of a nested tensor must have the same length. Tensors can have any number of dimensions. A single number is a tensor with no dimensions, and the two are used interchangeably.

## Lists

A list that contains anything other than numbers is a general list. It can mix kinds and nest freely:

```antares
mixed = ["a", 1, yes]
people = [{name: "Ada"}, {name: "Alan"}]
```

## Maps

A map holds named fields. Keys are written as plain names; fields are read and written with a dot:

```antares
person = {name: "Ada", age: 36}
show person.name
person.age = 37
person.city = "London"          # adding a new field
show keys(person)
```

```output
Ada
["name", "age", "city"]
```

Fields keep the order in which they were added. `has(map, "key")` tests whether a field exists, and `values(map)` returns the field values.

## Value semantics: no accidental sharing

Values are never shared by accident. Giving a list to a function, or assigning it to another name, gives the receiver its own copy. Copies are made lazily, only when one side actually changes the value, so passing large tensors around is cheap.

```antares
a = [1, 2, 3]
b = a
b[0] = 99
show a, b
```

```output
[1, 2, 3] [99, 2, 3]
```

## Equality

`==` and `!=` compare values structurally: two lists, maps or tensors are equal when they have the same contents. This is how the self-check suites compare whole results in one line:

```antares
check(grad(loss, w) == [6, 12, 18], "power gradient")
```

# Expressions and Operators {#exprs}

## Arithmetic

| Operator | Meaning |
|---|---|
| `a + b` | Addition; joins text when either side is text |
| `a - b` | Subtraction |
| `a * b` | Multiplication (element by element for tensors) |
| `a / b` | Division; always gives an ordinary number |
| `a % b` | Remainder |
| `a ^ b` | Power (right-associative: `2 ^ 3 ^ 2` is `2 ^ 9`) |
| `a @ b` | Matrix multiplication |
| `-a` | Negation |

`^` binds tighter than unary minus, so `-2 ^ 2` is `-4`, as in mathematics.

When either side of `+` is text, the other side is converted to text and the two are joined: `"n=" + 5` gives `"n=5"`.

## Tensor arithmetic and broadcasting

`+ - * / % ^` work on whole tensors, element by element. When the shapes differ, the smaller shape is stretched to fit the larger one ("broadcasting"): a number combines with every element, and a row combines with every row of a matrix.

```antares
m = [[1, 2], [3, 4]]
show m * 2 + 1
show m + [10, 20]
show m @ m
```

```output
[[3, 5],
 [7, 9]]
[[11, 22],
 [13, 24]]
[[7, 10],
 [15, 22]]
```

> **Note.** Because a list of numbers is a tensor, `+` on two number lists *adds* them; it does not join them. `[1] + [2, 3]` is `[3, 4]`. To add an item to a list of numbers, use `push(list, item)`. Lists of other values (text, maps) are joined by `+`.

## Comparison

`==`, `!=`, `<`, `<=`, `>`, `>=` compare values and give `yes` or `no`. Comparisons do not chain: write `0 < x and x < 10`, not `0 < x < 10`.

## Logic

`and`, `or` and `not` combine yes/no values. `and` and `or` stop early: the right side is only evaluated when it is needed.

```antares
when n > 0 and total / n > 10 { show "large average" }
```

## Bit operators

These work on 64-bit integers (Chapter 13): `&` (and), `|` (or), `xor`, `~` (flip all bits), `<<` and `>>` (shifts).

## Operator precedence

From loosest (evaluated last) to tightest (evaluated first):

| Level | Operators | Associativity |
|---|---|---|
| 1 | `->` (pipeline) | left |
| 2 | `or` | left |
| 3 | `and` | left |
| 4 | `not` | prefix |
| 5 | `==  !=  <  <=  >  >=` | does not chain |
| 6 | `&  \|  xor` | left |
| 7 | `<<  >>` | left |
| 8 | `+  -` | left |
| 9 | `*  /  %  @` | left |
| 10 | `-` (negate), `~` (flip bits) | prefix |
| 11 | `^` | right |
| 12 | calls `f(x)`, indexing `v[i]`, fields `p.name` | left |

Bit operations bind tighter than comparisons, so `a & b != 0` means `(a & b) != 0`.

## Indexing and slicing

Indexes start at 0. Negative indexes count from the end. A matrix is indexed with a comma, `m[row, column]`, or step by step, `m[row][column]`.

```antares
v = [10, 20, 30, 40, 50]
show v[0], v[-1]
m = [[1, 2], [3, 4]]
show m[1, 0], m[1][0]
```

```output
10 50
3 3
```

A range `start:end` takes a part. The end is not included, and either side may be left out. Ranges that run past the end are clipped rather than reported as errors.

```antares
v[1:3]          # items 1 and 2: [20, 30]
v[:2]           # the first two
v[-2:]          # the last two
v[:-1]          # all but the last
m[:, 0]         # every row, first column
m[1:, :2]       # from row 1 on, first two columns
"antares"[:3]   # "ant" -- slicing works on text and lists too
```

Gradients flow through indexing and slices (Chapter 11).

## Pipelines

`->` passes the value on its left into the function on its right as the **first** input, so code reads in the order it happens:

```antares
[3, 1, 2] -> sort -> map(x => x * 10) -> sum      # 60
```

`a -> f` means `f(a)`, and `a -> f(b, c)` means `f(a, b, c)`. An unnamed function in parentheses can be a pipeline step. A line that starts with `->` continues the line above:

```antares
to net(input) = input @ w1 + b1 -> tanh -> (h => h @ w2 + b2) -> sigmoid

[3, 1, 2]
    -> sort
    -> map(x => x * 10)
    -> show_it
```

## Unnamed functions

`params => expression` makes a small function value. With one parameter the parentheses are optional:

```antares
double = x => x * 2
add = (a, b) => a + b
answer = () => 42
show double(21), add(2, 3), answer()
```

The body of an unnamed function is a single expression. For anything longer, write a named function with `to` (Chapter 7). Note that `{` after `=>` starts a map, not a block.

# Statements and Control Flow {#control}

## Assignment

`name = value` gives a name a value. At the top level of a file this creates or replaces a global; inside a function it always creates a local variable (see Section 7.5).

Parts of a list, tensor or map can be assigned directly, at any depth:

```antares
person.age = 37
person.skills[0] = "maths"
grid[0, 1] = 9
```

`+=`, `-=`, `*=` and `/=` change a value by an amount: `x += 1` is the same as `x = x + 1`. They work on names and on parts of values, and `+=` joins text too:

```antares
score = 10
score += 5                      # 15
score *= 2                      # 30
counts[word] += 1
person.name += " Lovelace"
```

With a part of a value, the positions must be names, numbers, text or arithmetic on them (`xs[i + 1] += 2`), not calls, which would otherwise run twice.

## show

`show` prints one or more values separated by spaces, followed by a new line:

```antares
show "total:", total, "items"
```

Text is printed as it is; other values are printed in Antares notation, with matrices laid out row by row.

## Decisions: when

```antares
when score > 90 {
    show "great"
} else when score > 50 {
    show "ok"
} else {
    show "hmm"
}
```

`when` takes a yes/no condition. Any number of `else when` arms may follow, and one final `else`. The `else` may start on the line after the closing brace.

A `when` statement also has a **value**: the value of the last expression in the arm that ran. This makes it work like a conditional expression at the end of a function:

```antares
to describe(n) {
    when n == 0 { "zero" } else { "positive" }
}
```

## Counting loops: repeat

`repeat count { ... }` runs the block a fixed number of times. With `as name`, a counter runs 1, 2, ... up to the count:

```antares
repeat 3 { show "hi" }
repeat 10 as step { show step }        # 1, 2, ... 10
```

## Looping over items: each

`each item in collection { ... }` runs the block once for each item of a list, the rows of a tensor, the elements of an integer row, the characters of a text, or the field names of a map.

```antares
each skill in ["maths", "engines"] { show " -", upper(skill) }
each i in range(5) { show i }          # 0, 1, 2, 3, 4
```

## Condition loops: while

```antares
while n > 0 { n = n - 1 }
while yes { ... }                      # loop until stop
```

## stop and skip

`stop` leaves the innermost loop at once. `skip` jumps to the next round of the innermost loop. Both work in `repeat`, `each` and `while`, and they pass correctly through `try` blocks.

```antares
total = 0
each n in range(10) {
    when n % 2 == 0 { skip }
    when n > 7 { stop }
    total = total + n            # 1 + 3 + 5 + 7
}
```

## Expression statements

Any expression on its own line is evaluated. In the interactive session its value is printed; in a program it is the value of the block (used as a function's result when it is the last line).

# Functions {#functions}

## Defining functions with to

A one-line function uses `=`:

```antares
to square(x) = x * x
to mse(guess, target) = mean((guess - target) ^ 2)
```

A longer function uses a block. The value of the last expression is the result:

```antares
to describe(n) {
    when n < 0 { give "negative" }
    when n == 0 { "zero" } else { "positive" }
}
```

## give: returning early

`give value` leaves the function at once with that value. `give` on its own gives `nothing`. `give` works from inside nested loops:

```antares
to find_pair(limit, target) {
    repeat limit as a {
        repeat limit as b {
            when a * b == target { give [a, b] }
        }
    }
    nothing
}
show find_pair(10, 42)          # [6, 7]
```

## Calling functions

Functions are called with parentheses. Calling with the wrong number of inputs is a problem that names the function:

```output
Problem on line 2: 'f' takes 1 input but was given 2
     2 | f(1, 2)
```

## Order does not matter

Functions written with `to` can be used anywhere in the file, even above the place where they are written. This lets a program put its main logic first and its helpers below.

## Scope and change

Inside a function, `x = ...` always makes a **local** variable, even if a global of the same name exists. This prevents functions from changing the program's state by accident. To replace a variable that lives outside the function, say so with `change`:

```antares
count = 0
to bump() { change count += 1 }
bump()
bump()
show count                      # 2
```

Reading an outer variable needs no keyword; only replacing it does.

## Functions are values

Functions can be stored in variables, passed to other functions and returned from them. A function remembers the variables around it when it was made (a closure):

```antares
to adder(n) = x => x + n
add5 = adder(5)
show add5(10)                   # 15

to compose(f, g) = x => g(f(x))
inc_then_double = compose(x => x + 1, x => x * 2)
show inc_then_double(4)         # 10
```

Built-in functions such as `map`, `filter` and `reduce` accept any function value:

```antares
show map([1, 2, 3], x => x * x)                  # [1, 4, 9]
show filter(range(10), x => x % 2 == 0)          # [0, 2, 4, 6, 8]
show reduce([1, 2, 3, 4], (acc, x) => acc + x, 0)  # 10
```

## Recursion

Functions may call themselves and each other. Deep recursion (thousands of levels) is fine. Endless recursion is caught and reported as an ordinary, catchable problem instead of crashing the program.

```antares
to fib(n) { when n < 2 { n } else { fib(n - 1) + fib(n - 2) } }
to is_even(n) { when n == 0 { yes } else { is_odd(n - 1) } }
to is_odd(n) { when n == 0 { no } else { is_even(n - 1) } }
```

# Errors and Problem Handling {#errors}

## How problems are reported

When something goes wrong, Antares stops and reports the problem with the line number, the line itself, and a plain explanation. A misspelled name comes with a suggestion:

```antares
total = 5
show totl
```

```output
Problem on line 2: I don't know what 'totl' is. Did you mean 'total'?
     2 | show totl
```

A problem inside a file brought in with `use` names that file and its line.

## try and else

If anything inside `try` fails, the `else` block runs instead. The message is placed in the name written after `else` (the name is optional):

```antares
try {
    answer = ask claude "..."
} else problem {
    show "That failed: {problem}"
    answer = "unknown"
}
```

`try` catches every kind of problem: those raised by your own code, by built-in functions (a bad index, a missing file) and by language-model calls. Problems travel up through any number of function calls to the nearest `try`. A `try` has a value, so it can be the result of a function:

```antares
to safe_div(a, b) {
    try { when b == 0 { fail("no") } else { a / b } } else { 0 }
}
```

`try` blocks may be nested. A problem inside an `else` block goes to the next `try` outside it.

## Raising problems: fail and check

`fail("message")` raises a problem of your own. `check(condition, "message")` raises it only when the condition is `no`, which makes it ideal for tests and for guarding inputs:

```antares
to sqrt_of(x) {
    check(x >= 0, "sqrt_of needs a number that is zero or more, not {x}")
    sqrt(x)
}
```

## Ranges: within

A top-level variable can be given the range its numbers must stay in, by adding `within lowest to highest` where it is first given a value. Every value it is given from then on, anywhere in the program (assignments, `change`, and changing one of its numbers), is checked. For a row, list or tensor, every number in it must be in range:

```antares
board = ints(64) within 0 to 15     # every number on the board is 0 to 15
moves = int(0) within 0 to 255
board[10] = 7                       # fine
board[11] = 16                      # problem: board only holds numbers from 0 to 15, not 16
```

A function's inputs can have ranges too. They are checked when the function is called, before its body runs:

```antares
to put(piece within 0 to 15, square within 0 to 63) {
    board[square] = piece
}
put(3, 64)                          # problem: square only holds numbers from 0 to 63, not 64
```

The lowest and highest numbers are numbers, names of numbers given before, or arithmetic on them (`within 0 to SIZE * SIZE - 1`). Fractions are allowed (`within -1 to 9.5`). A variable with a range can't be the counter of `each` or `repeat`.

The built-in `within(x, lowest, highest)` does the same check once: it gives back `x`, or reports a problem. An optional fourth input names the value in the message.

Ranges catch mistakes where they happen instead of where a bad number is used later. In a compiled program they also make code faster: a number known to be in range needs no check when it is used as a position (see "Writing fast compiled code" in [Performance and the Native Compiler](#perf)).

# Splitting a Program Across Files {#modules}

`use "name"` runs `name.ant` from the same folder as the current file, once:

```antares
use "nn"                  # runs nn.ant from the same folder
```

Everything the other file defines (functions and global variables) becomes available. Using the same file a second time does nothing, so libraries can safely `use` each other. `use` belongs at the top level of a file, not inside a block.

A small neural-network library, `examples/nn.ant`, shows the idea:

```antares
# A tiny neural-network library. Bring it into a program with:  use "nn"

to layer(inputs, outputs) {
    learn w = randn(inputs, outputs) * 0.5
    learn b = zeros(outputs)
    give {w: w, b: b}
}

to forward(l, x) = x @ l.w + l.b

to mse(guess, target) = mean((guess - target) ^ 2)
```

> **Tip.** Every file shares one set of global names. If two files use the same global name for different things, they will interfere with each other, and in compiled programs this also slows the code down. Give library globals distinctive names.

# Tensors and Numerical Computing {#tensors}

## Making tensors

| Function | Result |
|---|---|
| `zeros(2, 3)` | A 2x3 tensor of zeros. Sizes may also be given as one list: `zeros([2, 3])` |
| `ones(4)` | Four ones |
| `rand(50)` | 50 random numbers, evenly spread from 0 to 1. `rand()` gives one number |
| `randn(2, 8)` | Random numbers from a normal distribution (mean 0, spread 1) |
| `range(5)` | `[0, 1, 2, 3, 4]`; also `range(start, end)` and `range(start, end, step)` |
| `onehot([0, 2], 3)` | `[[1, 0, 0], [0, 0, 1]]`: each label as a row with a single 1 |
| `seed(7)` | Makes the random numbers repeatable from this point on |

## Shape

`shape(x)` gives the size of each dimension. `reshape(x, 2, 3)` rearranges the same numbers into a new shape. `t(m)` transposes a matrix. `len(x)` is the size of the first dimension.

```antares
show reshape(range(6), 2, 3)
show shape(zeros(2, 3)), t([[1, 2], [3, 4]])
```

```output
[[0, 1, 2],
 [3, 4, 5]]
[2, 3] [[1, 3],
 [2, 4]]
```

## Reductions

`sum(x)` adds every number; `mean(x)` averages them. With a second input they work along one dimension: `sum(m, 0)` adds down the rows (one result per column) and `sum(m, 1)` adds across the columns (one result per row). Negative dimensions count from the end.

```antares
m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
show sum(m, 0), sum(m, 1), mean(m, -1)
```

```output
[12, 15, 18] [6, 15, 24] [2, 5, 8]
```

`max` and `min` give the largest or smallest number among all their inputs. `argmax(x)` gives the position of the largest value along the last dimension, one result per row. `argmax(row, first, last)` gives the position of the first largest number of a row from `first` up to (not including) `last`, or -1 if the range is empty; on a fixed table in a compiled program it reads the table directly.

## Element-by-element maths

`exp log sqrt abs sin cos tanh sigmoid relu floor round trunc` apply to every number of a tensor. `clip(x, low, high)` forces every number into a range. `softmax(x)` turns each row (the last dimension) into probabilities that add up to 1.

```antares
show softmax([1, 2, 3])
show clip([-1, 0.5, 2], 0, 1)
```

```output
[0.090031, 0.244728, 0.665241]
[0, 0.5, 1]
```

## Speed of tensor maths

Tensor operations run as compiled native code, not through the interpreter, which is why tensor-heavy programs are fast even when interpreted. Multiplying 300x300 matrices three times takes about 0.03 seconds in Antares, against 14.7 seconds for plain Python lists (Chapter 16).

# Learning: Automatic Differentiation {#learning}

## learn and descend

`learn` declares a value the program is allowed to adjust. `descend` adjusts every `learn` variable that a number depends on, to make that number smaller. That is the whole training interface.

```antares
learn w = randn(2, 8) * 0.5
...
descend loss by 0.05
```

Behind the scenes, Antares records how every number was computed from learnable values (reverse-mode automatic differentiation). `descend` works backwards through that record to find the gradient of the loss with respect to each learnable value, then takes a step in the direction that reduces the loss.

## Optimisers

```antares
descend loss by 0.05                 # plain gradient descent
descend loss by 0.05 using sgd       # the same, said explicitly
descend loss by 0.05 using momentum  # gradient descent with momentum (0.9)
descend loss by 0.05 using adam      # the Adam optimiser
descend loss by 0.05 using adamw     # Adam with weight decay (0.01)
descend loss by 0.05 using adam clip 1
```

The number after `by` is the learning rate. Adam keeps a running memory of past gradients for each learnable value and usually trains neural networks much faster. AdamW also pulls every learnable value slightly towards zero at each step, which helps large networks generalise. Momentum keeps a running sum of past gradients, so steps build up speed in a steady direction.

`clip <limit>` works with every method. Before the step, the gradients' overall size (the square root of the sum of the squares of every gradient of every learnable value) is measured; when it is above the limit, all gradients are scaled down so that it equals the limit. This keeps training steady when an occasional batch would otherwise make a huge step.

## Gradients without changing anything

`grad(loss, w)` returns the gradient of `loss` with respect to the learnable value `w`, with the same shape as `w`, and changes nothing:

```antares
learn w = [1, 2, 3]
show grad(sum(w ^ 2), w)             # [2, 4, 6]
```

## What gradients flow through

Gradients flow through `+ - * / ^ @`, indexing and slices, the functions `exp log tanh sigmoid relu sqrt abs sin cos gelu silu softplus leaky_relu`, `sum mean softmax log_softmax t reshape` (including `sum` and `mean` along a dimension), `take concat layer_norm dropout`, and the losses `cross_entropy mse bce`. `floor`, `round`, `trunc` and comparisons produce plain values that do not carry a gradient.

## Example: learning a straight line

```antares
seed(1)
x = rand(50) * 4
y = 3 * x + 2 + randn(50) * 0.1

learn slope = 0
learn offset = 0

repeat 400 as step {
    guess = slope * x + offset
    loss = mean((guess - y) ^ 2)
    descend loss by 0.05
    when step % 100 == 0 { show "step {step}: loss {loss}" }
}

show "learned: y = {slope}x + {offset}"
```

```output
step 100: loss 0.011665
step 200: loss 0.010596
step 300: loss 0.010588
step 400: loss 0.010588
learned: y = 3.017847x + 1.961574
```

## Example: a neural network for XOR

```antares
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
    descend loss by 0.05 using adam
}

show net(x) -> round
```

```output
[[0],
 [1],
 [1],
 [0]]
```

## Classification

`cross_entropy(scores, answers)` is the usual loss for choosing between classes. `scores` has one row per example and one column per class, and holds raw scores: softmax is part of the loss, computed in a way that can't overflow. `answers` are the right class numbers, one per row, or the right probabilities, shaped like the scores:

```antares
learn cw = zeros(2, 2)
repeat 200 {
    descend cross_entropy(pts @ cw, labels) by 0.5
}
show accuracy(pts @ cw, labels)
```

The same loss can be written from the formula, `-sum(onehot(labels, 2) * log_softmax(scores)) / rows`, which gives the same result.

## Deep learning building blocks

| Function | What it does |
|---|---|
| `gelu(x)`, `silu(x)`, `softplus(x)`, `leaky_relu(x)` | Activations. GELU (the tanh form used by GPT models), SiLU (x times sigmoid(x)), log(1 + e^x), and relu that lets 1% through below zero |
| `log_softmax(x)` | The log of softmax along the last dimension, without overflow |
| `cross_entropy(scores, answers)` | The classification loss above |
| `mse(guess, target)` | Mean squared error |
| `bce(probability, target)` | Binary cross-entropy, for yes/no predictions made with `sigmoid` |
| `take(table, positions)` | The rows of `table` at those positions, in order: an embedding lookup. Gradients reach exactly the rows taken. A single position gives a single row |
| `concat(tensors [, dimension])` | Joins a list of tensors along a dimension (0 unless given); the other dimensions must match |
| `layer_norm(x [, eps])` | Each row scaled to mean 0 and spread 1 along the last dimension. Multiply by a learned gain and add a learned bias afterwards |
| `dropout(x, p)` | Sets each value to 0 with probability `p` and scales the rest by 1 / (1 - p); `dropout(x, 0)` is `x` |
| `causal_mask(n)` | An n by n table: 0 on and below the diagonal, -1000000000 above. Added to attention scores, it stops a position from seeing later ones |
| `he(inputs, outputs)`, `xavier(inputs, outputs)` | Random starting weights scaled for relu-like and for tanh-like activations |
| `accuracy(scores, answers)` | The fraction of rows whose highest score is the right class |
| `top(values, k)` | The positions of the k largest values, largest first |
| `sample(probabilities [, temperature])` | A random position, chosen with those probabilities. Below 1, the temperature favours likely choices; above 1, it spreads them out |

`seed` controls `dropout`, `sample`, `shuffle`, `he` and `xavier` as well as `rand` and `randn`.

## Attention and transformers

With these, a transformer block is written directly. Self-attention, where every position of a sequence looks back at the ones before it:

```antares
to attention(h) {
    n = shape(h)[0]
    q = h @ wq
    k = h @ wk
    v = h @ wv
    weights = softmax(q @ t(k) / sqrt(width) + causal_mask(n))
    weights @ v @ wo
}

to block(h) {
    h = h + attention(layer_norm(h) * g1)
    h + gelu(layer_norm(h) * g2 @ w_up + b_up) @ w_down + b_down
}
```

`examples/tiny_gpt.ant` builds a complete character-level GPT this way: token and position embeddings with `take`, one block, and an output layer, trained with `cross_entropy` and `descend ... using adamw clip 1`. It learns a page of text in a few seconds and then writes new text with `sample`.

## Training on the GPU

`use gpu` sends tensor maths to the graphics card from that point in a program, and `use cpu` sends it back. `gpu()` gives the name of the graphics card Antares uses, or `nothing`.

<!-- norun -->
```antares
use gpu
learn w1 = he(784, 1024)
learn w2 = xavier(1024, 10)
repeat 10 {
    descend cross_entropy(gelu(x @ w1) @ w2, labels) by 0.001 using adamw clip 1
}
```

With the GPU on:

- Matrix products of at least about 256 x 256 x 256 multiply-adds run on the GPU, and leave their result there.
- Element-wise arithmetic (`+ - * / ^ %` and minus) and every activation run on the GPU when an input is already there and there are at least 4096 numbers, when the inputs have the same shape, or one is a single number, or one is a row repeated for every row (a bias). `reshape` keeps numbers where they are.
- The backward pass works on the GPU for all of these, and for `sum` and `mean` of numbers there; the gradients of a repeated row or number are added up on the GPU.
- `descend` runs SGD, momentum, Adam and AdamW on the GPU for gradients that are there, keeping the optimiser's memory and the new values there; `clip` measures the gradients' size on the GPU.
- Anything else (softmax, `cross_entropy`, indexing, `show`, saving...) brings the numbers back when it reads them, and works as usual.

The GPU works in 32-bit numbers (about 7 significant digits) where the CPU uses 64-bit ones, so results agree with the CPU's to about 6 digits. Programs without `use gpu` are unaffected and give exactly the results they always did. `use gpu` and `use cpu` belong at the top level of a file; `use_gpu(yes)` and `use_gpu(no)` do the same anywhere.

On Windows, Antares uses Direct3D 11 compute shaders, which every PC with a graphics card or built-in graphics has. On Linux, it uses NVIDIA GPUs through CUDA, as on Google Colab and Kaggle: it opens the driver's library (libcuda) and CUDA's run-time compiler (libnvrtc, part of the CUDA toolkit) when first needed, looking in the usual CUDA folders. Either way it picks the GPU with the most memory of its own. The first time a program needs one of its GPU programs, it compiles it, which takes a moment, and keeps it (in the local app data folder on Windows, in `~/.cache/antares` on Linux), so later programs don't wait. Without a usable GPU, and on macOS for now, `use gpu` notes this once on the error output and everything stays on the CPU. The environment variable `ANTARES_GPU=warp` uses Windows' software GPU (for testing on machines without a graphics card), and `ANTARES_GPU=none` behaves as if there were no GPU.

Training a 784-1024-1024-10 classifier for 10 steps takes 0.55 s with `use gpu` on Google Colab's Tesla T4, against 29.0 s on Colab's CPU, and 1.89 s on a laptop's Intel HD Graphics 520, against 29.5 s on its 4-core CPU. A 1024 x 1024 matrix product takes 7.9 ms on the T4 and 49 ms on the HD 520.

On Colab or Kaggle, choose a GPU runtime and install Antares in a cell:

```shell
!curl -fsSL https://raw.githubusercontent.com/Madushan996/Antares-Programming-Language/main/install.sh | sh
```

Then save programs with `%%writefile program.ant` at the top of a cell, and run them with `!~/.antares/bin/antares program.ant`.

## Data

| Function | What it does |
|---|---|
| `read_csv(file)` | A CSV file as a list of maps, named by the first line. Numbers become numbers, empty fields `nothing`, and quoted fields may hold commas. Embedded files (`"embedded:<name>"`) work too |
| `columns(rows, names)` | Numbers from a list of maps as a table, one row per map and one column per name. With a single name (not a list), one row of numbers |
| `shuffle(x)` | The items of a list, or the rows of a tensor, in random order |
| `batches(x, size)` | A list of pieces of a list or tensor, each `size` items or rows long (the last may be shorter) |

Shuffling positions, rather than the data itself, keeps inputs and answers together:

```antares
rows = read_csv("houses.csv")
x = columns(rows, ["size", "rooms"])
y = columns(rows, "price")
repeat 50 {
    each batch in batches(shuffle(range(len(y))), 32) {
        descend mse(take(x, batch) @ w + b, take(y, batch)) by 0.01 using adam
    }
}
```

## Speed

Large matrix multiplications, and their gradients, are shared across all the CPU's cores. Each number is still added up by one core, in the same order as on one core, so results are exactly the same however many cores a computer has.

## Learnable values inside functions and maps

`learn` can be used inside a function, and learnable values can be stored in maps and lists. `descend` still finds them through the computation. This is how the `nn` library's `layer` function works (Chapter 9):

```antares
use "nn"
net = {hidden: layer(2, 8), out: layer(8, 1)}
to predict(net, input) = forward(net.hidden, input) -> tanh -> (h => forward(net.out, h)) -> sigmoid
repeat 1000 {
    descend mse(predict(net, x), y) by 0.05 using adam
}
```

## Saving and loading what was learned

`save(value, "file.json")` writes any value (tensors, maps, lists, text, numbers, yes/no) to a JSON file. `load("file.json")` reads it back exactly. A loaded tensor can be used directly or declared learnable again to carry on training:

```antares
save({w1: w1, b1: b1}, "model.json")

saved = load("model.json")
learn w1 = saved.w1                  # carry on training, or just use it
```

`exists("model.json")` says whether a file is there. Combined with `try`, a program can train on its first run and reuse the trained model afterwards (see `train_save.ant` in Appendix B).

# Language Models {#llm}

## Declaring a model

```antares
model claude = "anthropic:claude-opus-5-5" with {system: "Be brief."}
```

The text names the provider and model, separated by a colon. A name that starts with `claude` may leave out the `anthropic:` prefix.

| Provider | Example | Needs |
|---|---|---|
| `anthropic` | `"anthropic:claude-opus-5-5"` | `ANTHROPIC_API_KEY` |
| `openai` | `"openai:gpt-5"` | `OPENAI_API_KEY` |
| `openai` with `url` | `"openai:llama-3.3"` with `{url: "http://localhost:1234/v1"}` | Any service with OpenAI's interface; its key in `OPENAI_API_KEY` if it needs one |
| `ollama` | `"ollama:llama3.2"` | Ollama running on this computer, and the model fetched with `ollama pull llama3.2` |
| `echo` | `"echo"` | Nothing: a pretend model for working offline |

The optional `with` map sets options:

| Option | Kind | Meaning | Default |
|---|---|---|---|
| `system` | text | A system prompt that guides every answer | none |
| `max_tokens` | number | The longest answer allowed, in tokens | 16000 (64000 when streaming) for Anthropic; the service's own for others |
| `effort` | text | How hard Claude thinks: `"low"`, `"medium"`, `"high"`, `"xhigh"` or `"max"` (Anthropic only) | the model's own |
| `stream` | yes/no | Show the reply as it arrives | `no` |
| `tools` | list | Functions the model may call (below) | none |
| `url` | text | The service's address, for proxies and OpenAI-style services | the provider's own |

## Asking

`ask model prompt` sends the prompt and gives the reply as text. Text interpolation makes it easy to build prompts:

```antares
answer = ask claude "Explain {topic} in one sentence."
```

## Answers as values

`as` turns the reply into a real value you can calculate with:

```antares
count  = ask claude "How many moons does Mars have?" as number
sure   = ask claude "Is 91 a prime number?" as yes/no
ideas  = ask claude "Give me five names for a cat." as list
person = ask claude "Describe Ada Lovelace." as {name: text, born: number, known_for: list of text}
```

| Form | Result |
|---|---|
| `as text` | The reply as text (the same as leaving `as` out) |
| `as number` | The number in the reply |
| `as yes/no` | `yes` or `no` |
| `as list` | A list of text, one item per line of the reply |
| `as list of <shape>` | A list whose items have that shape: `list of number`, `list of {name: text}` |
| `as {field: <shape>, ...}` | A map with exactly those fields; shapes can nest |

Shapes (`list of ...` and maps) are sent to the model as a JSON schema and use its structured output, so the answer always fits. If a reply can't be read as the requested kind, that is a problem that `try` can catch.

## Conversations

`conversation(model)` makes a model that remembers: every `ask` sends what was said before, and adds the new question and answer.

```antares
chat = conversation(claude)
ask chat "My name is Ada."
show ask chat "What is my name?"
show history(chat)          # [{role: "user", text: "My name is Ada."}, ...]
```

`history(chat)` is the conversation so far, as a list of `{role, text}` maps (roles are `user`, `assistant` and `tool`). `conversation(chat)` starts a new conversation from a copy of an existing one, so the two can go different ways.

## Tools: models that call your functions

A model given `tools` can call those Antares functions while it answers. Antares runs each call with the inputs the model chose, gives the result back to the model, and repeats until the model gives its final answer:

<!-- norun -->
```antares
to weather(city) = "sunny and 31 degrees in {city}"
to percent(amount, rate) = amount * rate / 100

model helper = "anthropic:claude-opus-5-5" with {tools: [weather, percent]}
show ask helper "Is it hot in Kandy today? And what is 17% of 2450?"
```

The model sees each function's name and the names of its inputs. A map describes a tool more fully:

```antares
{run: weather, name: "today_weather", about: "Today's weather for a city", inputs: {city: "text"}}
```

`run` is the function; `name` (letters, digits, `_` and `-`), `about`, `inputs` and `confirm` are optional. Inputs are `"number"`, `"text"`, `"yes/no"` or `"list"`, in the function's order. A function's result goes back to the model as text (maps and lists as JSON). A problem in a tool doesn't stop the program: it is reported to the model, which can try something else. A model may call several tools at once and up to 25 rounds of tools per `ask`. Tools work in conversations, where the calls and results are remembered too.

## Built-in tools, and asking first

Some built-in functions can be tools by themselves, so an agent that runs commands, reads the web or works with files needs no functions of your own:

| Tool | What the model can do | Asks first |
|---|---|---|
| `shell` | Run a command and see its output, errors and exit code (Chapter 14) | yes |
| `write` | Write a text file, replacing what was in it | yes |
| `read` | Read a text file | no |
| `files` | List a folder | no |
| `read_page` | Read a web page as text (Chapter 14) | no |
| `fetch` | Fetch a web address exactly as it is sent | no |

<!-- norun -->
```antares
model coder = "ollama:qwen2.5-coder" with {tools: [shell, read, write, files]}
show ask coder "Which file in this folder is the biggest?"
```

Before Antares runs a tool that asks first, it shows what the model wants to do and waits for an answer:

```output
The model wants to use the tool 'shell':
    command: dir /s
Allow it? (y/n)
```

Typing `y` runs it; anything else tells the model that the user did not allow it, and the model carries on without. When there is no one to answer (the program's input is not a keyboard, or has ended), the answer is no. `confirm: yes` in a tool's map asks first for your own functions too, and `confirm: no` turns asking off: `{run: shell, confirm: no}`. Turn it off only where you would be happy for the model to run any command unseen, such as a throwaway virtual machine. `examples/coder.ant` is a coding helper built this way, and `examples/researcher.ant` answers questions from Wikipedia.

## Images

`image(file)` reads a .png, .jpg, .gif or .webp file for a prompt, after `with`:

<!-- norun -->
```antares
show ask claude "What is in this picture?" with image("photo.jpg")
show ask claude "Which is bigger?" with [image("a.png"), image("b.png")]
```

A prompt can also be a list of text and images: `ask claude ["Compare", image("a.png"), image("b.png")]` when the list is held in a variable.

## Streaming

With `stream: yes`, the reply is shown as it arrives and still given back as the value of `ask`. It works with every provider, with tools and in conversations. Structured answers (`as` with a shape) stream too, but aren't shown, since they are JSON.

## Embeddings

`embed(model, text)` gives a row of numbers that stands for the meaning of a text; with a list of texts, one row per text. `similarity(a, b)` is the cosine similarity of two rows (from -1 to 1), or with a table as `b`, one score per row:

<!-- norun -->
```antares
model embedder = "openai:text-embedding-3-small"
notes = ["The museum is closed on Mondays.", "Bus 138 goes to Colombo every half hour."]
vectors = embed(embedder, notes)
scores = similarity(embed(embedder, "When can I visit the museum?"), vectors)
show notes[top(scores, 1)[0]]
```

Anthropic doesn't make embeddings: use OpenAI, an OpenAI-style service, Ollama (for example `"ollama:nomic-embed-text"`), or `"echo"`, whose embeddings count letter triples so that texts sharing words come out similar. `examples/search.ant` answers questions from a set of notes this way.

## Setting up access

Set the API key for the provider (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`) before running the program; Ollama needs none. Requests are sent with the `curl` program that comes with Windows, macOS and Linux, so Antares needs no networking library and stays small. Keys are passed to curl through its standard input, so they never appear in the list of running programs.

## Working offline: the echo model

`model fake = "echo"` makes a pretend model that repeats the prompt back. Programs can be written and tested without a network connection or API key, then switched to a real model by changing one line. Asked for a shape, it makes up an answer of that shape (0, empty lists, and the prompt as text):

```antares
model claude = "echo"

topic = "red giant stars"
show ask claude "Explain {topic} in one sentence."

legs = ask claude "How many legs does a spider have? (8)" as number
show legs + 1

ideas = ask claude "apple\nbanana\ncherry" as list
each idea in ideas { show " *", idea }
```

```output
Explain red giant stars in one sentence.
9
 * apple
 * banana
 * cherry
```

## Handling failures

Network errors, a missing key, a refusal and unreadable answers are ordinary problems, so wrap calls that must not stop the program in `try`:

```antares
try {
    answer = ask claude "Summarise: {report}"
} else problem {
    show "The model call failed: {problem}"
    answer = ""
}
```

# Integers and Bit Manipulation {#integers}

Ordinary numbers are decimals. For bit manipulation, such as board games, hashing and compression, Antares has 64-bit integers.

## Making integers

```antares
mask = 0xFF00                 # a 0x literal is an integer
n = int(12)                   # so is int(x)
row = ints(64)                # a row of 64 integers, all 0
masks = ints([1, 2, 4, 8])    # a row from a list
r = rand_bits()               # 64 random bits
```

## Operators

```antares
n & 10      n | 3      n xor 5      ~n        # and, or, exclusive or, flip
1 << 4      n >> 2                            # shifts
```

| Expression (n = 12) | Result |
|---|---|
| `n & 10` | 8 |
| `n \| 3` | 15 |
| `n xor 5` | 9 |
| `~n` | -13 |
| `1 << 4` | 16 |
| `div(n, 5)` | 2 |

## Rules

- An integer combined with a whole number stays an integer: `n + 1` is an integer.
- `/` always gives an ordinary number: `int(12) / 8` is `1.5`. Use `div(a, b)` for division that drops the remainder.
- Combining an integer with a fraction is refused, to prevent silent loss of precision. Convert first with `number(n)`.
- Arithmetic wraps around at 64 bits. `>>` brings in zeros from the left. Shifting by 64 or more gives 0.
- Integers can be used as positions in lists and tensors.

## Counting bits

| Function | Result |
|---|---|
| `popcount(x)` | How many of the 64 bits are set |
| `lowest_bit(x)` | Position of the lowest set bit (64 if none) |
| `highest_bit(x)` | Position of the highest set bit (-1 if none) |

A common pattern walks the set bits of a number:

```antares
bits = 0x8421
found = []
while bits != 0 {
    found = push(found, number(lowest_bit(bits)))
    bits = bits & (bits - 1)          # clear the lowest set bit
}
show found                            # [0, 5, 10, 15]
```

In compiled programs, integers live in machine registers, so this kind of code runs at native speed.

# Input, Output and Data {#io}

## Printing and reading

`show` prints values (Section 6.2). `input("prompt")` prints an optional prompt and reads one line typed by the user, without the line ending; at the end of input it gives `nothing`. `input_waiting()` says whether input is waiting to be read, without pausing.

```antares
name = input("What is your name? ")
show "Hello, {name}!"
```

## Files

| Function | What it does |
|---|---|
| `read(file)` | The whole file as text |
| `write(file, value)` | Writes a value (as text) to a file, replacing it |
| `exists(file)` | Whether the file exists |
| `save(value, file)` | Writes any value as JSON |
| `load(file)` | Reads a value written by `save` |
| `read_binary(file, kind, start, count)` | Reads `count` whole numbers from a binary file, starting at byte `start`. `kind` is `"u8"`, `"i8"`, `"u16"`, `"i16"`, `"u32"` or `"i32"` (little-endian) |
| `program_folder()` | The folder the running program is in, so it can find files next to it wherever it is started from |
| `append(file, value)` | Adds a value (as text) to the end of a file, creating it if needed |
| `delete(file)` | Removes a file |
| `files([folder])` | The names of the files and folders in a folder (the current one if left out), sorted |

## JSON

`json(text)` parses JSON into Antares values (objects become maps, arrays become lists or tensors, `true`/`false` become `yes`/`no`, `null` becomes `nothing`). `to_json(value)` does the reverse.

```antares
data = json("\{\"n\": [1, 2, 3], \"ok\": true}")
show sum(data.n), data.ok             # 6 yes
```

## Text functions

`upper`, `lower` and `trim` change text. `split(text)` splits on spaces (or `split(text, sep)` on a separator), `lines(text)` splits into lines, `replace(text, from, to)` replaces every occurrence, and `join(list, sep)` puts items back together. `find(text, part)` gives the position where `part` first appears, or -1. `text(v)` turns any value into text, and `number(t)` reads a number from text. `code(t)` gives the number of a character and `char(n)` the character with a number.

Two functions lay numbers out. `fixed(x, places)` gives `x` as text with exactly that many digits after the point, and `pad(value, width)` adds spaces to make a value `width` characters wide. Numbers are lined up on the right and text on the left, as in a table; a third input, `"left"` or `"right"`, chooses, and a fourth fills with another character instead of spaces: `pad(5, 2, "right", "0")` is `"05"` (a minus sign stays in front of the zeros). What `fixed` gives is text, so line it up on the right yourself:

```antares
show fixed(3.14159, 2), fixed(2, 3)
each row in [["tea", 2.5], ["cake", 12]] {
    show pad(row[0], 6) + pad(fixed(row[1], 2), 7, "right")
}
```

```output
3.14 2.000
tea      2.50
cake    12.00
```

```antares
show split("a,b,c", ","), join(["x", "y"], "-"), replace("cat", "c", "b")
```

```output
["a", "b", "c"] x-y bat
```

## Time

`now()` gives the local date and time as a map with the fields `year`, `month`, `day`, `hour`, `minute`, `second` and `weekday` (the day's name, such as `"Monday"`):

```antares
t = now()
show "Today is {t.weekday} {t.day}/{t.month}/{t.year}"
show "The time is {t.hour}:{pad(t.minute, 2, "right", "0")}"     # 9:05, not 9:5
```

`clock()` gives the current time in seconds, as a decimal, useful for measuring how long something takes:

```antares
start = clock()
result = fib(25)
show "took", clock() - start, "seconds"
```

## The command line and settings

`args()` gives what was typed after the program's name when it was started, as a list of text. `env(name)` gives a setting of the computer (an environment variable), or `nothing` if it isn't set:

```antares
# antares greet.ant Maya
when len(args()) > 0 {
    show "Hello, {args()[0]}!"
} else {
    show "Hello, whoever you are!"
}
show env("HOME")
```

Compiled programs get their arguments the same way: `greet.exe Maya`.

## Running commands

`shell(command)` runs a command, as if typed in a terminal, and gives back a map: `output` (what it printed), `errors` (what it printed as errors) and `code` (its exit code: 0 for success). Commands run in `cmd` on Windows and `sh` elsewhere:

```antares
r = shell("echo hello")
show r.output, r.code
show shell("exit 4").code
```

```output
hello 0
4
```

A command that fails is not a problem for the program: look at `code`. The ends of lines at the very end of `output` and `errors` are removed. A map of options can follow the command:

| Option | Meaning | Default |
|---|---|---|
| `folder` | The folder to run the command in | the current folder |
| `input` | Text sent to the command, as if typed | nothing (the command gets no input) |
| `timeout` | Seconds before a command still running is stopped, together with any programs it started, such as a game window (its `code` is then -1) | 120 |

<!-- norun -->
```antares
show shell("git log --oneline -3", {folder: "my_project"}).output
show shell("sort", {input: "pear\napple\n"}).output
```

> **Watch out.** `shell` runs whatever text it is given, with your permissions. Never build a command from text you don't trust, such as a web page or a model's reply, unless someone looks at it first: tools that run commands ask first for this reason.

## The web

`fetch(url)` gives what a web address sends back, as text: a web page's HTML, or the JSON of a web service, which `json` turns into values. Requests go through the `curl` program that comes with Windows, macOS and Linux. Redirects are followed, and an answer of 400 or more (such as 404, not found) or no answer at all is a problem, which `try` can catch. A map of options can follow the address:

| Option | Meaning | Default |
|---|---|---|
| `query` | A map of values added to the address after `?`, made safe for addresses | none |
| `method` | `"GET"`, `"POST"`, `"PUT"`, `"DELETE"` and so on | `"GET"`, or `"POST"` with a `body` |
| `headers` | A map of extra headers, such as `{Authorization: "Bearer ..."}` | none |
| `body` | Text to send; a map or list is sent as JSON | none |
| `timeout` | Seconds to wait for the whole answer | 60 |

<!-- norun -->
```antares
found = json(fetch("https://en.wikipedia.org/w/api.php", {query: {action: "query", list: "search", srsearch: "Sigiriya", format: "json"}}))
show map(found.query.search, r => r.title)

reply = json(fetch("https://example.com/api/notes", {body: {title: "Shopping", text: "eggs, rice"}}))
```

Three functions read web pages. `page_text(html)` gives a page's readable text: no markup, scripts or styles, no menus, side bars or footers, only the title and the main part when the page marks one, and a line for each heading, paragraph and list item. `read_page(url)` fetches a page and gives its readable text in one step, the easiest way for a program or a model to read the web. `page_links(html, address)` gives a page's links as `{text, url}` maps, made into full addresses when the page's own address is given:

```antares
html = "<h1>Moons</h1><p>Earth has <a href='/moon'>one</a>.</p><script>count()</script>"
show page_text(html)
show page_links(html, "https://example.com/space/")
```

```output
Moons

Earth has one.
[{text: "one", url: "https://example.com/moon"}]
```

# Games and Live Terminal Programs {#games}

Four built-ins turn the terminal into a live screen, which is enough for simple games, dashboards and animations:

| Function | What it does |
|---|---|
| `key()` | The key being pressed, as text (`"up"`, `"down"`, `"left"`, `"right"`, or the character), or `nothing`. Never pauses |
| `wait(seconds)` | Pauses the program |
| `draw(text)` | Redraws the screen in place from the top-left corner, in one go, so animation does not flicker |
| `clear()` | Clears the screen |

A game loop reads keys, updates its state and redraws. This is the heart of `examples/snake.ant`, a complete Snake game in under 90 lines:

```antares
clear()
while yes {
    pressed = key()
    while pressed != nothing {
        when pressed == "q" { stop }
        heading = turn(moving, pressed)
        pressed = key()
    }
    ...
    draw(picture(snake, food, score))
    wait(speed)
}
```

Run it in a terminal with `antares examples/snake.ant`. Steer with the arrow keys or W A S D, and press Q to quit.

## Games in a window

On Windows, a program can open a window and draw in it. Antares draws every picture itself and the window only shows it, so nothing else needs to be installed. Shapes have smooth edges, colours can be see-through, and there are gradients, pictures from PNG files, real fonts, soft shadows and lighting effects. Small pixel-art games work as well as detailed ones: a 160 by 120 game is scaled up in whole steps so its pixels stay square and sharp, and a large one, like 1280 by 720, is shown as it is (or shrunk smoothly if the screen is smaller). Positions are in the picture's own pixels, from (0, 0) at the top-left, and can have fractions: something at x = 10.5 is drawn half a pixel further right, so slow movement glides instead of jumping a pixel at a time.

| Function | What it does |
|---|---|
| `window(width, height [, title [, scale]])` | Opens a window of `width` by `height` pixels, scaled to fit the screen (or by a whole-number `scale`) |
| `window_open()` | `yes` until the player closes the window |
| `frame()` | Shows what you've drawn, then waits, so the game runs at 60 frames a second. Gives the seconds since the last frame |
| `fullscreen([yes or no])` | Fills the screen with the game, or goes back to a window; `fullscreen()` says which it is |
| `fill([colour])` | Paints the whole picture (black if no colour is given) |
| `rect(x, y, width, height [, colour [, radius]])` | A filled rectangle, its top-left corner at `x`, `y`; `radius` rounds the corners |
| `circle(x, y, radius [, colour])` | A filled circle centred on `x`, `y` |
| `ellipse(x, y, radius_x, radius_y [, colour])` | A filled ellipse |
| `ring(x, y, radius, thickness [, colour])` | The outline of a circle |
| `polygon(points [, colour])` | A filled shape with corners at the points, given as `[[x, y], [x, y], ...]` |
| `line(x1, y1, x2, y2 [, colour [, width]])` | A line, 1 pixel wide unless a width is given, with round ends |
| `path(points [, colour [, width]])` | Lines joining the points; a closed outline when the last point is the first |
| `pixel(x, y [, colour])` | One pixel |
| `label(text, x, y [, colour [, size]])` | Text, its top-left corner at `x`, `y` |
| `font([name [, size [, style]]])` | The font labels use, like `font("Segoe UI", 24, "bold")`; `font()` goes back to the pixel font |
| `measure(text [, size])` | `{width, height}`: the size a label of the text would take |
| `sprite(rows, x, y, colours [, size])` | Pixel art written as rows of text (see below) |
| `picture(file)` | Reads a PNG file as a picture |
| `picture(width, height)` | A new, see-through picture to draw on |
| `draw(picture, x, y [, options])` | Draws a picture, its top-left corner at `x`, `y`: scaled, turned, faded, flipped or in part (see below) |
| `canvas([picture])` | Drawing goes on the picture from now on; `canvas()` goes back to the window |
| `colour_at(x, y)` | The colour of a pixel, as `[red, green, blue]` (with an alpha if it is see-through) |
| `gradient(x1, y1, x2, y2, colours)` | Colours changing smoothly along a line, to use anywhere a colour goes |
| `radial(x, y, radius, colours)` | Colours changing smoothly outwards from a centre |
| `hsl(hue, saturation, lightness [, alpha])` | A colour from its hue (0 to 360 round the colour wheel) and its saturation and lightness (0 to 1) |
| `mix(colour1, colour2, amount)` | A colour between two: `amount` from 0 (the first) to 1 (the second) |
| `blend([mode])` | How what is drawn next combines with what is there: `"normal"`, `"add"`, `"multiply"` or `"screen"` |
| `smooth(yes or no)` | Smooth edges (the default), or hard pixel edges for the classic pixel-art look |
| `shadow([across, down, softness [, colour]])` | What is drawn next casts a soft shadow; `shadow()` turns shadows off |
| `blur(radius [, x, y, width, height])` | Softens the picture, or part of it |
| `glow(amount [, radius [, threshold]])` | Makes the bright parts of the picture glow |
| `vignette(amount)` | Darkens the edges and corners, from 0 (not at all) to 1 |
| `held(key)` | `yes` while a key is held down |
| `pressed(key)` | `yes` if a key went down since the last frame |
| `mouse()` | `{x, y, left, right}`: where the mouse is, in the game's pixels, and which buttons are held |
| `screenshot(file)` | Saves the picture as a PNG file |

A game loop asks `window_open()`, reads the keys, updates the game, draws the whole picture and calls `frame()`:

<!-- norun -->
```antares
window(160, 120, "Dodge")
x = 72
while window_open() {
    when held("left") { x -= 2 }
    when held("right") { x += 2 }
    fill("navy")
    rect(x, 100, 16, 6, "orange")
    label("Hello!", 4, 4, "yellow")
    frame()
}
```

`examples/catch.ant` is a complete pixel-art game built this way, with sound, in under 90 lines, and `examples/night.ant` is a night scene that uses gradients, see-through light, glow, shadows and real fonts.

**The window.** Without a scale, the window is as large as fits comfortably on the screen. A program has one window at a time. Closing it makes `window_open()` give `no`; a program that calls `frame()` after that ends, so `while yes` loops end too. `fullscreen(yes)` fills the screen, with black bars where the shapes of the game and the screen differ; `fullscreen(not fullscreen())` switches between the two. A program built with `antares build` and started by double-clicking hides its own console window, and a problem that stops it is then shown in a message box. With `--icon`, the window shows the program's icon.

**Smooth movement.** `frame()` gives the seconds since the last frame, about 1/60. Moving things by a speed times that time keeps a game running at the same pace when a frame is slow: `x += 120 * dt` moves 120 pixels a second.

**Colours** are names, codes or lists. The names are the usual web colours: `black`, `white`, `gray`, `grey`, `lightgray`, `darkgray`, `silver`, `red`, `maroon`, `orange`, `gold`, `yellow`, `olive`, `lime`, `green`, `teal`, `cyan`, `skyblue`, `blue`, `navy`, `purple`, `magenta`, `pink`, `brown`, and `transparent`. Codes are `"#rrggbb"` or `"#rgb"`, and lists are `[red, green, blue]` with each part from 0 to 255. Without a colour, shapes and text are white, and `fill()` is black.

**See-through colours** have a fourth part, the **alpha**, from 0 (invisible) to 1 (solid): `[255, 0, 0, 0.5]` is half-see-through red, and so are the codes `"#ff000080"` and `"#f008"`. What is under a see-through colour shows through it. `fill` with a see-through colour paints over the picture (a quick way to leave fading trails behind moving things), and `fill("transparent")` empties a picture.

**Gradients** go anywhere a colour does, in shapes, text and `fill`. `gradient(0, 0, 0, 240, ["skyblue", "navy"])` changes from sky blue at the top to navy 240 pixels down; `radial(x, y, r, ["white", "#fff0"])` fades from white at the centre to nothing at radius `r`, which makes lights and glows. A gradient can have any number of colours, spaced evenly, and the positions are in the picture, not in the shape, so one gradient can colour many shapes.

**Edges and positions.** Shapes are drawn with smooth edges: a pixel half covered by a shape gets half its colour. Shapes at whole-number positions cover whole pixels exactly, and lines run through the middle of pixels, so a line at whole numbers is sharp. `smooth(no)` switches to hard pixel edges and whole-number positions (fractions are rounded down), the look of classic pixel-art games; `smooth(yes)` switches back.

**Blending.** `blend("add")` adds what is drawn to what is there, so light builds up where things overlap: fire, sparks, lasers and glows. `blend("multiply")` darkens, like tinted glass or a shadow, and `blend("screen")` lightens more softly. `blend()` goes back to normal painting.

**Shadows.** After `shadow(4, 6, 10)`, everything drawn casts a shadow 4 pixels right and 6 down, softened by 10 pixels, until `shadow()`. The colour is half-see-through black unless one is given. Shadows follow the shape of what casts them, including the see-through parts of pictures and the letters of text.

**Pictures.** `ship = picture("ship.png")` reads a PNG file, with its see-through parts; `ship.width` and `ship.height` give its size. A file is read only once, however often `picture` is called with its name. JPEG and other formats can be saved as PNG with any image editor. `draw(ship, x, y)` draws it, and a map of options changes how:

| Option | What it does |
|---|---|
| `scale` | How many times as big: 2 is twice the size, 0.5 half |
| `width`, `height` | The size to draw it at, in pixels |
| `angle` | Turns it, in degrees clockwise, about its middle |
| `alpha` | How solid it is, from 0 (invisible) to 1 |
| `flip` | `yes` mirrors it left to right; `"vertical"` top to bottom; `"both"` both |
| `part` | `[x, y, width, height]`: only that part of the picture, such as one frame of a sprite sheet |
| `tint` | A colour it is multiplied by: `"red"` makes it reddish, `[255, 255, 255, 0.5]` half see-through |
| `smooth` | `no` keeps the pixels sharp when it is scaled up, for pixel art |

<!-- norun -->
```antares
ship = picture("ship.png")
draw(ship, 100, 80, {angle: 30, scale: 2})
draw(explosion, 40, 40, {part: [64 * step, 0, 64, 64]})   # frame `step` of a strip
```

**Drawing on pictures.** `picture(width, height)` makes a new, see-through picture, and `canvas(p)` sends drawing to it until `canvas()`. Parts of a scene that don't change can be drawn once on a picture and the picture drawn every frame, which is much quicker. Pictures work on every system, even without a window, so a program on Linux or macOS can draw pictures and save them with `screenshot(file)`, which saves the picture being drawn on.

**Text.** Labels are in the built-in pixel font until `font` chooses another. In the pixel font, each character is 5 by 7 pixels in a 6 by 8 cell, times `size`, a whole number; characters outside plain English letters, digits and punctuation show as `?`. With a font such as `font("Segoe UI", 20)`, `"Arial"`, `"Georgia"` or `"Consolas"`, text is smooth, any character works, and `size` is the height of the letters in pixels. The style can be `"bold"`, `"italic"` or `"bold italic"`. `measure(text)` gives the size a label takes, to centre it: `label(t, (W - measure(t).width) / 2, 10)`. Fonts come from Windows; elsewhere, labels in pictures use the pixel font.

**Effects** change the whole picture, so they come after everything else in the frame, just before `frame()`. `glow(1)` makes the bright parts shine into their surroundings, as light does in a camera; the radius sets how far, and the threshold (0 to 1) how bright something must be to glow. `vignette(0.5)` darkens the corners, which draws the eye to the middle. `blur(8)` softens everything, for a paused game behind a menu; `blur(8, x, y, w, h)` softens only a rectangle, like frosted glass.

**Speed.** Shapes, text, pictures and gradients are quick, and thousands can be drawn each frame in a small game. Effects work on every pixel, so they cost more in big windows: at 1280 by 720, `glow` and a large `blur` each take a noticeable part of a frame on a slow laptop. Drawing what doesn't change once, on a picture, and smaller game sizes like 960 by 540 keep big scenes smooth.

**Keys** are named `"left"`, `"right"`, `"up"`, `"down"`, `"space"`, `"enter"`, `"escape"`, `"backspace"`, `"tab"`, `"shift"`, `"ctrl"`, `"alt"`, `"delete"`, `"a"` to `"z"`, `"0"` to `"9"` and `"f1"` to `"f12"`, and `"click"` and `"right click"` are the mouse buttons. `held` is for movement (it stays `yes` while the key is down); `pressed` is for actions like jumping (it is `yes` for one frame per press). A name `held` and `pressed` don't know is a problem, which catches typing mistakes. While a window is open, `key()` gives the keys pressed in it, including a key's repeats when it is held, and `wait()` keeps the window answering.

**Sprites** are pixel art written as text, one character per pixel. The map says which character is which colour, and every other character (`.` by convention) shows what's behind:

<!-- norun -->
```antares
heart = [".r.r.", "rrrrr", ".rrr.", "..r.."]
sprite(heart, 20, 10, {r: "red"}, 3)      # three times as big
```

**Screenshots.** `screenshot(file)` saves the picture at its own size, as a PNG file, and gives back the file name.

Windows only for now: on macOS and Linux, `window()` stops with a problem that says so, and the functions that need a window (`frame`, `held`, `mouse` and so on) ask for one. Pictures, drawing on them and saving them work everywhere.

## 3D

Antares draws 3D worlds too, into the same picture as 2D drawing, so labels and effects go on top. A program describes each frame: where the camera is, the light, and the shapes; `render()` draws them (and `frame()` does if the program doesn't). The graphics card does the drawing, through Direct3D 11, which every Windows PC has; a computer without a graphics card uses Windows' own software renderer, which is slower but looks the same. Nothing needs installing.

<!-- norun -->
```antares
window(960, 540, "3D")
t = 0
while window_open() {
    t += frame()
    camera(0, 3, 8, 0, 0.5, 0)              # where it is, and what it looks at
    sky("#4f86d9", "#d8e6f3")
    sun(0.5, 1, 0.3)
    plane(0, 0, 0, 40, 40, "#6f9a4f")       # the ground
    cube(0, 0.75, 0, 1.5, {colour: "red", turn: t * 45})
    sphere(2.5, 1, 0, 1, {colour: "white", metal: 1, shine: 0.9})
    render()                                # draw it, then 2D on top
    label("Hello, 3D", 20, 20)
}
```

The world's `y` points up, `x` to the right and `z` towards the viewer, so the ground is flat at `y = 0`. Sizes and positions are in any units you like (metres work well); every shape is placed by its middle.

| Function | What it does |
|---|---|
| `camera(x, y, z, look_x, look_y, look_z [, options])` | Puts the camera at a point, looking at another. Options: `fov` (how wide it sees, in degrees, 60 unless given), `near` and `far` (the nearest and furthest it draws, 0.1 and 1000) |
| `cube(x, y, z, size [, look])` | A cube |
| `box(x, y, z, width, height, depth [, look])` | A box |
| `sphere(x, y, z, radius [, look])` | A ball |
| `cylinder(x, y, z, radius, height [, look])` | A cylinder, standing up |
| `cone(x, y, z, radius, height [, look])` | A cone, point up |
| `plane(x, y, z, width, depth [, look])` | A flat rectangle facing up, such as the ground or a floor |
| `mesh(file)` | Reads a model from a Wavefront `.obj` file |
| `mesh(points, triangles [, places])` | A model made from its corners `[[x, y, z], ...]` and triangles `[[a, b, c], ...]` (corner numbers, from 0), with optional places on a picture `[[u, v], ...]` |
| `mesh(model, points)` | The same model with its corners moved: for water, flags and cloth that change shape every frame |
| `draw(model, x, y, z [, look])` | Draws a model, its own origin at `x, y, z` |
| `sun(x, y, z [, colour [, strength]])` | Sunlight coming from that direction; `sun()` turns it off |
| `lamp(x, y, z [, colour [, strength [, reach]]])` | A light at a point, for this frame (up to 8) |
| `sky([top [, horizon [, ground]]])` | The sky behind everything, which also lights the scene softly; `sky()` leaves the background as it is |
| `fog([colour,] near, far)` | Things fade into fog from `near` away to `far` away; `fog()` clears it |
| `shadows(yes or no)` | Whether the sun casts shadows (it does unless told not to) |
| `render()` | Draws the scene described since the last render |
| `to_screen(x, y, z)` | Where a point shows in the picture, as `[x, y]`, or `nothing` if it is behind the camera |

**Looks.** The last input of a shape is its colour, or a map of options:

| Option | What it does |
|---|---|
| `colour` | Its colour. A see-through colour, like `[120, 200, 255, 0.4]`, makes glass |
| `shine` | From 0 (rough, like chalk) to 1 (polished, like a mirror); 0.3 unless given |
| `metal` | From 0 to 1: metal reflects the world in its own colour |
| `glow` | Light it gives off itself, in its colour: 1 glows softly, 3 brightly. `glow()` (the 2D effect) after `render()` makes it shine |
| `ripples` | Fine moving ripples drawn on it by the graphics card, too small for its corners: 1 for water, 0.5 for a calm pond |
| `picture` | A picture wrapped onto it (a texture) |
| `tiles` | How many times the picture repeats across it, for floors and walls |
| `turn` | Degrees to turn it: a number turns it about the up axis; `[x, y, z]` about each axis in turn |
| `scale` | Stretches it: a number, or `[x, y, z]` |
| `shadow` | `no` if it shouldn't cast a shadow |

<!-- norun -->
```antares
cube(0, 1, 0, 2, {colour: "gold", metal: 1, shine: 0.8, turn: [0, 30, 0]})
plane(0, 0, 0, 50, 50, {picture: picture("grass.png"), tiles: 25})
sphere(0, 3, 0, 0.2, {colour: "orange", glow: 3})
```

**Light.** The look aims for realism with little effort. Light is worked out the way real light behaves (rough and polished, plain and metal surfaces reflect it differently), the sun casts soft shadows, the sky lights everything a little from above, and edges are smoothed. Without `sun` or `sky`, there is a gentle default sun and sky light. A lamp's `strength` is 1 unless given, and its light fades out by `reach` (12 units). For a night scene, choose a dark sky, a faint bluish sun for the moon, and lamps.

**Models.** `mesh` reads the triangles of an `.obj` file, the format most 3D programs (such as Blender) can save, with its normals and its places on a picture if it has them. A file is read once however often `mesh` is called with its name; it gives `{mesh, size, triangles}`, where `size` is `[width, height, depth]`. A model made from points gets normals worked out from its faces: smooth where faces meet at gentle angles, sharp at edges. Make models once, before the loop, and `draw` them each frame. A model that changes shape, like the surface of the sea, keeps its triangles and gets new corners each frame with `mesh(model, points)`; the points can be worked out quickly with tensor maths and given as a table of `[x, y, z]` rows.

**Water.** A shiny dark-blue surface with `ripples: 1` looks like water: the ripples break the sun's reflection into a path of glints, and where they are too small to see, far away, they make the surface a little rougher instead, as real distant water looks. Bigger waves come from moving the corners of a mesh, so an ocean combines both: long waves shape the mesh, and ripples add the detail.

**Mixing with 2D.** `render()` draws over the picture as it is: with a sky, the sky fills it; without one, 3D shapes go over whatever was drawn before, such as a 2D background. Anything drawn after `render()`, like a score or `glow()`, goes on top; when `frame()` draws a scene that `render()` didn't, it goes over everything drawn that frame, so a program with 2D on top calls `render()` itself. `to_screen` puts 2D labels next to 3D things.

**Speed.** On a graphics card, scenes of hundreds of shapes take a few milliseconds. With Windows' software renderer, each frame takes several times longer.

3D is Windows-only for now: elsewhere, `render()` stops with a problem that says so. Placing shapes and making models work everywhere.

## Sound

Programs can play sound files, tones, tunes written as text, ready-made effects and looping music, all at the same time, in a window or in the terminal. Antares mixes every sound itself, 44,100 numbers a second, and hands the result to the speakers.

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

<!-- norun -->
```antares
effect("coin")
beep(440, 0.2, "sine")
tune("C E G C5:2", 0.15)
sound("splash.wav", 0.5)                  # at half volume
music("C - E - G - E -", 0.2, "triangle")
```

**Tunes** are notes separated by spaces. A note is a letter from `A` to `G`, then `#` (sharp) or `b` (flat) if needed, then an octave from 0 to 8 (4 if none is given): `C`, `F#`, `Bb3`, `A5`. A `-` is a rest. Each note or rest lasts one step, 0.15 seconds unless another is given, or several steps with `:n`: `G:3` lasts three. Notes are separated by a short gap, so repeated notes stay apart.

**Waves** give a tone its character: `"square"` (the default, the classic sound of old games), `"sine"` (soft and pure), `"triangle"` (between the two), `"saw"` (buzzy) and `"noise"` (hiss, for drums and crashes). They are balanced to sound about equally loud.

**Sound files** are WAV files: 8, 16, 24 or 32-bit whole numbers, or 32-bit floating point, at any sample rate, mono or stereo. Each file is read once and kept, so playing it every frame is cheap. A file packed into a program with `antares build --embed` is played as `sound("embedded:<file name>")`. Other formats, like MP3, can be saved as WAV with a free sound editor.

**Music** loops until it is stopped or replaced: there is one piece of music at a time. `playing()` doesn't count it, so a program can wait for its sound effects with `while playing() > 0 { wait(0.01) }`.

**Timing.** The sound device is opened the first time a sound is played, in the background, which can take up to a second; sounds asked for meanwhile start as soon as it is ready. After that, sounds start at most 60 ms after they are asked for. When a program ends, it waits for the sounds still playing (not the music) to finish.

On macOS and Linux, and on a computer without a sound device, the sound functions do nothing (a missing or broken sound file is still a problem), so games still run there, silently.

# Performance and the Native Compiler {#perf}

## The interpreter

Antares translates each function once into compact instructions and runs them in a tight loop. Every name is resolved to a numbered slot before the program starts, so variables are found without searching. Measured on one Windows laptop, best of 7 runs (`python bench/bench.py` repeats this on your machine):

| Test | Antares | Python 3.12 | Node 24 |
|---|---|---|---|
| `fib(30)`, 1.3 million calls | 0.28 s | 0.35 s | 0.013 s |
| 5 million additions in a loop | 0.17 s | 0.96 s | 0.007 s |
| `while` loop, 3 million rounds | 0.31 s | 0.93 s | 0.012 s |
| 3 million list lookups | 0.55 s | 0.92 s | 0.008 s |
| 300x300 matrix multiply, three times | 0.03 s | 14.7 s (plain lists) | 0.15 s |

Antares is faster than Python on all of these (1.3 to 5.5 times on ordinary code). It is much slower than Node on ordinary code, because Node turns hot code into machine code while it runs and the Antares interpreter does not. Tensor maths runs as compiled code, which is why the matrix test beats both.

## Compiling with antares build

```shell
antares build program.ant
```

`antares build` turns a program into an executable placed next to the source file, in one of two ways:

- **Compiled** (`--native`): native code, much faster than the interpreter. Needs Rust.
- **Packed** (`--pack`): a copy of `antares` with the program, the files it brings in with `use` and any `--embed` files attached at the end. Runs at interpreter speed, needs nothing to build, and is made in under a second.

Without either option, `antares build` compiles when Rust (`cargo`) is installed and packs otherwise, saying so. Either way the program behaves the same as when interpreted; `python tests/compare_compiled.py` checks this on the test suites and examples (add `--pack` to check packed programs).

The rest of this chapter is about compiled programs.

The compiler looks at the whole program and works out which variables only ever hold a plain number, a yes/no, an integer, or a plain row of numbers. Those become machine numbers and direct memory, and functions called by name become direct calls. Everything else (text, lists, maps, tensors being learned, functions passed around as values) keeps working through the same runtime the interpreter uses.

| Test | Interpreted | Compiled |
|---|---|---|
| `fib(30)` | 0.28 s | 0.009 s |
| Chess move generation (perft), positions per second | 25,000 | 5,000,000 to 6,000,000 |
| Chess engine search, positions per second | 4,000 | 275,000 to 292,000 |

## Requirements and targets

- The compiler needs Rust (`cargo`) installed, because it writes Rust code and builds it. A build takes about 30 to 40 seconds. Install Rust from rustup.rs, or use `--pack`.
- Packed programs work on Windows, macOS and Linux: every change is checked to make sure packed programs behave exactly like interpreted ones on all three.
- Compiled programs are built against the Antares source code. `antares build` uses the folder named by the `ANTARES_HOME` environment variable if it is set, then the folder this copy of Antares was built from if it still exists. Otherwise it downloads the source of the matching release from GitHub once and keeps it in your cache folder, so a downloaded copy of Antares can compile programs too.
- By default the program is built for the processor of the machine that compiles it, so it may not run on an older one. Choose another target with `--cpu`: `x86-64` (any 64-bit PC), `x86-64-v2` (SSE4.2) or `x86-64-v3` (AVX2).

```shell
antares build program.ant out.exe --cpu x86-64
```

## Embedding files

`--embed file` packs a file into the executable (compiled or packed). The program reads it under the name `"embedded:<file name>"` with `read_binary`, and `exists` says whether it is there, so the same source can fall back to a file on disk when run interpreted. The chess engine embeds its 25 MB neural network this way.

## Icons

`--icon file.ico` gives a Windows executable its icon, the one Explorer and the taskbar show. A packed program needs nothing else, and neither does compiling with Microsoft's Rust toolchain; with the GNU toolchain it needs `windres` (part of MinGW binutils). If that isn't there the program is still built, without the icon, and `antares build` says so. On other systems the option is ignored.

## Writing fast compiled code

- **Constants.** A global given one literal number at the top level and never changed (`SIZE = 64`) is treated as a constant, and a list made only of such numbers is built once. Both make arithmetic like `x % SIZE` much cheaper.
- **Counters are integers.** Whole-number counters (`repeat ... as i`, `each i in range(...)`, and variables only ever changed by `i = i + 1` and the like) are machine integers, so positions like `row[i + 1]` cost only a range check. They still behave and print exactly like ordinary numbers.
- **Fixed tables.** A top-level row made once with `ints(n)` or `zeros(n)`, with a size known when compiling (a number, or arithmetic on such constants), and afterwards only used as `t[i]` and `t[i] = v`, becomes a fixed table: the fastest kind of row. Using the row as a whole anywhere (`len(t)`, passing `t` to a function, `each x in t`) turns this off.
- **Simple loops.** The innermost `each i in range(...)`, `repeat ... as i` or counting `while` loop (`while i < n { ...; i = i + 1 }`, where `n` doesn't change in the loop) that uses rows and fixed tables only as `row[i + c]` checks positions once, before it starts, and then reads and writes them directly.
- **Clearing and filling tables.** Giving a fixed table `ints(n)` or `zeros(n)` of the same size again (as a program does to clear its tables) keeps it a fixed table: the new assignment just fills it with zeros. `ints([...])` of constant numbers makes a fixed table that starts with them.
- **Constants from constants.** A top-level variable given, once, arithmetic on constants given before it (`AH = HIST + 1`, `SIZE = ROWS * 8`) is a constant as well.
- **Positions the compiler can prove.** A position into a fixed table that is certainly inside it needs no check. The compiler works this out from constants, loop counters, `x & 63`, `x % 8`, `x >> 3`, `lowest_bit(x)` inside `while x != 0` (or `when x != 0`), the numbers stored in tables, and what functions called by name are given and give back.
- **Ranges.** Where the compiler can't work a range out, say it: `board = ints(64) within 0 to 15`, `count = int(0) within 0 to 255`, `to put(p within 0 to 15, sq within 0 to 63)`. The check moves to where values are given (an input's check to the calls, where it is often left out because the value is known to fit), and every use as a position is then free. A function whose last check goes away also stops being able to fail, which makes calling it cheaper. In the chess engine, ranges on the board, the side to move, the move counter and the inputs of a dozen functions removed most of the checks in move generation.
- **Say what is a number.** One value of unknown kind can slow down everything it touches. If a number comes out of a map, a list or JSON, wrap it in `number(...)` so the compiler knows it is a number. In the chess engine, one such lookup made the whole board slow until it was wrapped.
- **Set before reading.** A number variable that is read before it is given a value reads as 0 in a compiled program, where the interpreter reports a problem.
- **Find the slow spots.** Build with the environment variable `ANTARES_EXPLAIN=1`. The compiler then names every variable that ended up as a general value and what it was given. A common cause is two files using the same global name for different things.

## Fast row arithmetic

A group of built-ins does common neural-network inner loops in a single call, without building intermediate rows. They were written for the chess engine's network evaluation:

| Function | What it does |
|---|---|
| `rows(matrix)` | The rows of a matrix as a list, so one can be picked without copying |
| `sum_rows(matrix, rows, count?)` | Adds up the chosen rows of a matrix |
| `update_rows(base, matrix, add, n_add, remove, n_remove, old?)` | `base` plus some rows of the matrix minus others, in one step. `old` lets the result reuse a row that is about to be replaced |
| `sum_product(a, b, ...)` | Multiplies rows position by position and adds everything up |
| `screlu(x, cap)` | Every number forced into 0..cap, then squared |
| `screlu_dot(x, weights, cap)` | The same as `sum_product(screlu(x, cap), weights)`, without building the row in between |
| `compact(x)` / `compact(x, rows, cols)` | Whole numbers from -32768 to 32767 stored in 2 bytes instead of 8. Much faster for the row built-ins on large tables |
| `expand(c)` | Compact numbers back as an ordinary row or matrix |

Compact numbers are read and changed by position like other rows. `m[r]` is row `r` of a compact matrix (a compact row) and `c[i]` a number of a compact row; `m[r] = row` and `c[i] = x` change them, with the usual copy-on-write, and `len` gives the number of rows (of a matrix) or numbers (of a row). Keeping many accumulators as the rows of one compact matrix and changing them by row is what makes network evaluation fast in a compiled program:

```antares
accs = compact(zeros(rows * width), rows, width)
accs[d] = update_rows(accs[s], weights, add, n_add, remove, n_remove)
accs[d] = accs[s]
total = screlu_dot(accs[r], out[k], 255)
```

The compiler does these three forms in place on the matrix's memory, with no row built in between; the results and problems are the same as doing them step by step.

## A real program: a chess engine

The NARC Genesys 5.1 chess engine has been rewritten in Antares: about 2,000 lines covering bitboard move generation, neural-network evaluation and the full search, verified against the original C++ engine. Compiled, it searches about 285,000 positions per second, roughly 60% of the speed of the original C++ generic build.

# Built-in Function Reference {#builtins}

Inputs in brackets `[ ]` are optional. "Tensor" means a number or any list of numbers.

## Making tensors

| Function | Description |
|---|---|
| `zeros(d1, d2, ...)` | Tensor of the given shape filled with 0. Sizes may also be one list |
| `ones(d1, d2, ...)` | Tensor filled with 1 |
| `rand(d1, ...)` | Uniform random numbers in 0..1. `rand()` gives one number |
| `randn(d1, ...)` | Normally distributed random numbers (mean 0, standard deviation 1) |
| `range(end)`, `range(start, end [, step])` | Evenly spaced numbers, end not included |
| `onehot(labels, n)` | Each label becomes a row of `n` with a 1 at its position |
| `seed(n)` | Sets the random seed so results repeat |

## Shape

| Function | Description |
|---|---|
| `shape(x)` | Sizes of each dimension, as a tensor |
| `reshape(x, d1, d2, ...)` | Same numbers, new shape |
| `t(m)` | Transpose of a matrix |
| `len(x)` | Length of a list, text, map, integer row, or first dimension of a tensor |

## Reductions

| Function | Description |
|---|---|
| `sum(x [, dim])` | Sum of all numbers, or along one dimension |
| `mean(x [, dim])` | Average of all numbers, or along one dimension |
| `max(a, ...)` | Largest number among all inputs |
| `min(a, ...)` | Smallest number among all inputs |
| `argmax(x)` | Position of the largest value along the last dimension |
| `argmax(row, first, last)` | Position of the first largest number from `first` up to (not including) `last`; -1 for an empty range |

## Maths (element by element)

| Function | Description |
|---|---|
| `exp(x)`, `log(x)` | Exponential and natural logarithm |
| `sqrt(x)`, `abs(x)` | Square root and absolute value |
| `sin(x)`, `cos(x)` | Trigonometry (radians) |
| `tanh(x)`, `sigmoid(x)`, `relu(x)` | Activation functions |
| `softmax(x)` | Probabilities along the last dimension |
| `floor(x)`, `round(x)`, `trunc(x)` | Rounding down, to nearest, toward zero |
| `clip(x, low, high)` | Forces every number into low..high |

## Fast row arithmetic

| Function | Description |
|---|---|
| `rows(matrix)` | The rows as a list, without copying |
| `sum_rows(matrix, rows [, count])` | Sum of the chosen rows |
| `update_rows(base, m, add, n_add, rem, n_rem [, old])` | Base plus some rows minus others |
| `sum_product(a, b, ...)` | Position-by-position product, all added up |
| `screlu(x, cap)` | Clamp to 0..cap, then square |
| `screlu_dot(x, w, cap)` | `sum_product(screlu(x, cap), w)` in one step |
| `compact(x [, rows, cols])` | 16-bit storage of whole numbers; `m[r]` is a row of a compact matrix |
| `expand(c)` | Compact numbers back to an ordinary tensor |

## Integers

| Function | Description |
|---|---|
| `int(x)` | `x` as a 64-bit integer |
| `ints(n)`, `ints(list)` | An integer row of `n` zeros, or from a list |
| `popcount(x)` | Number of set bits |
| `lowest_bit(x)` | Position of the lowest set bit (64 if none) |
| `highest_bit(x)` | Position of the highest set bit (-1 if none) |
| `within(x, lo, hi [, name])` | `x`, if it (or every number in it) is from `lo` to `hi`; a problem otherwise |
| `div(a, b)` | Division dropping the remainder |
| `rand_bits()` | 64 random bits |

## Learning

| Function | Description |
|---|---|
| `grad(loss, w)` | Gradient of `loss` with respect to the learnable value `w` |
| `gelu`, `silu`, `softplus`, `leaky_relu`, `log_softmax` | Activations |
| `cross_entropy(scores, answers)`, `mse(a, b)`, `bce(p, y)` | Losses |
| `take(table, positions)` | Rows by position (embedding lookup) |
| `concat(tensors [, dim])` | Join tensors |
| `layer_norm(x [, eps])`, `dropout(x, p)`, `causal_mask(n)` | Layers |
| `he(i, o)`, `xavier(i, o)` | Starting weights |
| `accuracy(scores, answers)`, `top(values, k)`, `sample(p [, temperature])` | Reading results |
| `read_csv(file)`, `columns(rows, names)`, `shuffle(x)`, `batches(x, size)` | Data |
| `gpu()` | The graphics card's name, or `nothing` (see `use gpu` in Chapter 11) |
| `use_gpu(yes or no)` | What `use gpu` and `use cpu` do |

The statements `learn` and `descend` are described in Chapter 11.

## Language models

| Function | Description |
|---|---|
| `conversation(model)` | A model that remembers the conversation |
| `history(chat)` | A conversation so far, as `{role, text}` maps |
| `image(file)` | A picture for a prompt |
| `embed(model, text)` | A row of numbers for a text (a table for a list of texts) |
| `similarity(a, b)` | Cosine similarity (one score per row when `b` is a table) |

The statements `model` and `ask` are described in Chapter 12.

## Lists and maps

| Function | Description |
|---|---|
| `push(list, item)` | A new list with `item` added at the end |
| `map(list, f)` | `f` applied to every item |
| `filter(list, f)` | The items for which `f` answers `yes` |
| `reduce(list, f, start)` | Combines items: `f(f(start, a), b)` ... |
| `sort(list)` | Sorted copy; all numbers or all text |
| `reverse(x)` | Reversed list or text |
| `slice(x, start [, end])` | Part of a list or text |
| `has(x, item)` | Whether a list contains an item, a map has a key, or a text contains a text |
| `find(x, item)` | Position of the first equal item in a list, or of a text in a text; -1 if not there |
| `insert(list, pos, item)` | A new list with `item` put in at `pos` |
| `remove(list, pos)` | A new list without the item at `pos` (negative counts from the end) |
| `unique(list)` | The items without repeats, in the order they first appear |
| `sort_by(list, f)` | Sorted copy, by the number or text `f` gives for each item |
| `keys(map)`, `values(map)` | Field names and field values |
| `join(list [, sep])` | Items joined into text |

## Text

| Function | Description |
|---|---|
| `upper(t)`, `lower(t)` | Change case |
| `trim(t)` | Remove spaces from both ends |
| `split(t [, sep])` | Split on spaces, or on a separator |
| `lines(t)` | Split into lines |
| `replace(t, from, to)` | Replace every occurrence |
| `text(v)` | Any value as text |
| `number(v)` | Text, yes/no or integer as an ordinary number |
| `fixed(x, places)` | A number as text with exactly that many digits after the point |
| `pad(v, width [, side [, fill]])` | Text padded to a width with spaces (or `fill`); numbers on the right, text on the left |
| `code(t)`, `char(n)` | The number of a character, and the character with a number |

## Data and files

| Function | Description |
|---|---|
| `save(value, file)`, `load(file)` | Store and restore any value as JSON |
| `exists(file)` | Whether a file (or embedded file) exists |
| `read(file)`, `write(file, value)` | Whole-file text input and output |
| `read_binary(file, kind, start, count)` | Whole numbers from a binary file |
| `json(t)`, `to_json(v)` | Parse and produce JSON |
| `program_folder()` | Folder of the running program |
| `append(file, value)`, `delete(file)` | Add to the end of a file; remove a file |
| `files([folder])` | Names in a folder |
| `args()` | What was typed after the program's name |
| `env(name)` | A setting of the computer, or `nothing` |
| `input([prompt])` | Read a line from the user |
| `input_waiting()` | Whether input is waiting |

## Commands and the web

| Function | Description |
|---|---|
| `shell(command [, options])` | Runs a command; gives `{output, errors, code}` |
| `fetch(url [, options])` | What a web address sends back, as text |
| `read_page(url)` | A web page's readable text |
| `page_text(html)` | The readable text of a page's HTML |
| `page_links(html [, address])` | A page's links, as `{text, url}` maps |

## Live screens

| Function | Description |
|---|---|
| `key()` | The pressed key, or `nothing` |
| `wait(seconds)` | Pause |
| `draw(text)` | Redraw the screen in place |
| `clear()` | Clear the screen |

## Games in a window

| Function | Description |
|---|---|
| `window(width, height [, title [, scale]])` | Open a window (Windows) |
| `window_open()` | `yes` until the window is closed |
| `frame()` | Show the picture; wait for the next of 60 frames a second; gives the seconds since the last frame |
| `fullscreen([yes or no])` | Fill the screen, or go back to a window; with nothing, whether it is fullscreen |
| `fill([colour])` | Paint the whole picture |
| `rect(x, y, width, height [, colour [, radius]])` | Filled rectangle, with rounded corners if a radius is given |
| `circle(x, y, radius [, colour])` | Filled circle |
| `ellipse(x, y, radius_x, radius_y [, colour])` | Filled ellipse |
| `ring(x, y, radius, thickness [, colour])` | Outline of a circle |
| `polygon(points [, colour])` | Filled shape through `[[x, y], ...]` |
| `line(x1, y1, x2, y2 [, colour [, width]])` | Line |
| `path(points [, colour [, width]])` | Lines joining points |
| `pixel(x, y [, colour])` | One pixel |
| `colour_at(x, y)` | A pixel's colour |
| `label(text, x, y [, colour [, size]])` | Text |
| `font([name [, size [, style]]])` | The font for labels; `font()` is the pixel font |
| `measure(text [, size])` | `{width, height}` of a label |
| `sprite(rows, x, y, colours [, size])` | Pixel art written as text |
| `picture(file)`, `picture(width, height)` | Read a PNG file; make a new picture |
| `draw(picture, x, y [, options])` | Draw a picture: `scale`, `width`, `height`, `angle`, `alpha`, `flip`, `part`, `tint`, `smooth` |
| `canvas([picture])` | Draw on a picture; `canvas()` goes back to the window |
| `gradient(x1, y1, x2, y2, colours)`, `radial(x, y, radius, colours)` | Gradients, used as colours |
| `hsl(hue, saturation, lightness [, alpha])` | A colour from its hue |
| `mix(colour1, colour2, amount)` | A colour between two |
| `blend([mode])` | `"normal"`, `"add"`, `"multiply"` or `"screen"` |
| `smooth(yes or no)` | Smooth or hard pixel edges |
| `shadow([across, down, softness [, colour]])` | Soft shadows for what is drawn next |
| `blur(radius [, x, y, width, height])`, `glow(amount [, radius [, threshold]])`, `vignette(amount)` | Effects on the whole picture |
| `held(key)` | Whether a key is held down |
| `pressed(key)` | Whether a key went down since the last frame |
| `mouse()` | `{x, y, left, right}` |
| `screenshot(file)` | Save the picture as a PNG |

## 3D

| Function | Description |
|---|---|
| `camera(x, y, z, look_x, look_y, look_z [, options])` | Where the camera is and what it looks at; options `fov`, `near`, `far` |
| `cube(x, y, z, size [, look])`, `box(x, y, z, width, height, depth [, look])` | A cube, a box |
| `sphere(x, y, z, radius [, look])` | A ball |
| `cylinder(x, y, z, radius, height [, look])`, `cone(x, y, z, radius, height [, look])` | A cylinder, a cone |
| `plane(x, y, z, width, depth [, look])` | A flat rectangle facing up |
| `mesh(file)`, `mesh(points, triangles [, places])`, `mesh(model, points)` | A model from an `.obj` file, or from corners and triangles; the same model with its corners moved |
| `draw(model, x, y, z [, look])` | Draw a model |
| `sun([x, y, z [, colour [, strength]]])` | Sunlight from a direction; `sun()` turns it off |
| `lamp(x, y, z [, colour [, strength [, reach]]])` | A light at a point, for this frame |
| `sky([top [, horizon [, ground]]])` | The sky; `sky()` leaves the background |
| `fog([colour,] near, far)`, `fog()` | Fog, or none |
| `shadows(yes or no)` | Whether the sun casts shadows |
| `render()` | Draw the scene |
| `to_screen(x, y, z)` | Where a point shows in the picture, or `nothing` |

Looks: a colour, or a map with `colour`, `shine`, `metal`, `glow`, `ripples`, `picture`, `tiles`, `turn`, `scale`, `shadow`.

## Sound

| Function | Description |
|---|---|
| `sound(file [, volume])` | Play a WAV file once |
| `music(file)`, `music(notes, step [, wave])` | Loop music; `music()` stops it |
| `beep(frequency, seconds [, wave])` | A tone |
| `tune(notes [, step [, wave]])` | Play a tune once; its length in seconds |
| `effect(name)` | A ready-made sound effect |
| `volume(level)` | Overall loudness, 0 to 1 |
| `silence()` | Stop all sound |
| `playing()` | How many sounds are playing |

## Other

| Function | Description |
|---|---|
| `kind(v)` | The kind of a value, as text |
| `clock()` | Current time in seconds |
| `now()` | The local date and time, as a map |
| `check(cond [, message])` | Raise a problem when `cond` is `no` |
| `fail([message])` | Raise a problem |

# How Antares Is Built {#internals}

## The pipeline

A program passes through these stages:

```diagram
source text  ->  lexer  ->  tokens  ->  parser  ->  tree with resolved slots
                                                     |
                     +-------------------------------+-----------------+
                     |                                                 |
             vm.rs: compact instructions                 aot.rs: type inference
             run in a tight loop                         -> Rust code -> native executable
             (interp.rs for rarer constructs)            (linked with rt.rs)
```

The parser resolves every name to a numbered slot up front, so the engine never looks names up while running. Tensor operations and their gradients live in `tensor.rs` and are shared by the interpreter and compiled programs.

## Source files

| File | What it does |
|---|---|
| `src/main.rs` | Command line, interactive session, problem reports |
| `src/lexer.rs` | Source text to tokens |
| `src/parser.rs` | Tokens to a tree; resolves every name to a slot |
| `src/ast.rs` | The shape of the tree |
| `src/vm.rs` | The fast engine: translates functions to instructions and runs them |
| `src/interp.rs` | Values, indexing, and the fallback evaluator for rarer constructs |
| `src/value.rs` | The value kinds and how they print |
| `src/nn.rs` | Deep learning and data: activations, losses, layers, starting weights, sampling, CSV and batches |
| `src/gpu.rs` | The GPU: Direct3D 11 compute shaders for matrix products, element-wise maths, gradients and optimisers |
| `src/gpu_cuda.rs` | The same on Linux with NVIDIA GPUs, through CUDA |
| `src/tensor.rs` | Tensors, broadcasting, reverse-mode autodiff, the optimisers, and matrix maths on every core |
| `src/builtins.rs` | The built-in functions |
| `src/gfx.rs` | Games in a window: drawing, the font, PNG screenshots, and the Windows window |
| `src/sound.rs` | Sound: the mixer, WAV files, tones, tunes and effects, and Windows sound output |
| `src/llm.rs` | Language-model calls through `curl` |
| `src/web.rs` | Commands (`shell`) and the web (`fetch` through `curl`, and reading pages) |
| `src/json.rs` | JSON reading and writing |
| `src/aot.rs` | The compiler behind `antares build --native` |
| `src/pack.rs` | `antares build --pack`: attaches a program to a copy of `antares` |
| `src/rt.rs` | Support code linked into compiled programs |

## Current limits

This is version 0.11. Not there yet:

- Giving a value to a slice (`v[1:3] = 0`).
- The GPU on macOS, and on Linux without an NVIDIA GPU.
- Game windows and sound on macOS and Linux.

# Grammar Summary {#grammar .appendix}

An informal grammar of Antares. `{ x }` means zero or more, `[ x ]` means optional, and quoted words are literal.

```grammar
program     = { statement NEWLINE }
block       = "{" { statement NEWLINE } "}"

statement   = "use" TEXT
            | "learn" NAME "=" expr
            | "model" NAME "=" expr [ "with" expr ]
            | "to" NAME "(" [ params ] ")" ( "=" expr | block )
            | "show" expr { "," expr }
            | "give" [ expr ]
            | "when" expr block { "else" "when" expr block } [ "else" block ]
            | "repeat" expr [ "as" NAME ] block
            | "each" NAME "in" expr block
            | "while" expr block
            | "descend" expr "by" expr [ "using" ( "adam" | "sgd" ) ]
            | "try" block "else" [ NAME ] block
            | "change" NAME ( "=" | "+=" | "-=" | "*=" | "/=" ) expr
            | "stop" | "skip"
            | NAME "=" expr [ range ]
            | target ( "=" | "+=" | "-=" | "*=" | "/=" ) expr
            | expr

target      = NAME { "." NAME | "[" index { "," index } "]" }
params      = param { "," param }
param       = NAME [ range ]
range       = "within" expr "to" expr

expr        = or_expr { "->" postfix }
or_expr     = and_expr { "or" and_expr }
and_expr    = not_expr { "and" not_expr }
not_expr    = "not" not_expr | comparison
comparison  = bits [ ( "==" | "!=" | "<" | "<=" | ">" | ">=" ) bits ]
bits        = shift { ( "&" | "|" | "xor" ) shift }
shift       = sum { ( "<<" | ">>" ) sum }
sum         = product { ( "+" | "-" ) product }
product     = unary { ( "*" | "/" | "%" | "@" ) unary }
unary       = ( "-" | "~" ) unary | power
power       = postfix [ "^" unary ]
postfix     = primary { "(" [ args ] ")" | "[" index { "," index } "]" | "." NAME }
index       = expr | [ expr ] ":" [ expr ]

primary     = NUMBER | HEX | TEXT | "yes" | "no" | "nothing" | NAME
            | "[" [ args ] "]"
            | "{" [ NAME ":" expr { "," NAME ":" expr } ] "}"
            | "(" expr ")"
            | lambda
            | "ask" postfix sum [ "as" ( "text" | "number" | "yes/no" | "list" ) ]
lambda      = ( NAME | "(" [ params ] ")" ) "=>" expr
args        = expr { "," expr }
```

# Example Programs {#examples .appendix}

## hello.ant: a tour of the basics

```antares
# The basics of Antares

name = "world"
show "Hello, {name}!"

# Numbers and tensors are the same thing
a = [1, 2, 3]
b = [[1, 2], [3, 4]]
show a * 2 + 1
show b @ b
show "mean of a is {mean(a)}"

# Functions are defined with `to`
to square(x) = x * x

to describe(n) {
    when n < 0 { give "negative" }
    when n == 0 { "zero" } else { "positive" }
}

show square(7), describe(-3), describe(0), describe(5)

# Pipelines read left to right
[3, 1, 2]
    -> sort
    -> map(x => x * 10)
    -> show_it

to show_it(x) { show "piped:", x }

# Lists, maps, loops
person = {name: "Ada", skills: ["maths", "engines"]}
person.age = 36
person.skills[0] = "mathematics"
show person

each skill in person.skills { show " -", upper(skill) }

total = 0
repeat 5 as i { total = total + i }
show "1+2+3+4+5 =", total

to fib(n) { when n < 2 { n } else { fib(n - 1) + fib(n - 2) } }
show "fib(20) =", fib(20)
```

```output
Hello, world!
[3, 5, 7]
[[7, 10],
 [15, 22]]
mean of a is 2
49 negative zero positive
piped: [10, 20, 30]
{name: "Ada", skills: ["mathematics", "engines"], age: 36}
 - MATHEMATICS
 - ENGINES
1+2+3+4+5 = 15
fib(20) = 6765
```

## train_save.ant: train once, reuse afterwards

```antares
# Train a network once, save it, and reuse it on later runs.
# Shows: use (imports), slices, try/else, save/load.

use "nn"

data = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
x = data[:, 0:2]          # every row, first two columns
y = data[:, 2:]           # every row, last column

to predict(net, input) = forward(net.hidden, input) -> tanh -> (h => forward(net.out, h)) -> sigmoid

file = "xor_model.json"

try {
    net = load(file)
    show "Loaded a trained model from {file}"
} else {
    show "No saved model yet, training one..."
    seed(7)
    net = {hidden: layer(2, 8), out: layer(8, 1)}
    repeat 1000 {
        descend mse(predict(net, x), y) by 0.05 using adam
    }
    save(net, file)
    show "Trained and saved to {file}"
}

show "predictions:", predict(net, x)[:, 0] -> round
```

## xor.ant: a neural network with progress reports

```antares
# A tiny neural network that learns XOR

seed(7)
x = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

learn w1 = randn(2, 8) * 0.5
learn b1 = zeros(8)
learn w2 = randn(8, 1) * 0.5
learn b2 = zeros(1)

to net(input) = input @ w1 + b1 -> tanh -> (h => h @ w2 + b2) -> sigmoid

repeat 1000 as step {
    loss = mean((net(x) - y) ^ 2)
    descend loss by 0.05 using adam
    when step % 250 == 0 { show "step {step}: loss {loss}" }
}

show "predictions:"
show net(x) -> round
```

```output
step 250: loss 6.1426e-5
step 500: loss 2.2382e-5
step 750: loss 1.1538e-5
step 1000: loss 7.0320e-6
predictions:
[[0],
 [1],
 [1],
 [0]]
```
