# Introduction {#intro}

Antares is a small programming language for artificial intelligence. Tensors, automatic differentiation and calls to large language models are part of the language itself, not libraries you have to install and import. A model that learns, a neural network, and a program that asks Claude a question are each a few lines of plain, readable code.

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

This manual describes **Antares 0.5.0**. Chapters 2 to 9 cover the core language: values, operators, control flow, functions, errors and modules. Chapters 10 to 15 cover what makes Antares different: tensors, learning, language models, integers, files and live terminal programs. Chapter 16 explains performance and the native compiler. Chapter 17 is a complete reference of built-in functions, and the appendices give a grammar summary and complete example programs.

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
Antares 0.5.0 — type a line and press Enter. `exit` to leave.
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

## Optimisers: sgd and adam

```antares
descend loss by 0.05                 # plain gradient descent
descend loss by 0.05 using sgd       # the same, said explicitly
descend loss by 0.05 using adam      # the Adam optimiser
```

The number after `by` is the learning rate. Adam keeps a running memory of past gradients for each learnable value and usually trains neural networks much faster.

## Gradients without changing anything

`grad(loss, w)` returns the gradient of `loss` with respect to the learnable value `w`, with the same shape as `w`, and changes nothing:

```antares
learn w = [1, 2, 3]
show grad(sum(w ^ 2), w)             # [2, 4, 6]
```

## What gradients flow through

Gradients flow through `+ - * / ^ @`, indexing and slices, the functions `exp log tanh sigmoid relu sqrt abs sin cos`, and `sum mean softmax t reshape` (including `sum` and `mean` along a dimension). `floor`, `round`, `trunc` and comparisons produce plain values that do not carry a gradient.

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

## Classification with softmax

Cross-entropy classification is written directly from the formula:

```antares
target = onehot(labels, 2)
learn cw = zeros(2, 2)
repeat 200 {
    ce = -sum(target * log(softmax(pts @ cw))) / 60
    descend ce by 0.5
}
```

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

The text names the provider and model, separated by a colon. A name that starts with `claude` may leave out the `anthropic:` prefix. The optional `with` map sets options:

| Option | Kind | Meaning | Default |
|---|---|---|---|
| `system` | text | A system prompt that guides every answer | none |
| `max_tokens` | number | The longest answer allowed, in tokens | 16000 |

## Asking

`ask model prompt` sends the prompt and gives the reply as text. Text interpolation makes it easy to build prompts:

```antares
answer = ask claude "Explain {topic} in one sentence."
```

## Typed answers

`as` turns the reply into a real value you can calculate with:

```antares
count = ask claude "How many moons does Mars have?" as number
sure  = ask claude "Is 91 a prime number?" as yes/no
ideas = ask claude "Give me five names for a cat." as list
```

| Form | Result |
|---|---|
| `as text` | The reply as text (the same as leaving `as` out) |
| `as number` | The number in the reply |
| `as yes/no` | `yes` or `no` |
| `as list` | A list of text, one item per line of the reply |

If the reply cannot be read as the requested kind, that is a problem that `try` can catch.

## Setting up access

Set the `ANTHROPIC_API_KEY` environment variable to your API key before running the program. Requests are sent with the `curl` program that comes with Windows, macOS and Linux, so Antares needs no networking library and stays small.

## Working offline: the echo model

`model fake = "echo"` makes a pretend model that repeats the prompt back. Programs can be written and tested without a network connection or API key, then switched to a real model by changing one line:

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

Network errors, a missing key and unreadable answers are ordinary problems, so wrap calls that must not stop the program in `try`:

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
- Packed programs are checked on Windows. On macOS, attaching data to a program may upset its code signature; compile with `--native` there if a packed program won't start.
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

The statements `learn` and `descend` are described in Chapter 11.

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

## Live screens

| Function | Description |
|---|---|
| `key()` | The pressed key, or `nothing` |
| `wait(seconds)` | Pause |
| `draw(text)` | Redraw the screen in place |
| `clear()` | Clear the screen |

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
| `src/tensor.rs` | Tensors, broadcasting, reverse-mode autodiff, SGD and Adam |
| `src/builtins.rs` | The built-in functions |
| `src/llm.rs` | Language-model calls through `curl` |
| `src/json.rs` | JSON reading and writing |
| `src/aot.rs` | The compiler behind `antares build --native` |
| `src/pack.rs` | `antares build --pack`: attaches a program to a copy of `antares` |
| `src/rt.rs` | Support code linked into compiled programs |

## Current limits

This is version 0.5. Not there yet:

- Giving a value to a slice (`v[1:3] = 0`).
- Tools and multi-turn chat for language models.
- Model providers other than Anthropic.
- GPU support.

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
