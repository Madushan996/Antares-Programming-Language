use antares::{aot, builtins, interp, lexer, parser, value};

use std::io::{BufRead, Write};
use value::Value;

const VERSION: &str = env!("CARGO_PKG_VERSION");

struct Session {
    parser: parser::Parser,
    interp: interp::Interp,
}

impl Session {
    fn new() -> Self {
        let b = builtins::all();
        let names: Vec<&str> = b.iter().map(|(n, _)| *n).collect();
        let globals: Vec<Value> = b.iter().map(|(n, f)| Value::Native(n, *f)).collect();
        Session { parser: parser::Parser::new(&names), interp: interp::Interp::new(globals) }
    }

    fn run(&mut self, src: &str) -> Result<Value, (String, u32)> {
        let toks = lexer::lex(src, 1)?;
        let stmts = self.parser.parse(toks).map_err(|e| (e.msg, e.line))?;
        self.interp.ensure_globals(self.parser.global_count());
        self.interp.run(&stmts)
    }
}

fn report(session: &Session, src: &str, msg: &str, line: u32) {
    eprintln!();
    // The top bits of a line number say which `use`d file it is in.
    let (file, line) = (line >> 20, line & 0xFFFFF);
    let (src, place) = match if file > 0 { session.parser.file(file) } else { None } {
        Some((name, text)) => (text, format!(" of {}", name)),
        None => (src, String::new()),
    };
    if line > 0 {
        eprintln!("Problem on line {}{}: {}", line, place, msg);
        if let Some(text) = src.lines().nth(line as usize - 1) {
            eprintln!("  {:>4} | {}", line, text.trim_end());
        }
    } else {
        eprintln!("Problem: {}", msg);
    }
}

fn run_file(path: &str) -> i32 {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Couldn't open \"{}\": {}", path, e);
            return 1;
        }
    };
    let mut session = Session::new();
    session.parser.set_main(path);
    builtins::set_program_folder(std::path::Path::new(path));
    match session.run(&src) {
        Ok(_) => 0,
        Err((msg, line)) => {
            let _ = std::io::stdout().flush();
            report(&session, &src, &msg, line);
            1
        }
    }
}

fn balance(src: &str) -> i32 {
    let mut depth = 0;
    let mut in_text = false;
    let mut prev = '\0';
    for c in src.chars() {
        if c == '"' && prev != '\\' {
            in_text = !in_text;
        } else if !in_text {
            match c {
                '{' | '(' | '[' => depth += 1,
                '}' | ')' | ']' => depth -= 1,
                _ => {}
            }
        }
        prev = c;
    }
    depth
}

fn repl() -> i32 {
    println!("Antares {} — type a line and press Enter. `exit` to leave.", VERSION);
    let mut session = Session::new();
    let stdin = std::io::stdin();
    let mut buf = String::new();
    loop {
        print!("{}", if buf.is_empty() { "> " } else { ". " });
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        if buf.is_empty() && matches!(line.trim(), "exit" | "quit") {
            break;
        }
        buf.push_str(&line);
        if balance(&buf) > 0 {
            continue;
        }
        let src = std::mem::take(&mut buf);
        match session.run(&src) {
            Ok(Value::Nothing) | Ok(Value::Unset) => {}
            Ok(v) => println!("{}", v.repr(0)),
            Err((msg, line)) => report(&session, &src, &msg, line),
        }
    }
    0
}

fn real_main() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        None => repl(),
        Some("--version") | Some("-v") | Some("version") => {
            println!("Antares {}", VERSION);
            0
        }
        Some("--help") | Some("-h") | Some("help") => {
            println!("Antares {} — a small language for AI\n", VERSION);
            println!("  antares                 start an interactive session");
            println!("  antares program.ant     run a program");
            println!("  antares run program.ant the same");
            println!("  antares build program.ant   compile it to a fast native program");
            println!("      [output.exe] [--cpu native|x86-64|x86-64-v2|x86-64-v3] [--embed file] [--icon file.ico]");
            println!("  antares version         show the version");
            0
        }
        Some("build") => match args.get(1) {
            Some(p) => {
                // antares build file.ant [output] [--cpu NAME]
                let mut out = None;
                let mut cpu = None;
                let mut embed: Vec<String> = vec![];
                let mut icon = None;
                let mut i = 2;
                while i < args.len() {
                    if args[i] == "--cpu" {
                        cpu = args.get(i + 1).map(|s| s.as_str());
                        i += 2;
                    } else if args[i] == "--icon" {
                        icon = args.get(i + 1).map(|s| s.as_str());
                        i += 2;
                    } else if args[i] == "--embed" {
                        if let Some(f) = args.get(i + 1) {
                            embed.push(f.clone());
                        }
                        i += 2;
                    } else {
                        out = Some(args[i].as_str());
                        i += 1;
                    }
                }
                aot::build(p, out, cpu, &embed, icon)
            }
            None => {
                eprintln!("Which file? Try: antares build program.ant");
                1
            }
        },
        Some("run") => match args.get(1) {
            Some(p) => run_file(p),
            None => {
                eprintln!("Which file? Try: antares run program.ant");
                1
            }
        },
        Some(p) => run_file(p),
    }
}

fn main() {
    // Run on a thread with a large stack so deep recursion in Antares
    // programs gives a friendly error instead of crashing.
    let child = std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(real_main)
        .expect("couldn't start");
    let code = child.join().unwrap_or(1);
    std::process::exit(code);
}
