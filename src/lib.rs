//! Antares: a small language for AI.
//!
//! The same code serves the `antares` program (which interprets `.ant`
//! files) and programs produced by `antares build` (which link this library
//! for its values, tensors and built-in functions).

pub mod aot;
pub mod ast;
pub mod builtins;
pub mod gfx;
pub mod interp;
pub mod json;
pub mod lexer;
pub mod llm;
pub mod pack;
pub mod parser;
pub mod rt;
pub mod tensor;
pub mod value;
pub mod vm;
