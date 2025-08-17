#[cfg(feature = "lisp")]
use anyhow::{bail, Result};

#[cfg(feature = "lisp")]
#[derive(Debug, Clone)]
pub enum Expr {
    Num(f32),
    Var,
    Call(String, Vec<Expr>),
}

#[cfg(feature = "lisp")]
#[derive(Clone, Copy)]
struct Cursor<'a> {
    s: &'a [u8],
    i: usize,
}

#[cfg(feature = "lisp")]
impl<'a> Cursor<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            s: s.as_bytes(),
            i: 0,
        }
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn bump(&mut self) {
        self.i += 1;
    }
    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.bump();
            } else {
                break;
            }
        }
    }
}

#[cfg(feature = "lisp")]
pub fn parse(input: &str) -> Result<Expr> {
    parse_expr(&mut Cursor::new(input))
}

#[cfg(feature = "lisp")]
fn parse_expr(cur: &mut Cursor<'_>) -> Result<Expr> {
    cur.skip_ws();
    match cur.peek() {
        Some(b'(') => parse_list(cur),
        Some(c) if (c as char).is_ascii_digit() || c == b'-' || c == b'+' => parse_number(cur),
        Some(_) => parse_symbol(cur),
        None => bail!("unexpected end of input"),
    }
}

#[cfg(feature = "lisp")]
fn parse_number(cur: &mut Cursor<'_>) -> Result<Expr> {
    let start = cur.i;
    if matches!(cur.peek(), Some(b'+') | Some(b'-')) {
        cur.bump();
    }
    while let Some(c) = cur.peek() {
        if (c as char).is_ascii_digit() {
            cur.bump();
        } else {
            break;
        }
    }
    if matches!(cur.peek(), Some(b'.')) {
        cur.bump();
        while let Some(c) = cur.peek() {
            if (c as char).is_ascii_digit() {
                cur.bump();
            } else {
                break;
            }
        }
    }
    let s = std::str::from_utf8(&cur.s[start..cur.i]).unwrap();
    let v: f32 = s.parse().map_err(|_| anyhow::anyhow!("invalid number"))?;
    Ok(Expr::Num(v))
}

#[cfg(feature = "lisp")]
fn parse_symbol(cur: &mut Cursor<'_>) -> Result<Expr> {
    let start = cur.i;
    while let Some(c) = cur.peek() {
        let ch = c as char;
        if ch.is_ascii_whitespace() || ch == '(' || ch == ')' {
            break;
        }
        cur.bump();
    }
    let s = std::str::from_utf8(&cur.s[start..cur.i]).unwrap();
    Ok(match s {
        "x" => Expr::Var,
        "pi" => Expr::Num(std::f32::consts::PI),
        "e" => Expr::Num(std::f32::consts::E),
        other => Expr::Call(other.to_string(), vec![]),
    })
}

#[cfg(feature = "lisp")]
fn parse_list(cur: &mut Cursor<'_>) -> Result<Expr> {
    // assume current is '('
    cur.bump();
    cur.skip_ws();
    // head
    let head = parse_symbol(cur)?;
    let name = match head {
        Expr::Call(n, _) => n,
        Expr::Var => "x".to_string(),
        Expr::Num(_) => bail!("list head must be symbol"),
    };
    let mut args = Vec::new();
    loop {
        cur.skip_ws();
        match cur.peek() {
            Some(b')') => {
                cur.bump();
                break;
            }
            Some(_) => args.push(parse_expr(cur)?),
            None => bail!("unterminated list"),
        }
    }
    Ok(Expr::Call(name, args))
}

#[cfg(feature = "lisp")]
pub fn eval(expr: &Expr, x: f32) -> Result<f32> {
    use std::f32::consts::{E, PI};
    let r = match expr {
        Expr::Num(n) => *n,
        Expr::Var => x,
        Expr::Call(name, args) => match (name.as_str(), args.as_slice()) {
            ("+", [a, b]) => eval(a, x)? + eval(b, x)?,
            ("-", [a, b]) => eval(a, x)? - eval(b, x)?,
            ("*", [a, b]) => eval(a, x)? * eval(b, x)?,
            ("/", [a, b]) => eval(a, x)? / eval(b, x)?,
            ("neg", [a]) => -eval(a, x)?,
            ("pow", [a, b]) => eval(a, x)?.powf(eval(b, x)?),
            ("sin", [a]) => eval(a, x)?.sin(),
            ("cos", [a]) => eval(a, x)?.cos(),
            ("tanh", [a]) => eval(a, x)?.tanh(),
            ("exp", [a]) => eval(a, x)?.exp(),
            ("log", [a]) => eval(a, x)?.ln(),
            ("abs", [a]) => eval(a, x)?.abs(),
            ("sqrt", [a]) => eval(a, x)?.sqrt(),
            ("sqr", [a]) => {
                let v = eval(a, x)?;
                v * v
            }
            ("pi", []) => PI,
            ("e", []) => E,
            _ => bail!("unknown call or wrong arity: {}", name),
        },
    };
    if r.is_finite() {
        Ok(r)
    } else {
        bail!("non-finite result")
    }
}
