use std::collections::HashMap;
use anyhow::{anyhow, Result};
use candle_core::{DType, Tensor};

// ---------------- Parsing ----------------

#[derive(Debug, Clone)]
enum Ast {
    Num(f64),
    Var(String),
    Unary(char, Box<Ast>),
    Binary(Box<Ast>, char, Box<Ast>),
    Call(String, Vec<Ast>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokKind { Num, Ident, Sym, LParen, RParen, Comma, End }

#[derive(Debug, Clone)]
struct Tok { kind: TokKind, text: String }

fn lex(input: &str) -> Result<Vec<Tok>> {
    let mut toks = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() { i+=1; continue; }
        if c.is_ascii_digit() || (c=='.') { // number
            let start=i; let mut dot = c=='.'; i+=1;
            while i < chars.len() {
                let d=chars[i];
                if d.is_ascii_digit() { i+=1; continue; }
                if d=='.' && !dot { dot=true; i+=1; continue; }
                break;
            }
            toks.push(Tok{kind:TokKind::Num,text:chars[start..i].iter().collect()});
            continue;
        }
        if c.is_ascii_alphabetic() || c=='_' { let start=i; i+=1; while i<chars.len() && (chars[i].is_ascii_alphanumeric()||chars[i]=='_') { i+=1; } toks.push(Tok{kind:TokKind::Ident,text:chars[start..i].iter().collect()}); continue; }
        match c {
            '(' => toks.push(Tok{kind:TokKind::LParen,text:"(".into()}),
            ')' => toks.push(Tok{kind:TokKind::RParen,text:")".into()}),
            ',' => toks.push(Tok{kind:TokKind::Comma,text:",".into()}),
            '+'|'-'|'*'|'/'|'^' => toks.push(Tok{kind:TokKind::Sym,text:c.to_string()}),
            _ => return Err(anyhow!("unexpected character `{c}`")),
        }
        i+=1;
    }
    toks.push(Tok{kind:TokKind::End,text:"".into()});
    Ok(toks)
}

struct Parser { toks: Vec<Tok>, pos: usize }
impl Parser {
    fn new(toks: Vec<Tok>) -> Self { Self{toks,pos:0} }
    fn cur(&self) -> &Tok { &self.toks[self.pos] }
    fn eat(&mut self, kind: TokKind) -> Result<()> { if self.cur().kind==kind { self.pos+=1; Ok(()) } else { Err(anyhow!("expected {:?}", kind)) } }
    fn parse(mut self) -> Result<Ast> { let ast = self.expr()?; if self.cur().kind!=TokKind::End { return Err(anyhow!("unexpected trailing input")); } Ok(ast) }
    fn expr(&mut self) -> Result<Ast> { let mut node = self.term()?; while matches!(self.cur().text.as_str(), "+"|"-") { let op = self.cur().text.chars().next().unwrap(); self.pos+=1; let rhs = self.term()?; node = Ast::Binary(Box::new(node), op, Box::new(rhs)); } Ok(node) }
    fn term(&mut self) -> Result<Ast> { let mut node = self.power()?; while matches!(self.cur().text.as_str(), "*"|"/") { let op=self.cur().text.chars().next().unwrap(); self.pos+=1; let rhs=self.power()?; node=Ast::Binary(Box::new(node),op,Box::new(rhs)); } Ok(node) }
    fn power(&mut self) -> Result<Ast> { // right associative
        let mut node = self.unary()?; if self.cur().text=="^" { self.pos+=1; let rhs = self.power()?; node = Ast::Binary(Box::new(node),'^',Box::new(rhs)); } Ok(node) }
    fn unary(&mut self) -> Result<Ast> { if matches!(self.cur().text.as_str(), "+"|"-") { let op=self.cur().text.chars().next().unwrap(); self.pos+=1; let inner=self.unary()?; Ok(Ast::Unary(op,Box::new(inner))) } else { self.primary() } }
    fn primary(&mut self) -> Result<Ast> {
        match self.cur().kind {
            TokKind::Num => { let v=self.cur().text.clone(); self.pos+=1; Ok(Ast::Num(v.parse::<f64>()?)) }
            TokKind::Ident => {
                let name = self.cur().text.clone(); self.pos+=1;
                if self.cur().kind==TokKind::LParen { // function call
                    self.pos+=1; // consume (
                    let mut args=Vec::new();
                    if self.cur().kind!=TokKind::RParen { loop { args.push(self.expr()?); if self.cur().kind==TokKind::Comma { self.pos+=1; continue; } break; } }
                    self.eat(TokKind::RParen)?;
                    Ok(Ast::Call(name,args))
                } else { Ok(Ast::Var(name)) }
            }
            TokKind::LParen => { self.pos+=1; let e=self.expr()?; self.eat(TokKind::RParen)?; Ok(e) }
            _ => Err(anyhow!("unexpected token {:?}", self.cur()))
        }
    }
}

// ------------- Environment & Evaluation -------------

pub struct ExprEnv {
    pub x: Tensor,
    pub y: Tensor,
    pub params: HashMap<String,f64>,
}

impl ExprEnv { pub fn new(x: Tensor, y: Tensor, params: HashMap<String,f64>) -> Result<Self> { Ok(Self{x,y,params}) } }

fn scalar(env: &ExprEnv, v:f64) -> Result<Tensor> { Ok(Tensor::new(&[v], env.x.device())?.to_dtype(DType::F64)?.broadcast_as(env.x.shape())?) }

pub fn eval_expr(expr:&str, env:&ExprEnv) -> Result<Tensor> {
    let toks = lex(expr)?;
    let ast = Parser::new(toks).parse()?;
    fn eval(node:&Ast, env:&ExprEnv) -> Result<Tensor> {
        match node {
            Ast::Num(v) => scalar(env,*v),
            Ast::Var(name) => match name.as_str() {
                "x" => Ok(env.x.clone()),
                "y" => Ok(env.y.clone()),
                "pi" => scalar(env,std::f64::consts::PI),
                "e"  => scalar(env,std::f64::consts::E),
                other => {
                    if let Some(v)=env.params.get(other) { scalar(env,*v) } else { Err(anyhow!("unknown identifier: {other}")) }
                }
            },
            Ast::Unary(op, inner) => {
                let t = eval(inner, env)?; match op { '+' => Ok(t), '-' => Ok((-1f64 * t)?), _ => unreachable!() }
            }
            Ast::Binary(a, op, b) => {
                let lhs = eval(a, env)?; let rhs = eval(b, env)?;
                let out = match op {
                    '+' => (lhs + rhs)?,
                    '-' => (lhs - rhs)?,
                    '*' => (lhs * rhs)?,
                    '/' => (lhs / rhs)?,
                    '^' => { // tensor ^ tensor : exp(rhs * ln(lhs))
                        let ln_l = lhs.log()?; (rhs * ln_l)?.exp()? }
                    _ => return Err(anyhow!("unsupported binary operator: {op}"))
                }; Ok(out)
            }
            Ast::Call(name,args) => {
                let lower = name.to_ascii_lowercase();
                let mut eval_args = Vec::with_capacity(args.len());
                for a in args { eval_args.push(eval(a, env)?); }
                let t = match lower.as_str() {
                    "sin" | "cos" | "tanh" | "exp" | "log" | "sqrt" | "abs" | "floor" | "ceil" => {
                        if eval_args.len() != 1 { return Err(anyhow!("{lower} expects 1 arg")); }
                        let a = &eval_args[0];
                        match lower.as_str() {
                            "sin" => a.sin()?,
                            "cos" => a.cos()?,
                            "tanh"=> a.tanh()?,
                            "exp" => a.exp()?,
                            "log" => a.log()?,
                            "sqrt"=> a.sqrt()?,
                            "abs" => a.abs()?,
                            "floor"=> a.floor()?,
                            "ceil" => a.ceil()?,
                            _ => unreachable!(),
                        }
                    }
                    "pow" | "min" | "max" => {
                        if eval_args.len() != 2 { return Err(anyhow!("{lower} expects 2 args")); }
                        let a = &eval_args[0];
                        let b = &eval_args[1];
                        match lower.as_str() {
                            "pow" => { (b * a.log()?)?.exp()? },
                            "min" => a.minimum(b)?,
                            "max" => a.maximum(b)?,
                            _ => unreachable!(),
                        }
                    }
                    "clamp" => {
                        if eval_args.len() != 3 { return Err(anyhow!("clamp expects 3 args")); }
                        eval_args[0].maximum(&eval_args[1])?.minimum(&eval_args[2])?
                    }
                    _ => return Err(anyhow!("unsupported function: {lower}")),
                };
                Ok(t)
            }
        }
    }
    let t = eval(&ast, env)?;
    Ok(t.to_dtype(DType::F32)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::collections::HashMap;

    #[test]
    fn simple() -> Result<()> {
        let dev = Device::Cpu;
        // Use a 1-D shape so to_vec1 works directly.
        let x = Tensor::zeros(5, DType::F64, &dev)?; // shape [5]
        let y = x.clone();
        let env = ExprEnv::new(x, y, HashMap::new())?;
        let t = eval_expr("sin(0)+cos(0)", &env)?; // should be 1.
        let v: Vec<f32> = t.to_vec1()?;
        assert!((v[0] - 1.0).abs() < 1e-6);
        Ok(())
    }
}
