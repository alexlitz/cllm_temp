use std::env;
use std::fs;
use std::io::{self, Read};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() <= 1 {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf).unwrap();
        print!("{}", buf);
    } else {
        for path in &args[1..] {
            match fs::read_to_string(path) {
                Ok(contents) => print!("{}", contents),
                Err(e) => eprintln!("cat: {}: {}", path, e),
            }
        }
    }
}
