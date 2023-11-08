FROM rust

WORKDIR /app

COPY Cargo.toml Cargo.toml

COPY Cargo.lock Cargo.lock

RUN mkdir src && echo "fn main() {}" > src/main.rs

RUN cargo build --release --locked

RUN rm -rf src

COPY src src

RUN touch src/main.rs && cargo build --release --locked

CMD cargo run --release --locked
