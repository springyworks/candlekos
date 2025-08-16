pub const WITH_TIMER: bool = true;

mod app;
mod audio;
pub mod languages;
pub mod whisper_worker;
pub use app::App;
pub use whisper_worker::Worker;
