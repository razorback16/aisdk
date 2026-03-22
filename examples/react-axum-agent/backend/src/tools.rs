use aisdk::{core::Tool, macros::tool};

#[tool]
/// get the weather for a location
pub async fn get_weather(location: String) -> Tool {
    // sleep for 1 second
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    Result::Ok(format!("The weather in {} is sunny", location))
}
