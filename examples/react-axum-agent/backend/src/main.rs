use aisdk::{
    core::LanguageModelRequest,
    integrations::{axum::AxumSseResponse, vercel_aisdk_ui::VercelUIRequest},
    providers::Google,
};

mod tools;

// Example handler function
async fn chat_handler(axum::Json(request): axum::Json<VercelUIRequest>) -> AxumSseResponse {
    println!("Request: {:?}", request);

    // Convert the Message sent by the frontend to AISDK.rs Messages
    let messages = request.into();

    // Generate streaming response
    let response = LanguageModelRequest::builder()
        .model(Google::gemini_2_5_flash())
        .system("You are a helpful assistant.")
        .messages(messages)
        .with_tool(tools::get_weather())
        .build()
        .stream_text()
        .await;

    match response {
        Ok(response) => {
            println!("sending response chunk to client");

            // Convert to Axum SSE response (Vercel UI compatible)
            response.into()
        }
        Err(err) => {
            panic!("Error: {:?}", err);
        }
    }
}

#[tokio::main]
async fn main() {
    // load .env
    dotenv::dotenv().ok();

    let app = axum::Router::new()
        .route("/api/chat", axum::routing::post(chat_handler))
        .layer(tower_http::cors::CorsLayer::permissive());

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
