use aisdk::{
    core::LanguageModelRequest,
    integrations::{
        axum::AxumSseResponse,
        vercel_aisdk_ui::VercelUIRequest,
    },
    providers::OpenAI,
};

// Example handler function
async fn chat_handler(
    axum::Json(request): axum::Json<VercelUIRequest>,
) -> AxumSseResponse {

    // Convert the Message sent by the frontend to AISDK.rs Messages
    let messages = request.into();

    // Generate streaming response
    let response = LanguageModelRequest::builder()
        .model(OpenAI::gpt_4o())
        .messages(messages)
        .build()
        .stream_text()
        .await
        .unwrap();

    // Convert to Axum SSE response (Vercel UI compatible)
    response.into()
}

#[tokio::main]
async fn main() {
    let app = axum::Router::new()
        .route("/api/chat", axum::routing::post(chat_handler))
        .layer(tower_http::cors::CorsLayer::permissive());

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8080));
    println!("Listening on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
