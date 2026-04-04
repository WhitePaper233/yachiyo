use rag::{EmbeddingInput, EmbeddingProvider, EmbeddingRequest, OpenAiEmbeddingProvider};

/// 测试连接本地 Ollama 的 embedding 服务
///
/// 使用前请确保：
/// 1. Ollama 已安装并运行在 `http://localhost:11434`
/// 2. 已拉取 embedding 模型，例如：`ollama pull bge-m3`
///
/// 运行方式：`cargo test --package rag --test ollama_test -- --nocapture`
#[tokio::test]
async fn test_ollama_embedding() {
    // 连接到本地 Ollama（OpenAI 兼容接口）
    let provider = OpenAiEmbeddingProvider::with_base_url(
        "ollama",
        "http://localhost:11434/v1",
    );

    // 使用 bge-m3 模型（先确保 `ollama pull bge-m3`）
    let request = EmbeddingRequest::new("bge-m3", EmbeddingInput::text("Hello, Ollama embedding!"));

    let response = provider.embed(request).await.expect(
        "Failed to connect to Ollama. Make sure Ollama is running and bge-m3 model is pulled.",
    );

    let embedding = response.first().expect("No embedding returned");

    println!("Model: {}", response.model);
    println!("Embedding dimension: {}", embedding.len());
    println!(
        "First 10 values: {:?}",
        &embedding[..10.min(embedding.len())]
    );

    if let Some(usage) = &response.usage {
        println!("Tokens used: {}", usage.total_tokens);
    }

    // 基本断言验证返回结果合理性
    assert!(!embedding.is_empty(), "Embedding should not be empty");
    assert_eq!(
        response.data.len(),
        1,
        "Should return exactly one embedding result"
    );
}

/// 测试批量嵌入
#[tokio::test]
async fn test_ollama_batch_embedding() {
    let provider = OpenAiEmbeddingProvider::with_base_url("ollama", "http://localhost:11434/v1");

    let texts = vec![
        "Rust is a systems programming language".into(),
        "Machine learning models can be embedded".into(),
        "Retrieval augmented generation improves LLM accuracy".into(),
    ];

    let request = EmbeddingRequest::new("bge-m3", EmbeddingInput::texts(texts));

    let response = provider
        .embed(request)
        .await
        .expect("Failed to batch embed with Ollama");

    println!("Batch embedding count: {}", response.data.len());
    for (_, data) in response.data.iter().enumerate() {
        println!(
            "  [{}] dimension={}, first 5 values: {:?}",
            data.index,
            data.embedding.len(),
            &data.embedding[..5.min(data.embedding.len())]
        );
    }

    assert_eq!(
        response.data.len(),
        3,
        "Should return three embedding results"
    );
    assert!(
        response.data.iter().all(|d| !d.embedding.is_empty()),
        "All embeddings should be non-empty"
    );
}
