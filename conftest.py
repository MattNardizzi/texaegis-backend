def pytest_addoption(parser):
    parser.addoption(
        "--enable-llm",
        action="store_true",
        default=False,
        help="Enable LLM semantic analysis in tests (requires OPENAI_API_KEY)",
    )
