def pytest_addoption(parser):
    parser.addoption("--timeout", type=int, default=100)
