def pytest_addoption(parser):
    parser.addoption("--examples-timeout", type=int, default=1000)
