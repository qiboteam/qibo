# -*- coding: utf-8 -*-
def pytest_addoption(parser):
    parser.addoption("--examples-timeout", type=int, default=100000)
