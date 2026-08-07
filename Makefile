CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Werror

.PHONY: test pytest
pytest:
	.venv/bin/pytest -q
test: pytest
