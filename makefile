.PHONY: run


VERSION := $(shell uv run script/version.py)
GIT_REV := $(shell git rev-parse --short HEAD)
USER_SHELL := $(shell echo $$SHELL)
NIX_FLAGS := -L --extra-experimental-features "nix-command flakes"


define nix_run 
	if [ "$$IN_NIX_SHELL" = "" ]; then \
		nix develop $(NIX_FLAGS) --system $(1) --command bash -c "$(2)"; \
	else \
		$(2); \
	fi
endef


ifneq ($(shell uname -s),Darwin)
	$(error This software only available for Apple Silicon MacOS)
else ifneq ($(shell uname -m),arm64)
	$(error This software only available for Apple Silicon MacOS)
endif


help:
	@echo "Help page for zaneml. Version: $(VERSION)"


run:
	@echo "Running zaneml version: $(VERSION)"
	zig build run

perceptron:
	@echo "Running zaneml example/perceptron.zig: $(VERSION)"
	zig build perceptron
