.PHONY: test demo eval lint clean

test:
	uv run pytest tests/ -v

demo:
	uv run ase run --model mock:careful --model mock:reckless --model mock:lazy --model mock:injection_victim --suite v1 --seeds 3 --out results/demo
	uv run ase report results/demo

eval:
	uv run ase run --model gemini:gemini-2.5-flash --model anthropic:claude-sonnet-4-20250514 --suite v1 --seeds 3 --out results/eval
	uv run ase report results/eval

lint:
	uv run python -m py_compile agent_safety_eval/cli.py

clean:
	rm -rf results/demo results/eval dist/ *.egg-info/
