"""Top-level LPDM orchestration entry point.

Implementation TODO:
- Generate initial release
- Backward integration loop with met fetch + GPU stepping
- Footprint accumulation and periodic merge
- Persist outputs
"""

from lpdm.runtime import DEVICE


def main() -> None:
	"""Temporary entrypoint while orchestration logic is being implemented."""

	print(f"LPDM runtime device: {DEVICE}")


if __name__ == "__main__":
	main()
