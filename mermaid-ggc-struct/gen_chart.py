"""Generate a mermaid chart for a GGC structure."""

N = 4

def gen_chart(count: int = N, base: str = "", isa: bool = True) -> None:
    if count:
        print("\t" * (N - count + 2) + f"subgraph {base}{'BA'[isa]}")
        print("\t" * (N - count + 3) + "direction TB")
        gen_chart(count - 1, base + 'BA'[isa], True)
        gen_chart(count - 1, base + 'BA'[isa], False)
        print("\t" * (N - count + 2) + "end")


if __name__ == "__main__":
    print("flowchart LR\n\tsubgraph Top Level\n\t\tdirection TB")
    gen_chart()
    gen_chart(isa=False)
    print("\tend")
