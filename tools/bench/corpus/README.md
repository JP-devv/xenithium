# The benchmark corpus

Frozen text the long prompts of `xenithium-bench` are cut from, embedded into the binary at build time
(`xenithium-bench-embed`). Two sets, each `<set>/<kind>/<file>`:

- `hard/` (`--corpus hard`, the hard mode's): `code/` holds snapshots of this engine's own sources, `prose/` of its
  documentation, both as of commit f33d9ee.
- `medium/` (`--corpus medium`, the medium and easy modes'): conventional, template-made code (entity classes, CRUD
  repositories and services, tests, API clients, UI components, SQL, manifests, classic algorithms) and prose (a travel
  guide, product manuals, recipes, company histories, help-centre articles, local news), generated once from a fixed
  seed. Named `easy` before the easy mode got prompts of its own; the files are unchanged.

They are NOT kept in step with the tree: a benchmark number is comparable between two builds only if both read the
same text.

Changing, adding or removing a file of a set changes every prompt cut from it. Bump its version in
`tools/bench/corpus.hpp` (`kCorpusVersion` for hard, `kMediumCorpusVersion` for medium) and its pinned hash in
`tests/test_bench.cpp` in the same change; runs of different corpora do not compare (`xenithium-bench compare` warns).

A kind is a directory; files are read in name order. `--corpus DIR` runs the tool on another directory of the
`<kind>/<file>` layout.
