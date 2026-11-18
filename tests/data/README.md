# Test data

`contract/` holds days written by marketrec 0.1.0 (the build is named in each
`manifest.json`), so that the reader is tested against the recorder's real output rather
than against this repository's idea of it:

| day | made with | holds |
|-----|-----------|-------|
| `gap` | `marketrec convert --input tests/fixtures/gap.jsonl` | a gap at 305 ms and the resync at 360 ms |
| `reconnect` | `marketrec convert --input tests/fixtures/reconnect.jsonl` | a lost connection and the sync on the next one |
| `short` | `marketrec convert` on the first 3 s of a synthetic recording | every table with rows, trades included |

The `tests/fixtures/*.jsonl` files are in the marketrec repository. These files must not be
edited by hand: a change to the contract means writing them again with marketrec.
