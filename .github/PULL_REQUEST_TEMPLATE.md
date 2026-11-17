## What this changes

<!-- one paragraph: the behaviour before, the behaviour after -->

## Why

<!-- the number, fixture, or failure that motivated it -->

## Checklist

- [ ] A test asserts the new behaviour on a **specific value** (not "no exception")
- [ ] No new hard dependency beyond numpy + Pillow (optional extras are fine)
- [ ] Any approximation is marked `approximate: true` in the recipe
- [ ] `python -m unittest discover -s tests` passes locally
- [ ] `python -m reforge demo` still runs offline with no keys
- [ ] Docs updated if a contract, CLI flag, or dialect changed

## Repro / output

```
# command(s) and the relevant output
```

## Notes for the reviewer

<!-- trade-offs, things you are unsure about, follow-ups deliberately left out -->
