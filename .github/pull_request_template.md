## What this changes

<!-- Which workshop, and what changes for someone running it -->

## Checklist

See [CONTRIBUTING.md](../CONTRIBUTING.md). The automated check covers the files; these need a workspace.

- [ ] `python scripts/check_workshops.py` passes
- [ ] Deployed in a fresh workspace (Free Edition if possible) as a non-admin, and ran the notebook top to bottom
- [ ] A second deploy reports 0 changes, and a second copy in the same workspace fails without creating anything
- [ ] Destroy leaves nothing named after the workshop; anything it can't remove is listed under `## Tear down`
- [ ] The README lists every requirement beyond a Free Edition workspace (or there are none)
