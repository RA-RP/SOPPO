# Public Release Checklist

- [ ] Revoke the Hugging Face token that appeared in the original repository.
- [ ] Confirm that no token-shaped strings or private URLs remain.
- [ ] Replace `Anonymous Authors` in `CITATION.cff` with the final author list.
- [ ] Replace the conservative `LICENSE` notice with the authors' chosen license.
- [ ] Confirm that paper anonymity is no longer required, or retain anonymous metadata.
- [ ] Review `mypaper/` notes for personal information, unpublished reviewer text, and long quotations.
- [ ] Confirm that every file in `paper_artifacts/` may be redistributed.
- [ ] Publish checkpoints or raw artifacts only through a dedicated artifact host with checksums.
- [ ] Run unit tests, rebuild all five figures, and compile the English paper and supplement.
- [ ] Verify that no tracked file exceeds the intended repository size policy.

