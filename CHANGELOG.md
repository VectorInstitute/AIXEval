# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/).

## [Unreleased]

### Added

- Public methodology, interpreting-results, limitations, and
  privacy documentation.
- Fixtures attestation and labels on intentional negative examples.
- `NOTICE`, `SECURITY.md`, `CITATION.cff`, and this changelog.
- Streamlit `gatherUsageStats = false`.
- Secret scan (gitleaks CLI) in CI alongside pip-audit. The GitHub
  Action is not used: it requires a paid org license.

## [0.1.1] — 2026-11-11

First public-facing version of the `aixeval` package (post-hoc
scorecard, CLI, Streamlit UI, five judge providers including
self-hosted vLLM).
