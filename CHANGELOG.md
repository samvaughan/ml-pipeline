# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-13

### Added
- `FeatureEngineer`, `DataFinaliser`, and `ModelTrainingRunner` pipeline stages
- `BaseTransformer` and `BaseTrainer` abstract base classes
- `Registry` for name-based component registration
- YAML config support with round-trip serialization via `from_cfg` / `to_config`
- `autodiscover` utility for triggering `@register` decorators across a package
