# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

This is a new project repository. No application code has been added yet.

## Multi-Agent Workflow System

This repository uses a `.dagent` multi-agent orchestration system with specialized agents:

- **PM Agent**: Manages feature specs and tasks. Always updates the spec before creating tasks. Uses DAG-based task management with automatic dependency handling.
- **Developer Agent**: Implements tasks in isolated git worktrees. Focuses only on assigned task scope, not broader feature work.
- **QA Agent**: Reviews code changes against task specifications only (not broader feature). Automatic fail if code references `.dagent/` paths.
- **Merge Agent**: Handles branch integration and merge conflict resolution.

## Key Conventions

- Tasks work in isolated git worktrees (`.dagent-worktrees/`)
- Attachments in `.dagent/attachments/` are git-ignored - never reference these paths in code; copy assets to project folders instead
- Feature specs are the source of truth; requirements need matching acceptance criteria
- Task dependencies form a DAG (directed acyclic graph) - cycles are prevented
