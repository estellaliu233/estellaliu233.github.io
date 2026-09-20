# estellaliu233.github.io

Personal site: notes on large-scale ML systems — LLM inference, ads and recommendation.

Built with [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) (Jekyll) and deployed to
GitHub Pages by the `Build and Deploy` GitHub Action on every push to `main`.

## Layout

| Path | What it is |
|---|---|
| `_posts/` | Articles. Home page shows them newest first. |
| `_tabs/adsreco.md` | "Ads & Reco" tab — mind maps of the search / rec / ads literature. |
| `_tabs/about.md` | About me. |
| `_config.yml` | Title, tagline, social links, theme options. |
| `assets/img/` | Images (add `avatar:` in `_config.yml` to show a profile picture). |

## Writing a post

Create `_posts/YYYY-MM-DD-slug.md`:

```yaml
---
title: "Post title"
date: 2026-09-19 10:00:00 -0700
categories: [LLM Inference, KV Cache]   # [parent, child]
tags: [ml-systems, caching]             # lowercase, these are the skill keywords
math: true                              # enable $...$ formulas
mermaid: true                           # enable ```mermaid diagrams
pin: true                               # optional: keep at the top of the home page
---
```

Anything in `_drafts/` is not published.

## Local preview (optional)

Not required — pushing to `main` builds the site on GitHub. To preview locally you need
Ruby 3.x (the system Ruby on macOS is too old):

```bash
brew install ruby
bundle install
bundle exec jekyll serve
```
