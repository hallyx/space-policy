# GitHub Pages deployment

The project page is a buildless static website served from `docs/`:

- `index.html`: paper information, video, method, results, and citation.
- `style.css`: responsive styling; no external fonts or framework.
- `app.js`: BibTeX copy control.
- `assets/`: the demonstration video, poster, and paper method figure.
- `citation.bib`: a copy of the repository citation for download.

## Preview

From the repository root:

```bash
python3 tools/check_project_page.py
python3 -m http.server 8000 --directory docs
```

Open `http://localhost:8000`. All resources use relative URLs so the same page
works at a GitHub project path such as `/space-policy/`.

## Deploy from a branch

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
3. Select branch **main**, folder **/docs**, and click **Save**.
4. Wait for GitHub's Pages build, then open the URL shown in the Pages settings.
   A conventional project URL is `https://hallyx.github.io/space-policy/`.

The page is plain static HTML/CSS/JavaScript. No package install, custom build,
secrets, or repository Actions workflow is needed. `docs/.nojekyll` keeps Jekyll
processing disabled. After Pages is enabled, commits to the publishing branch
are handled by GitHub's built-in Pages deployment.

This change prepares the source files but does not enable Pages or change
repository visibility. The expected URL is not claimed to be live. Pages
availability depends on the account plan, repository visibility and organization
settings.

Official configuration reference:
https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

## Content and sources

The method image is Figure 3 from the linked paper. The comparison table and
highlights reproduce Table I, and the component-study sentence uses Table III.
Metric definitions follow Section IV-A. Numbers are paper-reported results,
not fresh evaluations of the released code. Asset provenance is documented in
`assets/README.md`.

To change the page, edit the HTML/CSS files and run the checker before pushing.
If the root `CITATION.bib` changes, update both `docs/citation.bib` and the
inline BibTeX block in `index.html`; the checker verifies that they agree.
