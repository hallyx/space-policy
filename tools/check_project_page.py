"""Validate project-page links, citations, assets and paper result arithmetic."""
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import re

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.ids, self.sources = [], set(), []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            assert values["id"] not in self.ids, "Duplicate HTML id"
            self.ids.add(values["id"])
        for key in ("href", "src", "poster"):
            if key in values:
                self.links.append(values[key])
        if tag == "source":
            self.sources.append(values)
        if tag == "img":
            assert values.get("alt"), "Image needs descriptive alt text"
        if tag == "video":
            assert "controls" in values and "playsinline" in values
            assert "autoplay" not in values, "Demo must not autoplay with sound"


def main():
    html = (SITE / "index.html").read_text()
    page = Page()
    page.feed(html)
    for link in page.links:
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc:
            assert parsed.scheme in ("https", "mailto"), link
            continue
        if parsed.path:
            assert not parsed.path.startswith("/"), "Use project-relative URLs"
            path = (SITE / unquote(parsed.path)).resolve()
            assert path.is_relative_to(SITE.resolve()) and path.is_file(), link
        elif parsed.fragment:
            assert parsed.fragment in page.ids, link
    citation = (ROOT / "CITATION.bib").read_text().strip()
    assert (SITE / "citation.bib").read_text().strip() == citation
    inline = re.search(r'<code id="citation-code">(.*?)</code>', html, re.S)
    assert inline and unescape(inline.group(1)).strip() == citation
    assert page.sources == [{"src": "assets/space-policy-demo.mp4", "type": "video/mp4"}]
    assert (SITE / ".nojekyll").is_file()
    assert 0 < (SITE / "assets/space-policy-demo.mp4").stat().st_size < 100 * 1024 * 1024
    # Source: Table I of the paper. The HTML is intentionally static.
    reported = {
        "ACT": ([1, 0, 0], 0.33, 70.8),
        "ACT + GJM": ([54, 45, 29], 42.67, 13.4),
        "DP + GJM": ([45, 36, 26], 35.67, 15.4),
        "SpacePolicy": ([68, 67, 55], 63.33, 7.8),
    }
    rows = re.findall(r'<tr(?: class="ours")?><th scope="row">(.*?)</tr>', html, re.S)
    assert len(rows) == 4
    for row, (name, (targets, mean, distance)) in zip(rows, reported.items()):
        assert row.startswith(name)
        values = [float(x) for x in re.findall(r'<td>([\d.]+)</td>', row)]
        assert values == [*targets, mean, distance], (name, values)
        assert abs(sum(targets) / 3 - mean) < 0.005
    assert abs((63.33 - 42.67) - 20.66) < 1e-8
    print(f"Project page verified: {len(page.links)} links/assets, 11-author citation, 4 comparison rows")


if __name__ == "__main__":
    main()
