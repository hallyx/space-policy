'use strict';
const button = document.getElementById('copy-citation');
const status = document.getElementById('copy-status');
button.addEventListener('click', async () => {
  const citation = document.getElementById('citation-code').textContent.trim() + '\n';
  try {
    await navigator.clipboard.writeText(citation);
    status.textContent = 'BibTeX copied.';
  } catch {
    status.textContent = 'Copy is unavailable. Use the .bib download below.';
  }
});
