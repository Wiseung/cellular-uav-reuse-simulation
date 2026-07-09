# Wiki Source Pages

This folder contains the prepared Markdown pages for the repository wiki.

GitHub does not expose a public API for first-time wiki page creation, and the wiki Git repository is not available until the wiki has been initialized once in the web UI. After that first page exists, these files can be copied into the repository wiki and pushed to `<repo>.wiki.git`.

Typical publish flow after the first wiki page exists:

```powershell
git clone https://github.com/Wiseung/cellular-uav-reuse-simulation.wiki.git
Copy-Item docs/wiki/* cellular-uav-reuse-simulation.wiki/ -Exclude README.md
cd cellular-uav-reuse-simulation.wiki
git add .
git commit -m "Update wiki pages"
git push
```
