/* GitHub Project Pages: /Repo (no slash) breaks relative assets on home only. */
(function () {
  var canonical = document.querySelector('link[rel="canonical"]');
  if (!canonical || !canonical.href) {
    return;
  }
  var target = new URL(canonical.href);
  if (location.pathname === target.pathname) {
    return;
  }
  if (location.pathname + "/" === target.pathname) {
    location.replace(target.pathname + location.search + location.hash);
  }
})();
