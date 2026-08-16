// Fills in the download section from the newest published release.
//
// Done at page load rather than baked into the HTML so a new tag needs no site
// edit: the asset names carry the version, the platform and the optional
// vendor-SDK variant, and grouping them by platform is all the page needs.  If
// the request fails (offline, or the unauthenticated API rate limit is hit)
// the static "all releases" link that is already in the markup stays as-is.
(function () {
  'use strict';

  var script = document.currentScript;
  var repo = (script && script.dataset.repo) || 'krrishnarraj/clpeak';
  var mount = document.getElementById('release-list');
  if (!mount) return;

  var PLATFORMS = [
    { id: 'macos', label: 'macOS', match: /macos/i },
    { id: 'linux', label: 'Linux', match: /linux/i },
    { id: 'windows', label: 'Windows', match: /windows/i }
  ];

  function prettySize(bytes) {
    if (!bytes) return '';
    var mb = bytes / (1024 * 1024);
    return mb >= 1 ? mb.toFixed(1) + ' MB' : Math.round(bytes / 1024) + ' KB';
  }

  // "clpeak-2.0.19-linux-x86_64-cuda.zip" -> "x86_64 · cuda"
  function prettyName(name, tag) {
    var stem = name
      .replace(/^clpeak-/, '')
      .replace(/\.(zip|dmg|tar\.gz|apk)$/i, '')
      .replace(tag + '-', '');
    PLATFORMS.forEach(function (p) {
      stem = stem.replace(new RegExp('^' + p.id + '-?', 'i'), '');
    });
    stem = stem.replace(/-/g, ' · ');
    if (/\.dmg$/i.test(name)) stem = stem ? stem + ' · dmg' : 'dmg';
    return stem || name;
  }

  function render(release) {
    var frag = document.createDocumentFragment();
    var used = 0;

    PLATFORMS.forEach(function (platform) {
      var assets = release.assets.filter(function (a) {
        return platform.match.test(a.name);
      });
      if (!assets.length) return;
      used += assets.length;

      // The macOS .dmg is the one-click GUI install, so float it first.
      assets.sort(function (a, b) {
        var da = /\.dmg$/i.test(a.name) ? 0 : 1;
        var db = /\.dmg$/i.test(b.name) ? 0 : 1;
        return da - db || a.name.localeCompare(b.name);
      });

      var group = document.createElement('div');
      group.className = 'rel-group';

      var h = document.createElement('h3');
      h.textContent = platform.label;
      group.appendChild(h);

      var row = document.createElement('div');
      row.className = 'rel-assets';
      assets.forEach(function (a) {
        var link = document.createElement('a');
        link.href = a.browser_download_url;
        link.appendChild(
          document.createTextNode(prettyName(a.name, release.tag_name)));
        var size = document.createElement('span');
        size.className = 'size';
        size.textContent = prettySize(a.size);
        link.appendChild(size);
        row.appendChild(link);
      });
      group.appendChild(row);
      frag.appendChild(group);
    });

    if (!used) return;

    var note = document.createElement('p');
    note.className = 'rel-note';
    note.textContent = 'Release ' + release.tag_name + ', published ' +
      new Date(release.published_at).toISOString().slice(0, 10) + '.';
    frag.insertBefore(note, frag.firstChild);

    mount.textContent = '';
    mount.appendChild(frag);
  }

  fetch('https://api.github.com/repos/' + repo + '/releases/latest', {
    headers: { Accept: 'application/vnd.github+json' }
  })
    .then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(render)
    .catch(function () { /* keep the static fallback already in the markup */ });
})();
