
const CACHE_NAME = "safehire-cache-v1";

self.addEventListener("install", (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.addAll([
                "./",
                "./index.html",
                "./style.css",
                "./script.js",
                "./manifest.json",
                "./assets/safehire-icon.png",
                "./assets/safehire-logo.png"
            ]);
        })
    );

    self.skipWaiting();
});

self.addEventListener("activate", (event) => {
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames
                    .filter((name) => name !== CACHE_NAME)
                    .map((name) => caches.delete(name))
            );
        })
    );

    self.clients.claim();
});
