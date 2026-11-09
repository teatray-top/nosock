const VERSION = "sw-5";
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("push", (e) => {
  let d = { title: "NoSock", body: "", url: "/nosock/" };
  try { d = Object.assign(d, e.data.json()); } catch (_) {}
  e.waitUntil(Promise.all([
    self.registration.showNotification(d.title, { body: d.body, icon: "/nosock/icon-192.png?v=2", data: { url: d.url }, tag: d.id || "nosock", renotify: true }),
    fetch("/nosock/api/state?ack=" + encodeURIComponent(d.id || "push"), { credentials: "same-origin" }).catch(() => null),
    self.clients.matchAll({ includeUncontrolled: true, type: "window" }).then((cs) => cs.forEach((c) => c.postMessage({ type: "push", id: d.id || "push" }))),
  ]));
});
self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  const url = (e.notification.data && e.notification.data.url) || "/nosock/";
  e.waitUntil(self.clients.matchAll({ type: "window" }).then((cs) => {
    for (const c of cs) if (c.url.includes("/nosock/")) { c.navigate(url); return c.focus(); }
    return self.clients.openWindow(url);
  }));
});
