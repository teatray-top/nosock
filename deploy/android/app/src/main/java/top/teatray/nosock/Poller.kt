package top.teatray.nosock

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.webkit.CookieManager
import android.webkit.WebSettings
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

object Poller {
    const val BASE = "https://teatray.top/nosock/"
    const val CHANNEL = "fx"
    private const val PREFS = "nosock"

    fun channel(ctx: Context) {
        val nm = ctx.getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL) == null) {
            val ch = NotificationChannel(CHANNEL, "환전 요청과 알림", NotificationManager.IMPORTANCE_HIGH)
            ch.description = "환전 요청, 테스트 알림"
            nm.createNotificationChannel(ch)
        }
    }

    fun notify(ctx: Context, id: Int, title: String, body: String) {
        channel(ctx)
        val open = PendingIntent.getActivity(ctx, 0, Intent(ctx, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK), PendingIntent.FLAG_IMMUTABLE)
        val n = android.app.Notification.Builder(ctx, CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_sock)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(android.app.Notification.BigTextStyle().bigText(body))
            .setContentIntent(open)
            .setAutoCancel(true)
            .build()
        ctx.getSystemService(NotificationManager::class.java).notify(id, n)
    }

    fun poll(ctx: Context): String {
        val cookie = CookieManager.getInstance().getCookie(BASE) ?: return "no-login"
        val c = URL(BASE + "api/state").openConnection() as HttpURLConnection
        c.connectTimeout = 15000
        c.readTimeout = 20000
        c.setRequestProperty("Cookie", cookie)
        c.setRequestProperty("User-Agent", WebSettings.getDefaultUserAgent(ctx))
        try {
            if (c.responseCode != 200) return "http-" + c.responseCode
            val s = JSONObject(c.inputStream.bufferedReader().readText())
            val prefs = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val seen = HashSet(prefs.getStringSet("seen", emptySet()) ?: emptySet())
            var shown = 0
            val fx = s.optJSONArray("fx")
            if (fx != null) {
                for (i in 0 until fx.length()) {
                    val f = fx.getJSONObject(i)
                    val id = f.optString("id")
                    if (id.isEmpty() || seen.contains(id) || !f.isNull("done")) continue
                    val arrow = if (f.optString("direction") == "KRW->USD") "원화 → 달러" else "달러 → 원화"
                    val given = f.optString("body")
                    val body = if (given.isNotEmpty() && given != "null") given else "%s %,d원 (약 $%,.2f).".format(arrow, f.optLong("amount_won"), f.optDouble("amount_usd"))
                    notify(ctx, id.hashCode(), "환전 요청", body)
                    seen.add(id)
                    shown++
                }
            }
            val al = s.optJSONArray("alerts")
            if (al != null) {
                for (i in 0 until al.length()) {
                    val x = al.getJSONObject(i)
                    val id = "al-" + x.optString("id")
                    if (seen.contains(id)) continue
                    notify(ctx, id.hashCode(), x.optString("title"), x.optString("body"))
                    seen.add(id)
                    shown++
                }
            }
            val ping = s.optJSONObject("ping")
            if (ping != null) {
                val pid = "ping-" + ping.optString("id")
                if (!seen.contains(pid)) {
                    notify(ctx, pid.hashCode(), "NoSock 알림 테스트", "앱 알림이 정상적으로 도착했습니다. (" + ping.optString("at") + ")")
                    seen.add(pid)
                    shown++
                }
            }
            prefs.edit().putStringSet("seen", seen).putLong("last_ok", System.currentTimeMillis()).apply()
            return "ok:" + shown
        } finally {
            c.disconnect()
        }
    }
}
