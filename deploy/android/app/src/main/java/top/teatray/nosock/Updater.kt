package top.teatray.nosock

import android.app.Activity
import android.app.AlertDialog
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.webkit.CookieManager
import android.webkit.WebSettings
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

object Updater {
    private const val PREFS = "nosock"

    private fun current(ctx: Context): Long {
        val pi = ctx.packageManager.getPackageInfo(ctx.packageName, 0)
        return pi.longVersionCode
    }

    private fun get(ctx: Context, path: String): HttpURLConnection {
        val c = URL(Poller.BASE + path).openConnection() as HttpURLConnection
        c.connectTimeout = 15000
        c.readTimeout = 60000
        CookieManager.getInstance().getCookie(Poller.BASE)?.let { c.setRequestProperty("Cookie", it) }
        c.setRequestProperty("User-Agent", WebSettings.getDefaultUserAgent(ctx))
        return c
    }

    fun check(act: Activity, manual: Boolean = false) {
        val prefs = act.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val now = System.currentTimeMillis()
        if (!manual && now - prefs.getLong("update_checked", 0) < 3_600_000L) return
        prefs.edit().putLong("update_checked", now).apply()
        Thread {
            try {
                val c = get(act, "api/state")
                if (c.responseCode != 200) return@Thread
                val app = JSONObject(c.inputStream.bufferedReader().readText()).optJSONObject("app") ?: return@Thread
                val code = app.optLong("version_code")
                if (code <= current(act)) return@Thread
                act.runOnUiThread {
                    AlertDialog.Builder(act)
                        .setTitle("새 버전 " + app.optString("version_name"))
                        .setMessage(app.optString("notes", "업데이트가 있습니다."))
                        .setPositiveButton("업데이트") { _, _ -> Thread { install(act, app) }.start() }
                        .setNegativeButton("나중에", null)
                        .show()
                }
            } catch (_: Exception) {
            }
        }.start()
    }

    private fun install(act: Activity, app: JSONObject) {
        try {
            val c = get(act, "api/state?app=apk")
            if (c.responseCode != 200) throw IllegalStateException("download " + c.responseCode)
            val bytes = c.inputStream.readBytes()
            val sha = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
            if (sha != app.optString("sha256")) throw IllegalStateException("hash mismatch")
            val pi = act.packageManager.packageInstaller
            val params = PackageInstaller.SessionParams(PackageInstaller.SessionParams.MODE_FULL_INSTALL)
            params.setAppPackageName(act.packageName)
            val id = pi.createSession(params)
            pi.openSession(id).use { s ->
                s.openWrite("nosock.apk", 0, bytes.size.toLong()).use { it.write(bytes); s.fsync(it) }
                val cb = PendingIntent.getBroadcast(act, id, Intent(act, InstallReceiver::class.java), PendingIntent.FLAG_MUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
                s.commit(cb.intentSender)
            }
        } catch (e: Exception) {
            act.runOnUiThread { AlertDialog.Builder(act).setTitle("업데이트 실패").setMessage(e.message ?: e.javaClass.simpleName).setPositiveButton("확인", null).show() }
        }
    }
}

class InstallReceiver : BroadcastReceiver() {
    override fun onReceive(ctx: Context, intent: Intent) {
        if (intent.getIntExtra(PackageInstaller.EXTRA_STATUS, -999) == PackageInstaller.STATUS_PENDING_USER_ACTION) {
            @Suppress("DEPRECATION")
            val confirm = intent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT) ?: return
            confirm.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            ctx.startActivity(confirm)
        }
    }
}
