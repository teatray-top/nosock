package top.teatray.nosock

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings
import android.view.View
import android.view.WindowInsetsController
import android.webkit.CookieManager
import android.webkit.JavascriptInterface
import android.webkit.WebView
import android.webkit.WebViewClient

class MainActivity : Activity() {
    private lateinit var web: WebView
    private val handler = Handler(Looper.getMainLooper())
    private val loop = object : Runnable {
        override fun run() {
            Thread { try { Poller.poll(applicationContext) } catch (_: Exception) {} }.start()
            handler.postDelayed(this, 30_000L)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        web = WebView(this)
        web.settings.javaScriptEnabled = true
        web.settings.domStorageEnabled = true
        web.webViewClient = WebViewClient()
        web.settings.userAgentString = web.settings.userAgentString + " NoSockApp/" + packageManager.getPackageInfo(packageName, 0).versionName
        web.addJavascriptInterface(Bridge(), "NoSockApp")
        CookieManager.getInstance().setAcceptCookie(true)
        setContentView(web)
        paintBars()
        Poller.channel(this)
        PollJob.schedule(this)
        Alarms.arm(this)
        open(intent)
        askNotifications()
        askBattery()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        open(intent)
    }

    private fun open(i: Intent?) {
        val d = i?.data
        if (d != null && d.scheme == "nosock" && d.host == "login") {
            val k = d.getQueryParameter("k")
            if (!k.isNullOrEmpty()) {
                web.loadUrl(Poller.BASE + "login?k=" + android.net.Uri.encode(k))
                return
            }
        }
        if (web.url == null) web.loadUrl(Poller.BASE)
    }

    @Suppress("DEPRECATION")
    private fun paintBars() {
        val night = (resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES
        val bg = if (night) Color.BLACK else Color.WHITE
        window.statusBarColor = bg
        window.navigationBarColor = bg
        window.decorView.setBackgroundColor(bg)
        web.setBackgroundColor(bg)
        if (Build.VERSION.SDK_INT >= 30) {
            val mask = WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS
            window.insetsController?.setSystemBarsAppearance(if (night) 0 else mask, mask)
        } else {
            var f = window.decorView.systemUiVisibility
            f = if (night) f and View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR.inv() else f or View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR
            if (Build.VERSION.SDK_INT >= 27) f = if (night) f and View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR.inv() else f or View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR
            window.decorView.systemUiVisibility = f
        }
    }

    inner class Bridge {
        @JavascriptInterface
        fun pollNow() {
            Thread { try { Poller.poll(applicationContext) } catch (_: Exception) {} }.start()
        }

        @JavascriptInterface
        fun checkUpdate() {
            runOnUiThread { Updater.check(this@MainActivity, true) }
        }

        @JavascriptInterface
        fun version(): String = packageManager.getPackageInfo(packageName, 0).versionName ?: ""
    }

    private fun askBattery() {
        val pm = getSystemService(PowerManager::class.java)
        val p = getSharedPreferences("nosock", MODE_PRIVATE)
        if (!pm.isIgnoringBatteryOptimizations(packageName) && !p.getBoolean("battery_asked", false)) {
            p.edit().putBoolean("battery_asked", true).apply()
            try {
                startActivity(Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, Uri.parse("package:" + packageName)))
            } catch (_: Exception) {
            }
        }
    }

    private fun askNotifications() {
        if (Build.VERSION.SDK_INT >= 33 && checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        } else {
            welcome()
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) welcome()
    }

    private fun welcome() {
        val p = getSharedPreferences("nosock", MODE_PRIVATE)
        if (!p.getBoolean("welcomed", false)) {
            Poller.notify(this, 1, "NoSock 알림 설정 완료", "이 알림이 보이면 앱 알림이 정상입니다. 환전 요청이 생기면 15분 안에 알려 드립니다.")
            p.edit().putBoolean("welcomed", true).apply()
        }
    }

    override fun onResume() {
        super.onResume()
        handler.post(loop)
        Updater.check(this)
    }

    override fun onPause() {
        super.onPause()
        handler.removeCallbacks(loop)
        CookieManager.getInstance().flush()
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        if (web.canGoBack()) web.goBack() else super.onBackPressed()
    }
}
