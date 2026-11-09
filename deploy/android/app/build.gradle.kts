import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val ksProps = Properties().apply {
    val f = file(System.getenv("NOSOCK_KEYSTORE_PROPS") ?: "../keystore.properties")
    if (f.exists()) f.inputStream().use { load(it) }
}

android {
    namespace = "top.teatray.nosock"
    compileSdk = 34

    defaultConfig {
        applicationId = "top.teatray.nosock"
        minSdk = 26
        targetSdk = 34
        versionCode = 6
        versionName = "0.1.5"
    }

    signingConfigs {
        create("release") {
            storeFile = file(ksProps.getProperty("storeFile", "missing.jks"))
            storePassword = ksProps.getProperty("storePassword")
            keyAlias = ksProps.getProperty("keyAlias")
            keyPassword = ksProps.getProperty("keyPassword")
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}
