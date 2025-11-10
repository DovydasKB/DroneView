package com.example.droneview;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;

public class MainActivity extends AppCompatActivity {

    private WebView webView;
    private EditText editText;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        webView = findViewById(R.id.webView);
        editText = findViewById(R.id.edit_text);
        Button goButton = findViewById(R.id.go_button);

        webView.setWebViewClient(new WebViewClient());

        WebSettings webSettings = webView.getSettings();
        webSettings.setJavaScriptEnabled(true);
        webSettings.setCacheMode(WebSettings.LOAD_CACHE_ELSE_NETWORK);
        webSettings.setDomStorageEnabled(true);
        webSettings.setAllowFileAccess(true);
        webSettings.setLoadWithOverviewMode(true);
        webSettings.setUseWideViewPort(true);

        // Load initial URL from hint
        if (editText.getHint() != null) {
            String initialIp = editText.getHint().toString();
            if (!initialIp.isEmpty()) {
                webView.loadUrl("http://" + initialIp);
            }
        }

        goButton.setOnClickListener(v -> loadUrlFromEditText());
    }

    private void loadUrlFromEditText() {
        String ipAddress = editText.getText().toString();
        if (!ipAddress.isEmpty()) {
            String streamUrl = "http://" + ipAddress;
            webView.loadUrl(streamUrl);
        } else if (editText.getHint() != null && !editText.getHint().toString().isEmpty()) {
            // Fallback to hint if input is empty
            String streamUrl = "http://" + editText.getHint().toString();
            webView.loadUrl(streamUrl);
        }
    }
}
