package kr.clpeak;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.AdapterView;
import android.widget.AdapterView.OnItemSelectedListener;
import android.widget.ArrayAdapter;
import android.widget.Spinner;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    public native void setenv(String key, String value);

    static {
        System.loadLibrary("clpeak");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.result_display);

        populatePlatformSpinner();

        findViewById(R.id.run_button).setOnClickListener(new OnClickListener() {
            @Override
            public void onClick(View view) {
                kr.clpeak.jni_connect clp = new kr.clpeak.jni_connect(MainActivity.this);
                clp.execute();
            }
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.main_menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        switch (item.getItemId()) {
            case R.id.menu_about:
                Intent intent = new Intent(MainActivity.this, kr.clpeak.AboutActivity.class);
                startActivity(intent);
        }

        return true;
    }

    public void populatePlatformSpinner() {

        final Spinner spinnerPlatform = (Spinner) findViewById(R.id.spinner_platform_list);

        final List<String> libopenclSoPaths = new ArrayList<String>(Arrays.asList(
                "/vendor/lib64/libOpenCL.so",
                "/system/lib64/libOpenCL.so",
                "/system/vendor/lib64/libOpenCL.so",
                "/system/lib/libOpenCL.so",
                "/system/vendor/lib/libOpenCL.so",
                "/system/vendor/lib64/egl/libGLES_mali.so",
                "/system/vendor/lib/egl/libGLES_mali.so",
                "/system/vendor/lib/libPVROCL.so",
                "/data/data/org.pocl.libs/files/lib/libpocl.so",
                "libOpenCL.so"
        ));

        final List<String> libopenclPlatforms = new ArrayList<String>(Arrays.asList(
                "vendor lib64",
                "system lib64",
                "system vendor lib64",
                "system lib",
                "system vendor lib",
                "mali",
                "powerVR",
                "pocl",
                "default"
        ));

        // Don't search for "default" & "pocl"
        for (int i = (libopenclSoPaths.size() - 3); i >= 0; i--) {
            if (!(new File(libopenclSoPaths.get(i)).exists())) {
                libopenclSoPaths.remove(i);
                libopenclPlatforms.remove(i);
            }
        }

        ArrayAdapter<String> dataAdapter = new ArrayAdapter<String>(this,
                android.R.layout.simple_spinner_item, libopenclPlatforms);
        dataAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinnerPlatform.setAdapter(dataAdapter);

        spinnerPlatform.setOnItemSelectedListener(new OnItemSelectedListener() {
              @Override
              public void onItemSelected(AdapterView<?> arg0, View arg1, int arg2, long arg3) {
                  if (libopenclPlatforms.get(arg2).equals("pocl")) {
                      if (!(new File(libopenclSoPaths.get(arg2)).exists())) {
                          AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);

                          builder.setMessage("pocl installation not found\ninstall it from playstore?");

                          builder.setPositiveButton("go", new DialogInterface.OnClickListener() {
                              public void onClick(DialogInterface dialog, int id) {
                                  Uri uri = Uri.parse("market://details?id=org.pocl.libs");
                                  Intent myAppLinkToMarket = new Intent(Intent.ACTION_VIEW, uri);
                                  startActivity(myAppLinkToMarket);
                              }
                          });

                          builder.setNegativeButton("leave it", new DialogInterface.OnClickListener() {
                              public void onClick(DialogInterface dialog, int id) {
                              }
                          });

                          builder.show();
                          spinnerPlatform.setSelection(0);
                          return;
                      }
                  }
                  setenv("LIBOPENCL_SO_PATH", libopenclSoPaths.get(arg2));
              }

              @Override
              public void onNothingSelected(AdapterView<?> arg0) {
              }
          }
        );
    }
}
