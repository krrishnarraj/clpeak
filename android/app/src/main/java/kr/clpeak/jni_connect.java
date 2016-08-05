package kr.clpeak;

import android.app.Activity;
import android.os.AsyncTask;
import android.widget.Button;
import android.widget.TextView;

public class jni_connect extends AsyncTask<Integer, String, Integer>
{
    Activity rootActivity;
    TextView text_view;
    Button runButton;

    public native int launchClpeak(int argc, String[] argv);

    jni_connect(Activity _activity)
    {
        rootActivity = _activity;
        text_view = (TextView) rootActivity.findViewById(R.id.clpeak_result_textview);
        runButton = (Button) rootActivity.findViewById(R.id.run_button);
    }

    @Override
    protected void onPreExecute()
    {
        runButton.setEnabled(false);
        text_view.setText("");
        text_view.setKeepScreenOn(true);
    }

    @Override
    protected Integer doInBackground(Integer... arg0)
    {
        String options[] = {"clpeak", "--all-tests"};

        return launchClpeak(options.length, options);
    }

    @Override
    protected void onPostExecute(Integer result)
    {
        if(result != 0)
        {
            text_view.append("\nSomething went wrong\n1. OpenCL platform may not be present. In that case install pocl from App Store\n2. clpeak exited with some error\n");
        }
        runButton.setEnabled(true);
        text_view.setKeepScreenOn(false);
    }

    @Override
    protected void onCancelled()
    {
        text_view.append("\nclpeak exited abnormally\n");
        runButton.setEnabled(true);
        text_view.setKeepScreenOn(false);
    }

    @Override
    protected void onProgressUpdate(String... str)
    {
        text_view.append(str[0]);
    }

    public void print_callback_from_c(String str)
    {
        publishProgress(str);
    }
}