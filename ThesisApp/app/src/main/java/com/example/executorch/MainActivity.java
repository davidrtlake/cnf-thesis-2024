package com.example.executorch;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.os.SystemClock;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import java.io.IOException;

import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {

  private static final int REQUEST_CODE_SELECT_IMAGE = 100;
  private static final int PERMISSION_REQUEST_CODE = 200;

  private static final String TAG = "ImageSegmentation";

  // ImageViews and Buttons for body sections
  private ImageView imageFrontTorso;
  private Button buttonUploadFrontTorso;
  private Button buttonProcessFrontTorso;
  private ImageView outputImageFrontTorso;
  private TextView textIoUFrontTorso;
  private TextView textTimerFrontTorso;

  private ImageView imageBackTorso;
  private Button buttonUploadBackTorso;
  private Button buttonProcessBackTorso;
  private ImageView outputImageBackTorso;
  private TextView textIoUBackTorso;
  private TextView textTimerBackTorso;

  private ImageView imageChestToHead;
  private Button buttonUploadChestToHead;
  private Button buttonProcessChestToHead;
  private ImageView outputImageChestToHead;
  private TextView textIoUChestToHead;
  private TextView textTimerChestToHead;

  private ImageView imageArms;
  private Button buttonUploadArms;
  private Button buttonProcessArms;
  private ImageView outputImageArms;
  private TextView textIoUArms;
  private TextView textTimerArms;

  private ImageView imageLegs;
  private Button buttonUploadLegs;
  private Button buttonProcessLegs;
  private ImageView outputImageLegs;
  private TextView textIoULegs;
  private TextView textTimerLegs;

  private TextView averageSeverityTextView;

  private String selectedBodyPart;

  // Bitmaps for the selected images
  private Bitmap bitmapFrontTorso;
  private Bitmap bitmapBackTorso;
  private Bitmap bitmapChestToHead;
  private Bitmap bitmapArms;
  private Bitmap bitmapLegs;

  private Module module;
  private Module skinModule;

  private float[] mean = new float[]{0.485f, 0.456f, 0.406f};
  private float[] std = new float[]{0.229f, 0.224f, 0.225f};

  // ExecutorService for background tasks
  private ExecutorService executorService;

  // Map to store severity scores
  private Map<String, Float> severityScores = new HashMap<>();

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main); // Ensure your layout file is named activity_main.xml

    // Initialize ExecutorService
    executorService = Executors.newFixedThreadPool(5);

    // Load the model
    try {
      module = Module.load(assetFilePath(this, "tumour_model_781_570_cpu.pte"));
      skinModule = Module.load(assetFilePath(this, "new_skin_model_781_570_cpu.pte"));
    } catch (IOException e) {
      Log.e(TAG, "Error loading model", e);
      finish();
    }

    // Initialize average severity score TextView
    averageSeverityTextView = findViewById(R.id.averageSeverityTextView);

    // Initialize views for Front Torso
    imageFrontTorso = findViewById(R.id.imageFrontTorso);
    buttonUploadFrontTorso = findViewById(R.id.buttonUploadFrontTorso);
    buttonProcessFrontTorso = findViewById(R.id.buttonProcessFrontTorso);
    outputImageFrontTorso = findViewById(R.id.outputImageFrontTorso);
    textIoUFrontTorso = findViewById(R.id.textIoUFrontTorso);
    textTimerFrontTorso = findViewById(R.id.textTimerFrontTorso);

    buttonUploadFrontTorso.setOnClickListener(v -> selectImage("frontTorso"));
    buttonProcessFrontTorso.setOnClickListener(v -> processImage("frontTorso"));

    // Initialize views for Back Torso
    imageBackTorso = findViewById(R.id.imageBackTorso);
    buttonUploadBackTorso = findViewById(R.id.buttonUploadBackTorso);
    buttonProcessBackTorso = findViewById(R.id.buttonProcessBackTorso);
    outputImageBackTorso = findViewById(R.id.outputImageBackTorso);
    textIoUBackTorso = findViewById(R.id.textIoUBackTorso);
    textTimerBackTorso = findViewById(R.id.textTimerBackTorso);

    buttonUploadBackTorso.setOnClickListener(v -> selectImage("backTorso"));
    buttonProcessBackTorso.setOnClickListener(v -> processImage("backTorso"));

    // Initialize views for Chest to Head
    imageChestToHead = findViewById(R.id.imageChestToHead);
    buttonUploadChestToHead = findViewById(R.id.buttonUploadChestToHead);
    buttonProcessChestToHead = findViewById(R.id.buttonProcessChestToHead);
    outputImageChestToHead = findViewById(R.id.outputImageChestToHead);
    textIoUChestToHead = findViewById(R.id.textIoUChestToHead);
    textTimerChestToHead = findViewById(R.id.textTimerChestToHead);

    buttonUploadChestToHead.setOnClickListener(v -> selectImage("chestToHead"));
    buttonProcessChestToHead.setOnClickListener(v -> processImage("chestToHead"));

    // Initialize views for Arms
    imageArms = findViewById(R.id.imageArms);
    buttonUploadArms = findViewById(R.id.buttonUploadArms);
    buttonProcessArms = findViewById(R.id.buttonProcessArms);
    outputImageArms = findViewById(R.id.outputImageArms);
    textIoUArms = findViewById(R.id.textIoUArms);
    textTimerArms = findViewById(R.id.textTimerArms);

    buttonUploadArms.setOnClickListener(v -> selectImage("arms"));
    buttonProcessArms.setOnClickListener(v -> processImage("arms"));

    // Initialize views for Legs
    imageLegs = findViewById(R.id.imageLegs);
    buttonUploadLegs = findViewById(R.id.buttonUploadLegs);
    buttonProcessLegs = findViewById(R.id.buttonProcessLegs);
    outputImageLegs = findViewById(R.id.outputImageLegs);
    textIoULegs = findViewById(R.id.textIoULegs);
    textTimerLegs = findViewById(R.id.textTimerLegs);

    buttonUploadLegs.setOnClickListener(v -> selectImage("legs"));
    buttonProcessLegs.setOnClickListener(v -> processImage("legs"));
  }

  private void selectImage(String bodyPart) {
    selectedBodyPart = bodyPart;
    // Show options to choose image from gallery
    Intent intent = new Intent(Intent.ACTION_PICK);
    intent.setType("image/*");
    startActivityForResult(intent, REQUEST_CODE_SELECT_IMAGE);
  }

  // Handle the result of image selection
  @Override
  protected void onActivityResult(int requestCode, int resultCode, Intent data) {
    if (resultCode == Activity.RESULT_OK) {
      if (requestCode == REQUEST_CODE_SELECT_IMAGE) {
        Uri selectedImageUri = data.getData();
        try {
          Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
          setBodyPartImage(bitmap, selectedBodyPart);
        } catch (IOException e) {
          Log.e(TAG, "Error loading image", e);
        }
      }
    }
    super.onActivityResult(requestCode, resultCode, data);
  }

  private void setBodyPartImage(Bitmap bitmap, String bodyPart) {
    switch (bodyPart) {
      case "frontTorso":
        bitmapFrontTorso = bitmap;
        imageFrontTorso.setImageBitmap(bitmap);
        break;
      case "backTorso":
        bitmapBackTorso = bitmap;
        imageBackTorso.setImageBitmap(bitmap);
        break;
      case "chestToHead":
        bitmapChestToHead = bitmap;
        imageChestToHead.setImageBitmap(bitmap);
        break;
      case "arms":
        bitmapArms = bitmap;
        imageArms.setImageBitmap(bitmap);
        break;
      case "legs":
        bitmapLegs = bitmap;
        imageLegs.setImageBitmap(bitmap);
        break;
    }
  }

  private void processImage(String bodyPart) {
    Bitmap inputBitmap = null;
    ImageView outputImageView = null;
    TextView textIoU = null;
    TextView textTimer = null;
    Button buttonProcess = null;

    switch (bodyPart) {
      case "frontTorso":
        if (bitmapFrontTorso == null) {
          Toast.makeText(this, "Please select an image for Front Torso", Toast.LENGTH_SHORT).show();
          return;
        }
        inputBitmap = bitmapFrontTorso;
        outputImageView = outputImageFrontTorso;
        textIoU = textIoUFrontTorso;
        textTimer = textTimerFrontTorso;
        buttonProcess = buttonProcessFrontTorso;
        break;
      case "backTorso":
        if (bitmapBackTorso == null) {
          Toast.makeText(this, "Please select an image for Back Torso", Toast.LENGTH_SHORT).show();
          return;
        }
        inputBitmap = bitmapBackTorso;
        outputImageView = outputImageBackTorso;
        textIoU = textIoUBackTorso;
        textTimer = textTimerBackTorso;
        buttonProcess = buttonProcessBackTorso;
        break;
      case "chestToHead":
        if (bitmapChestToHead == null) {
          Toast.makeText(this, "Please select an image for Chest to Head", Toast.LENGTH_SHORT).show();
          return;
        }
        inputBitmap = bitmapChestToHead;
        outputImageView = outputImageChestToHead;
        textIoU = textIoUChestToHead;
        textTimer = textTimerChestToHead;
        buttonProcess = buttonProcessChestToHead;
        break;
      case "arms":
        if (bitmapArms == null) {
          Toast.makeText(this, "Please select an image for Arms", Toast.LENGTH_SHORT).show();
          return;
        }
        inputBitmap = bitmapArms;
        outputImageView = outputImageArms;
        textIoU = textIoUArms;
        textTimer = textTimerArms;
        buttonProcess = buttonProcessArms;
        break;
      case "legs":
        if (bitmapLegs == null) {
          Toast.makeText(this, "Please select an image for Legs", Toast.LENGTH_SHORT).show();
          return;
        }
        inputBitmap = bitmapLegs;
        outputImageView = outputImageLegs;
        textIoU = textIoULegs;
        textTimer = textTimerLegs;
        buttonProcess = buttonProcessLegs;
        break;
    }

    // Disable the button to prevent multiple clicks
    buttonProcess.setEnabled(false);
    buttonProcess.setText("Processing...");

    // Reset timer text
    textTimer.setText("Processing...");

    // Record start time
    long startTime = SystemClock.elapsedRealtime();

    // Run the model on the selected image in a background thread
    final Bitmap finalInputBitmap = inputBitmap;
    final ImageView finalOutputImageView = outputImageView;
    final TextView finalTextIoU = textIoU;
    final TextView finalTextTimer = textTimer;
    final Button finalButtonProcess = buttonProcess;
    final String finalBodyPart = bodyPart;

    executorService.submit(() -> {

      Bitmap inputBitmapInner = finalInputBitmap;

      // Handle landscape images
      boolean rotated = false;
      if (inputBitmapInner.getWidth() > inputBitmapInner.getHeight()) {
        inputBitmapInner = rotateBitmap(inputBitmapInner, 90);
        rotated = true;
      }

      // Preprocess the image (resize, normalize)
      Bitmap resizedBitmap = Bitmap.createScaledBitmap(inputBitmapInner, 570, 781, true);
      Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(resizedBitmap, mean, std);

      // Update UI to indicate skin segmentation finished
      runOnUiThread(() -> {
        finalTextTimer.setText("Starting skin segmentation...");
      });

      // Run skin segmentation
      Tensor skinOutputTensor = skinModule.forward(EValue.from(inputTensor))[0].toTensor();

      float skinIOU = computeIoU(skinOutputTensor);

      // Create skin mask bitmap
      Bitmap skinMaskBitmap = processOutput(skinOutputTensor, resizedBitmap.getWidth(), resizedBitmap.getHeight(), resizedBitmap);

      // Update UI to indicate skin segmentation finished
      runOnUiThread(() -> {
        finalTextTimer.setText("Skin segmenting finished... Starting tumour segmentation...");
      });

      // Apply skin mask to the resized image
      Bitmap maskedBitmap = applyMaskToBitmap(resizedBitmap, skinMaskBitmap);

      // Prepare input tensor for lesion model
      Tensor lesionInputTensor = TensorImageUtils.bitmapToFloat32Tensor(maskedBitmap, mean, std);

      // Run inference
      Tensor outputTensor = module.forward(EValue.from(inputTensor))[0].toTensor();

      float tumourIOU = computeIoU(outputTensor);

      // Process the output tensor to create segmentation mask
      Bitmap outputBitmap = processOutput(outputTensor, resizedBitmap.getWidth(), resizedBitmap.getHeight(), resizedBitmap);

      // Update UI to indicate skin segmentation finished
      runOnUiThread(() -> {
        finalTextTimer.setText("Skin segmenting finished... Tumour segmenting finished...");
      });

      // Rotate back if needed
      if (rotated) {
        outputBitmap = rotateBitmap(outputBitmap, -90);
      }

      // Compute IoU
      float severityScore = tumourIOU / skinIOU * 100;

      // Store severity score
      severityScores.put(finalBodyPart, severityScore);

      // Compute average severity score
      float averageSeverityScore = computeAverageSeverityScore();

      // Compute elapsed time
      long elapsedTime = SystemClock.elapsedRealtime() - startTime;

      // Format elapsed time
      int seconds = (int) (elapsedTime / 1000) % 60;
      int minutes = (int) ((elapsedTime / (1000 * 60)) % 60);

      String timeString = String.format("Image was processed in %d minutes and %d seconds.", minutes, seconds);

      // Update the UI on the main thread
      Bitmap finalOutputBitmap = outputBitmap; // I added this david
      runOnUiThread(() -> {
        finalOutputImageView.setImageBitmap(finalOutputBitmap);
        finalTextIoU.setText(String.format("Severity Score: %.0f / 100", severityScore));
        finalTextTimer.setText(timeString);
        finalButtonProcess.setEnabled(true);
        finalButtonProcess.setText("View Output");

        // Update average severity score in UI
        averageSeverityTextView.setText(String.format("Average Severity Score: %.0f / 100", averageSeverityScore));
      });
    });
  }

  private Bitmap processOutput(Tensor outputTensor, int width, int height, Bitmap originalBitmap) {
    // Assuming the output tensor is a segmentation mask
    float[] scores = outputTensor.getDataAsFloatArray();
    long[] shape = outputTensor.shape();

    int numClasses = (int) shape[1];
    int outHeight = (int) shape[2];
    int outWidth = (int) shape[3];

    int[] maskPixels = new int[outWidth * outHeight];

    for (int y = 0; y < outHeight; y++) {
      for (int x = 0; x < outWidth; x++) {
        int maxClass = 0;
        float maxScore = Float.NEGATIVE_INFINITY;

        for (int c = 0; c < numClasses; c++) {
          int index = c * outWidth * outHeight + y * outWidth + x;
          float score = scores[index];
          if (score > maxScore) {
            maxScore = score;
            maxClass = c;
          }
        }

        if (maxClass == 1) {
          maskPixels[y * outWidth + x] = 0x80FF0000; // Semi-transparent red
        } else {
          maskPixels[y * outWidth + x] = 0x00000000; // Transparent
        }
      }
    }

    // Create bitmap from mask
    Bitmap maskBitmap = Bitmap.createBitmap(outWidth, outHeight, Bitmap.Config.ARGB_8888);
    maskBitmap.setPixels(maskPixels, 0, outWidth, 0, 0, outWidth, outHeight);

    // Resize mask to match original image size
    Bitmap resizedMask = Bitmap.createScaledBitmap(maskBitmap, width, height, true);

    // Overlay mask on original image
    Bitmap resultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int origPixel = originalBitmap.getPixel(x, y);
        int maskPixel = resizedMask.getPixel(x, y);
        int resultPixel = combinePixels(origPixel, maskPixel);
        resultBitmap.setPixel(x, y, resultPixel);
      }
    }

    return resultBitmap;
  }

  private Bitmap applyMaskToBitmap(Bitmap image, Bitmap mask) {
    int width = image.getWidth();
    int height = image.getHeight();

    Log.d("applyMaskToBitmap", "Image size: " + width + "x" + height);

    try {
      // Ensure both bitmaps are in ARGB_8888 format
      Bitmap imageARGB8888 = ensureBitmapFormat(image);
      Bitmap resizedMask = Bitmap.createScaledBitmap(mask, width, height, true);
      Bitmap maskARGB8888 = ensureBitmapFormat(resizedMask);

      int[] imagePixels = new int[width * height];
      int[] maskPixels = new int[width * height];
      int[] resultPixels = new int[width * height];

      // Get pixel data from bitmaps
      imageARGB8888.getPixels(imagePixels, 0, width, 0, 0, width, height);
      maskARGB8888.getPixels(maskPixels, 0, width, 0, 0, width, height);

      // Process pixels
      for (int i = 0; i < width * height; i++) {
        int maskPixel = maskPixels[i];
        int maskRed = (maskPixel >> 16) & 0xFF;

        if (maskRed > 128) {
          resultPixels[i] = imagePixels[i];
        } else {
          resultPixels[i] = 0xFF000000; // Black pixel
        }
      }

      // Create result bitmap
      Bitmap resultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
      resultBitmap.setPixels(resultPixels, 0, width, 0, 0, width, height);

      return resultBitmap;
    } catch (Exception e) {
      Log.e("applyMaskToBitmap", "Error applying mask to bitmap", e);
      return null;
    }
  }

  private Bitmap ensureBitmapFormat(Bitmap bitmap) {
    if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
      Bitmap newBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, false);
      if (newBitmap == null) {
        throw new IllegalArgumentException("Could not copy bitmap to ARGB_8888 format");
      }
      return newBitmap;
    }
    return bitmap;
  }

  private float computeAverageSeverityScore() {
    float total = 0.0f;
    int count = 0;

    for (Float score : severityScores.values()) {
      total += score;
      count++;
    }

    if (count == 0) {
      return 0.0f;
    } else {
      return total / count;
    }
  }

  private int combinePixels(int origPixel, int maskPixel) {
    int alpha = (maskPixel >> 24) & 0xFF;
    if (alpha == 0) {
      return origPixel;
    } else {
      int origRed = (origPixel >> 16) & 0xFF;
      int origGreen = (origPixel >> 8) & 0xFF;
      int origBlue = origPixel & 0xFF;

      int maskRed = (maskPixel >> 16) & 0xFF;
      int maskGreen = (maskPixel >> 8) & 0xFF;
      int maskBlue = maskPixel & 0xFF;

      int finalRed = (origRed + maskRed) / 2;
      int finalGreen = (origGreen + maskGreen) / 2;
      int finalBlue = (origBlue + maskBlue) / 2;

      return (0xFF << 24) | (finalRed << 16) | (finalGreen << 8) | finalBlue;
    }
  }

  private float computeIoU(Tensor outputTensor) {
    // Compute Intersection over Union
    float[] scores = outputTensor.getDataAsFloatArray();
    long[] shape = outputTensor.shape();
    int numClasses = (int) shape[1];
    int outHeight = (int) shape[2];
    int outWidth = (int) shape[3];

    int intersection = 0;
    int union = 0;

    for (int y = 0; y < outHeight; y++) {
      for (int x = 0; x < outWidth; x++) {
        int maxClass = 0;
        float maxScore = Float.NEGATIVE_INFINITY;

        for (int c = 0; c < numClasses; c++) {
          int index = c * outWidth * outHeight + y * outWidth + x;
          float score = scores[index];
          if (score > maxScore) {
            maxScore = score;
            maxClass = c;
          }
        }

        if (maxClass == 1) {
          intersection++;
        }
        union++;
      }
    }

    if (union == 0) {
      return 0.0f;
    } else {
      return (float) intersection / union;
    }
  }

  @Override
  public void onRequestPermissionsResult(int requestCode,
                                         @NonNull String[] permissions,
                                         @NonNull int[] grantResults) {
    if (requestCode == PERMISSION_REQUEST_CODE) {
      if (!(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
        Toast.makeText(this, "Permissions required", Toast.LENGTH_LONG).show();
        finish();
      }
    }
  }

  public static String assetFilePath(Activity activity, String assetName) throws IOException {
    // Same as provided in your code
    java.io.File file = new java.io.File(activity.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (java.io.InputStream is = activity.getAssets().open(assetName)) {
      try (java.io.OutputStream os = new java.io.FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    }
  }

  private Bitmap rotateBitmap(Bitmap source, float angle) {
    Matrix matrix = new Matrix();
    matrix.postRotate(angle);
    Bitmap rotated = Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    return rotated;
  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    if (executorService != null) {
      executorService.shutdown();
    }
  }
}
