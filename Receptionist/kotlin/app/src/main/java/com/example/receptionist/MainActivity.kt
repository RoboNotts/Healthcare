package com.example.receptionist

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.robotemi.sdk.Robot
import com.robotemi.sdk.TtsRequest
import com.robotemi.sdk.Robot.AsrListener
import com.robotemi.sdk.listeners.OnRobotReadyListener
import com.robotemi.sdk.SttLanguage
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate

class MainActivity : AppCompatActivity(), OnRobotReadyListener, AsrListener {
    private lateinit var robot: Robot
    
    // YAMNet variables
    private var classifier: AudioClassifier? = null
    private var audioRecord: android.media.AudioRecord? = null
    private var classificationTimer: Timer? = null

    private val REQUEST_RECORD_AUDIO = 1337
    private val MODEL_FILE = "yamnet.tflite"
    private val DOORBELL_CLASSES = listOf("Doorbell", "Ding-dong", "Chime", "Knock", "Ding", "Buzzer")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        robot = Robot.getInstance()

        // Request microphone permission for YAMNet
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) 
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO)
        } else {
            startAudioClassification()
        }
    }

    override fun onStart() {
        super.onStart()
        robot.addOnRobotReadyListener(this)
        robot.addAsrListener(this)
    }

    override fun onStop() {
        robot.removeOnRobotReadyListener(this)
        robot.removeAsrListener(this)
        stopAudioClassification()
        super.onStop()
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) 
            == PackageManager.PERMISSION_GRANTED) {
            startAudioClassification()
        }
    }

    // --- Temi Repeating Logic ---

    override fun onRobotReady(isReady: Boolean) {
        if (isReady) {
            robot.hideTopBar()
            // Start the initial conversation
            robot.askQuestion("Hi there, I am listening for your voice and doorbells. Say something!")
        }
    }

    override fun onAsrResult(asrResult: String, sttLanguage: SttLanguage) {
        robot.finishConversation()
        val ttsRequest = TtsRequest.create("You said: $asrResult", true)
        robot.speak(ttsRequest)
        
        // After speaking, wait 3 seconds and listen again to keep the loop going
        Handler(Looper.getMainLooper()).postDelayed({
            robot.askQuestion("I'm listening again.")
        }, 3000)
    }

    // --- YAMNet Doorbell Logic ---

    private fun startAudioClassification() {
        if (classifier != null) return

        try {
            classifier = AudioClassifier.createFromFile(this, MODEL_FILE)
            val record = classifier!!.createAudioRecord()
            record.startRecording()
            audioRecord = record

            val audioTensor = classifier!!.createInputTensorAudio()

            classificationTimer = Timer()
            classificationTimer?.scheduleAtFixedRate(0, 500) {
                audioTensor.load(audioRecord)
                val results = classifier!!.classify(audioTensor)

                val topDoorbell = results[0].categories.find { 
                    it.label in DOORBELL_CLASSES && it.score > 0.3f 
                }

                if (topDoorbell != null) {
                    runOnUiThread {
                        Log.d("AudioClassification", "Detected: ${topDoorbell.label} (${topDoorbell.score})")
                        // Stop any ongoing conversation to announce the doorbell
                        robot.finishConversation()
                        val ttsRequest = TtsRequest.create("Excuse me, someone is at the door", true)
                        robot.speak(ttsRequest)
                    }
                }
            }
        } catch (e: Exception) {
            Log.e("AudioClassification", "Error initializing classifier", e)
        }
    }

    private fun stopAudioClassification() {
        classificationTimer?.cancel()
        classificationTimer = null
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        classifier?.close()
        classifier = null
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_RECORD_AUDIO && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startAudioClassification()
        }
    }
}
