import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

class MonitorScreen extends StatefulWidget {
  @override
  _MonitorScreenState createState() => _MonitorScreenState();
}

class _MonitorScreenState extends State<MonitorScreen> {
  File? _video; // Store the selected video file

  // Function to pick a video
  Future<void> _pickVideo() async {
    final ImagePicker _picker = ImagePicker();
    final XFile? video = await _picker.pickVideo(
      source: ImageSource.gallery,
    ); // Pick video from gallery

    if (video != null) {
      setState(() {
        _video = File(video.path); // Set the selected video
      });
    } else {
      // Handle if the user cancels video picking
      print("No video selected.");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Monitor Your Movement"),
        backgroundColor: Color.fromARGB(255, 101, 207, 213),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text(
              "Upload a video of yourself walking for gait analysis",
              textAlign: TextAlign.center,
            ),
            SizedBox(height: 20),
            // Video upload button
            ElevatedButton(
              onPressed: _pickVideo,
              style: ElevatedButton.styleFrom(
                backgroundColor: Color.fromARGB(255, 101, 207, 213),
                padding: EdgeInsets.symmetric(horizontal: 50, vertical: 15),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
                elevation: 5,
              ),
              child: Text(
                "Upload Video",
                style: TextStyle(color: Colors.white, fontSize: 18),
              ),
            ),
            SizedBox(height: 20),
            // Display the selected video
            _video == null
                ? Text("No video selected. Please upload a video.")
                : Text(
                    "Video Selected: ${_video!.path.split('/').last}",
                  ), // Show the file name
          ],
        ),
      ),
    );
  }
}
