import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart'; // For icons
import 'monitor_screen.dart'; // Import your MonitorScreen
import 'updrs_screen.dart'; // Import your UpdrsScreen

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            // Left: Welcome Message
            Text("Welcome, Mohamed", style: TextStyle(fontSize: 18)),
            // Right: Profile Picture
            CircleAvatar(
              radius: 20,
              backgroundImage: AssetImage(
                "images/Profile.jpeg",
              ), // Image path for profile picture
            ),
          ],
        ),
        backgroundColor: Color.fromARGB(255, 101, 207, 213),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16.0),
        child: Column(
          children: [
            SizedBox(height: 20),

            // Buttons for Monitor and UPDRS Test (Using Column instead of Stack)
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                buildButton(
                  icon: FontAwesomeIcons.walking, // Monitor Icon
                  title: "Monitor",
                  description: "Monitor your movement using gait analysis",
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => MonitorScreen(),
                      ), // Navigate to Monitor Screen
                    );
                  },
                ),
                SizedBox(height: 20), // Add space between buttons
                buildButton(
                  icon: FontAwesomeIcons.fileMedicalAlt, // UPDRS Icon
                  title: "Parkinson UPDRS Test",
                  description:
                      "Take the UPDRS test to evaluate Parkinson’s severity",
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => UpdrsScreen(),
                      ), // Navigate to UPDRS Test Screen
                    );
                  },
                ),
              ],
            ),

            SizedBox(height: 325), // Space before the bottom navigation bar
            // Bottom Navigation Bar aligned with screen edge
            BottomNavigationBar(
              backgroundColor: Colors.white,
              selectedItemColor: Color.fromARGB(
                255,
                101,
                207,
                213,
              ), // Blue theme color for selected icons
              unselectedItemColor: Colors.grey, // Grey for unselected icons
              showSelectedLabels: true,
              showUnselectedLabels: true,
              type: BottomNavigationBarType
                  .fixed, // Keeps the bar fixed at the bottom
              items: [
                BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
                BottomNavigationBarItem(
                  icon: Icon(Icons.data_usage),
                  label: 'Data',
                ),
                BottomNavigationBarItem(icon: Icon(Icons.help), label: 'Help'),
                BottomNavigationBarItem(
                  icon: Icon(Icons.account_circle),
                  label: 'Profile',
                ),
              ],
              onTap: (index) {
                // Disable navigation for Data, Help, and Profile
                switch (index) {
                  case 0:
                    // Home - Do nothing since it's already on Home screen
                    break;
                  case 1:
                    // Data - Do nothing
                    break;
                  case 2:
                    // Help - Do nothing
                    break;
                  case 3:
                    // Profile - Do nothing
                    break;
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  // Reusable Button Widget with Icon and Description
  Widget buildButton({
    required IconData icon,
    required String title,
    required String description,
    required Function onPressed,
  }) {
    return Card(
      margin: EdgeInsets.symmetric(vertical: 10, horizontal: 20),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
      elevation: 5,
      child: InkWell(
        onTap: () => onPressed(),
        child: Padding(
          padding: EdgeInsets.all(20),
          child: Row(
            children: [
              // Icon on the left
              Container(
                width: 50,
                height: 50,
                decoration: BoxDecoration(
                  color: Color.fromARGB(255, 101, 207, 213),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: Colors.white, size: 30),
              ),
              SizedBox(width: 20), // Space between icon and text
              // Title and Description
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 18,
                      ),
                    ),
                    Text(description, style: TextStyle(color: Colors.grey)),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
