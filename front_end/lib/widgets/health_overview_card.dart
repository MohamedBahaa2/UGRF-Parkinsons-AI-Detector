import 'package:flutter/material.dart';

class HealthOverviewCard extends StatelessWidget {
  final String title;
  final String value;

  // Constructor to accept title and value
  HealthOverviewCard({required this.title, required this.value});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 5,
      margin: EdgeInsets.symmetric(vertical: 10),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            // Title for the health metric
            Text(title, style: TextStyle(fontWeight: FontWeight.bold)),
            // Display the health metric value
            Text(
              value,
              style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }
}
