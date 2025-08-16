import 'package:flutter/material.dart';

class UpdrsScreen extends StatefulWidget {
  @override
  _UpdrsScreenState createState() => _UpdrsScreenState();
}

class _UpdrsScreenState extends State<UpdrsScreen> {
  // Store answers to the questions (example of some questions from UPDRS)
  final _formKey = GlobalKey<FormState>();

  String? _question1Answer;
  String? _question2Answer;
  String? _question3Answer;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Parkinson UPDRS Test"),
        backgroundColor: Color.fromARGB(255, 101, 207, 213),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                "Please answer the following questions to evaluate Parkinson's disease severity.",
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 20),

              // Question 1: Tremor
              _buildQuestion(
                question: "1. Tremor in hands or arms",
                options: ["No Tremor", "Mild", "Moderate", "Severe"],
                onChanged: (value) {
                  setState(() {
                    _question1Answer = value;
                  });
                },
                selectedValue: _question1Answer,
              ),

              // Question 2: Speech
              _buildQuestion(
                question: "2. Speech (difficulty or slurred)",
                options: ["No Problem", "Mild", "Moderate", "Severe"],
                onChanged: (value) {
                  setState(() {
                    _question2Answer = value;
                  });
                },
                selectedValue: _question2Answer,
              ),

              // Question 3: Walking and Balance
              _buildQuestion(
                question: "3. Difficulty walking and balance issues",
                options: ["No Difficulty", "Mild", "Moderate", "Severe"],
                onChanged: (value) {
                  setState(() {
                    _question3Answer = value;
                  });
                },
                selectedValue: _question3Answer,
              ),

              // Submit Button
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: _submitForm,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Color.fromARGB(255, 101, 207, 213),
                  padding: EdgeInsets.symmetric(horizontal: 50, vertical: 15),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30),
                  ),
                  elevation: 5,
                ),
                child: Text(
                  "Submit Test",
                  style: TextStyle(color: Colors.white, fontSize: 18),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // Method to build individual questions
  Widget _buildQuestion({
    required String question,
    required List<String> options,
    required Function(String?) onChanged,
    String? selectedValue,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          question,
          style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
        ),
        SizedBox(height: 10),
        ...options.map((option) {
          return RadioListTile<String>(
            title: Text(option),
            value: option,
            groupValue: selectedValue,
            onChanged: onChanged,
          );
        }).toList(),
        SizedBox(height: 20),
      ],
    );
  }

  // Method to handle form submission
  void _submitForm() {
    if (_formKey.currentState!.validate()) {
      // Normally, you would send this data to a backend or save locally
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Test Submitted'),
            content: Text(
              'Your answers have been recorded.\n\n'
              'Tremor: $_question1Answer\n'
              'Speech: $_question2Answer\n'
              'Walking/Balance: $_question3Answer',
            ),
            actions: [
              TextButton(
                child: Text('OK'),
                onPressed: () {
                  Navigator.pop(context);
                },
              ),
            ],
          );
        },
      );
    }
  }
}
